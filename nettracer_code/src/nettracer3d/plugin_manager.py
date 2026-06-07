"""
NetTracer3D Plugin Manager
==========================

Provides a stable API for third-party plugins and handles discovery,
loading, lifecycle management, and conflict resolution.

Plugin authors should create a Python module/package that exposes:

    PLUGIN_INFO = {
        'name': 'My Plugin',
        'version': '1.0.0',
        'author': 'Author Name',
        'description': 'What it does',
        'api_version': (1, 0),          # minimum API version required
        'requires': [],                  # other plugin names this depends on
        'category': 'analysis',          # analysis | processing | visualization | io | other
    }

    def register(api: PluginAPI) -> None:
        # Called once when the plugin is loaded.
        # Use `api` to register menus, hooks, event listeners, etc.
        ...

    def unregister(api: PluginAPI) -> None:   # optional
        # Called when the plugin is disabled or the app is closing.
        ...

Plugins are discovered from:
  1. ~/.nettracer3d/plugins/            (drop-in .py files or packages)
  2. Python entry-point group           "nettracer3d.plugins"
  3. Paths listed in the env var        NETTRACER3D_PLUGIN_PATH  (colon-separated)
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import subprocess
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QPushButton, QTextEdit, QVBoxLayout, QWidget,
    QMessageBox, QSplitter, QGroupBox, QCheckBox, QApplication,
    QProgressBar, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QColor, QAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_VERSION: Tuple[int, int] = (1, 0)
ENTRY_POINT_GROUP = "nettracer3d.plugins"
DEFAULT_PLUGIN_DIR = Path.home() / ".nettracer3d" / "plugins"
CONFIG_FILE = Path.home() / ".nettracer3d" / "plugin_config.json"

logger = logging.getLogger("nettracer3d.plugins")


def is_frozen() -> bool:
    """Return True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def _detect_cuda_version() -> Optional[Tuple[int, int]]:
    """
    Try to detect the installed CUDA toolkit version.
    Returns (major, minor) or None.
    """
    # Method 1: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi reports CUDA version in its header output
            result2 = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5)
            import re
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)",
                              result2.stdout)
            if match:
                return (int(match.group(1)), int(match.group(2)))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            import re
            match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout)
            if match:
                return (int(match.group(1)), int(match.group(2)))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 3: check if torch is already installed with CUDA
    try:
        import torch
        if torch.cuda.is_available():
            ver = torch.version.cuda
            if ver:
                parts = ver.split(".")
                return (int(parts[0]), int(parts[1]))
    except ImportError:
        pass

    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PluginStatus(Enum):
    DISCOVERED = auto()
    LOADED = auto()
    FAILED = auto()
    DISABLED = auto()
    INCOMPATIBLE = auto()
    NEEDS_DEPS = auto()          # importable but has missing dependencies


@dataclass
class PluginRecord:
    """Everything the manager knows about one plugin."""
    name: str
    module_path: str                        # import path or file path
    status: PluginStatus = PluginStatus.DISCOVERED
    info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    module: Any = None                      # the loaded module object
    registered_menus: List[str] = field(default_factory=list)
    registered_hooks: List[str] = field(default_factory=list)
    registered_events: List[str] = field(default_factory=list)
    registered_view_items: list = field(default_factory=list)
    missing_deps: List[str] = field(default_factory=list)
    requirements_file: Optional[Path] = None


# ---------------------------------------------------------------------------
# Event bus (simple observer)
# ---------------------------------------------------------------------------

class _EventBus(QObject):
    """Central event dispatcher.  Plugins subscribe; the main app emits."""

    # Qt signal used only to marshal cross-thread calls onto the GUI thread.
    _dispatch_signal = pyqtSignal(str, object)

    # Events the app can emit (plugins listen)
    KNOWN_EVENTS = frozenset({
        "slice_changed",
        "selection_changed",
        "channel_loaded",
        "channel_deleted",
        "network_changed",
        "display_updated",
        "communities_changed",
        "identities_changed",
        "centroids_changed",
        "plugin_loaded",
        "plugin_unloaded",
        "session_loaded",
        "session_saved",
    })

    def __init__(self):
        super().__init__()
        self._listeners: Dict[str, List[Tuple[str, Callable]]] = {}
        self._dispatch_signal.connect(self._on_dispatch)

    # -- subscribe / unsubscribe ------------------------------------------

    def subscribe(self, event: str, callback: Callable, owner: str = ""):
        self._listeners.setdefault(event, []).append((owner, callback))

    def unsubscribe_owner(self, owner: str):
        for event in list(self._listeners):
            self._listeners[event] = [
                (o, cb) for o, cb in self._listeners[event] if o != owner
            ]

    # -- emit --------------------------------------------------------------

    def emit(self, event: str, data: Any = None):
        """Thread-safe emit.  Always dispatches on the GUI thread."""
        self._dispatch_signal.emit(event, data)

    def _on_dispatch(self, event: str, data: Any):
        for owner, cb in self._listeners.get(event, []):
            try:
                cb(data)
            except Exception:
                logger.warning(
                    "Event handler for '%s' (owner=%s) raised:\n%s",
                    event, owner, traceback.format_exc(),
                )


# ---------------------------------------------------------------------------
# Plugin API  —  the *only* object plugins should interact with
# ---------------------------------------------------------------------------

class PluginAPI:
    """
    Stable public interface handed to every plugin's ``register()`` call.

    All methods are intentionally high-level.  If you find yourself wanting
    direct access to Qt widgets or internal data structures, use
    ``get_unsafe_window()`` — but be aware that **anything** accessed
    through that handle may change or break between versions.
    """

    def __init__(self, main_window, network_ref, event_bus: _EventBus):
        self._win = main_window
        self._net = network_ref          # the global ``my_network`` object
        self._bus = event_bus
        self._caller: Optional[str] = None   # set by the manager during register()

    # -- version info -------------------------------------------------------

    @property
    def api_version(self) -> Tuple[int, int]:
        return API_VERSION

    # ======================================================================
    #  DATA ACCESS  — read
    # ======================================================================

    def get_channel_data(self, index: int) -> Optional[np.ndarray]:
        """Return a reference to channel data (0-3).  May be None."""
        if 0 <= index <= 3:
            return self._win.channel_data[index]
        return None

    def get_channel_names(self) -> List[str]:
        return list(self._win.channel_names)

    def get_active_channel(self) -> int:
        return self._win.active_channel

    def get_current_slice(self) -> int:
        return self._win.current_slice

    def get_shape(self) -> Optional[Tuple[int, ...]]:
        return self._win.shape

    def get_selection(self) -> Dict[str, list]:
        """Return a copy of the current clicked_values."""
        import copy
        return copy.deepcopy(self._win.clicked_values)

    def get_highlight_overlay(self) -> Optional[np.ndarray]:
        return self._win.highlight_overlay

    # -- network properties ------------------------------------------------

    def get_network(self):
        """Return the networkx Graph (or None)."""
        return self._net.network

    def get_network_lists(self):
        return self._net.network_lists

    def get_node_centroids(self) -> Optional[dict]:
        return self._net.node_centroids

    def get_edge_centroids(self) -> Optional[dict]:
        return self._net.edge_centroids

    def get_node_identities(self) -> Optional[dict]:
        return self._net.node_identities

    def get_communities(self) -> Optional[dict]:
        return self._net.communities

    def get_xy_scale(self) -> float:
        return self._net.xy_scale

    def get_z_scale(self) -> float:
        return self._net.z_scale

    # ======================================================================
    #  DATA ACCESS  — write  (validated & triggers display refresh)
    # ======================================================================

    def set_channel_data(self, index: int, data: np.ndarray,
                         assign_shape: bool = True):
        """
        Replace a channel's data.  Handles shape validation, undo snapshot,
        button/slider state, and display refresh.
        """
        if not 0 <= index <= 3:
            raise ValueError(f"Channel index must be 0-3, got {index}")
        self._win.load_channel(index, data, data=True, assign_shape=assign_shape)

    def set_highlight(self, node_indices: list = None,
                      edge_indices: list = None):
        """Update the highlight overlay from node/edge index lists."""
        self._win.clicked_values['nodes'] = node_indices or []
        self._win.clicked_values['edges'] = edge_indices or []
        self._win.evaluate_mini(subgraph_push=True)

    def set_communities(self, communities: dict):
        self._net.communities = communities
        self._win.update_graph_fields()
        self._bus.emit("communities_changed", communities)

    def set_node_identities(self, identities: dict):
        self._net.node_identities = identities
        self._win.update_graph_fields()
        self._bus.emit("identities_changed", identities)

    def set_node_centroids(self, centroids: dict):
        self._net.node_centroids = centroids
        self._win.update_graph_fields()
        self._bus.emit("centroids_changed", centroids)

    def set_xy_scale(self, value: float):
        self._net.xy_scale = value
        self._win.xy_scale_label.setText(
            f"xy_scale: {value:.2e}                   ")

    def set_z_scale(self, value: float):
        self._net.z_scale = value
        self._win.z_scale_label.setText(
            f"z_scale: {value:.2e}                   ")

    # ======================================================================
    #  UI OUTPUT
    # ======================================================================

    def add_table(self, title: str, dataframe: pd.DataFrame,
                  sort: bool = True):
        """Add a table tab to the upper-right tabbed data widget."""
        return self._win.format_for_upperright_table(
            data=dataframe.to_dict('list') if isinstance(dataframe, pd.DataFrame) else dataframe,
            title=title, sort=sort,
        )

    def add_table_from_dict(self, data: dict, metric: str = 'Metric',
                            value: str = 'Value', title: str = None,
                            sort: bool = True):
        """Convenience: format a dict directly into a table tab."""
        return self._win.format_for_upperright_table(
            data=data, metric=metric, value=value, title=title, sort=sort,
        )

    def add_widget_tab(self, title: str, widget: QWidget):
        """Add an arbitrary QWidget as a tab in the upper-right panel."""
        self._win.tabbed_data.add_table(title, widget)

    def show_message(self, title: str, text: str,
                     level: str = "info"):
        """Show a message box.  level: 'info', 'warning', 'error'."""
        funcs = {
            "info": QMessageBox.information,
            "warning": QMessageBox.warning,
            "error": QMessageBox.critical,
        }
        funcs.get(level, QMessageBox.information)(self._win, title, text)

    def print(self, msg: str):
        """Print to console (future: status bar)."""
        print(f"[plugin] {msg}")

    # ======================================================================
    #  MENU REGISTRATION
    # ======================================================================

    def register_menu_action(self, menu_path: str, callback: Callable,
                             tooltip: str = ""):
        """
        Add a menu item.  ``menu_path`` uses ``/`` separators, e.g.
        ``"Extensions/My Plugin/Run Analysis"``.

        The first component should usually be ``"Extensions"``.
        """
        parts = menu_path.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(
                "menu_path needs at least 'Menu/Action', got: " + menu_path)

        menubar = self._win.menuBar()
        current_menu = None

        # Walk / create intermediate menus
        for part in parts[:-1]:
            found = False
            source = menubar if current_menu is None else current_menu
            for action in source.actions():
                if action.menu() and action.text() == part:
                    current_menu = action.menu()
                    found = True
                    break
            if not found:
                if current_menu is None:
                    current_menu = menubar.addMenu(part)
                else:
                    current_menu = current_menu.addMenu(part)

        action = current_menu.addAction(parts[-1])
        action.triggered.connect(callback)
        if tooltip:
            action.setToolTip(tooltip)

        # Track for conflict reporting
        if self._caller:
            mgr = self._win._plugin_manager
            if self._caller in mgr._plugins:
                mgr._plugins[self._caller].registered_menus.append(menu_path)

    def register_context_action(self, label: str, callback: Callable):
        """
        Register an extra entry that appears in the right-click context
        menu on the image canvas.

        ``callback`` receives a dict ``{'x': int, 'y': int, 'z': int}``.
        """
        if not hasattr(self._win, '_plugin_context_actions'):
            self._win._plugin_context_actions = []
        self._win._plugin_context_actions.append((label, callback))

    # ======================================================================
    #  DISPLAY HOOKS
    # ======================================================================

    def register_display_hook(self, callback: Callable):
        """
        Register a function called at the end of every ``update_display``.

        Signature:  ``callback(view, current_slice, view_range)``
        where ``view`` is the pyqtgraph ViewBox.
        """
        if not hasattr(self._win, '_plugin_display_hooks'):
            self._win._plugin_display_hooks = []
        self._win._plugin_display_hooks.append(callback)

        if self._caller:
            mgr = self._win._plugin_manager
            if self._caller in mgr._plugins:
                mgr._plugins[self._caller].registered_hooks.append(
                    "display_hook")

    def add_view_item(self, item) -> None:
        """Add a pyqtgraph graphics item to the image view."""
        self._win.view.addItem(item)
        if self._caller:
            mgr = self._win._plugin_manager
            if self._caller in mgr._plugins:
                mgr._plugins[self._caller].registered_view_items.append(item)

    def remove_view_item(self, item) -> None:
        """Remove a previously added pyqtgraph item."""
        try:
            self._win.view.removeItem(item)
        except Exception:
            pass

    # ======================================================================
    #  EVENTS
    # ======================================================================

    def on(self, event: str, callback: Callable):
        """Subscribe to an application event.  See _EventBus.KNOWN_EVENTS."""
        owner = self._caller or ""
        self._bus.subscribe(event, callback, owner=owner)

        if self._caller:
            mgr = self._win._plugin_manager
            if self._caller in mgr._plugins:
                mgr._plugins[self._caller].registered_events.append(event)

    # ======================================================================
    #  UTILITIES
    # ======================================================================

    def refresh_display(self):
        """Request a full display redraw."""
        self._win.update_display()

    def navigate_to_slice(self, z: int):
        """Change the current Z slice."""
        self._win.slice_slider.setValue(z)

    def get_visible_channels(self) -> List[int]:
        """Return indices of currently visible channels."""
        return [i for i in range(4) if self._win.channel_visible[i]]

    # ======================================================================
    #  UNSAFE ESCAPE HATCH
    # ======================================================================

    def get_unsafe_window(self):
        """
        Return a direct reference to the main ``ImageViewerWindow``.

        .. warning::

            **This is not part of the stable API.**  Any attribute, method,
            or internal data structure you access through this reference may
            be renamed, removed, or restructured in *any* future release
            without notice.  Use this only for rapid prototyping or for
            accessing functionality not yet exposed by the public API.
            Published plugins that depend on internals obtained this way
            may break without warning.
        """
        #warnings.warn(
        #    "get_unsafe_window() exposes unstable internals. "
        #    "Anything accessed through it may break between versions.",
        #    stacklevel=2,
        #)
        return self._win

    def get_unsafe_network(self):
        """
        Return a direct reference to the ``my_network`` Network_3D object.

        Same caveats as ``get_unsafe_window()``.
        """
        warnings.warn(
            "get_unsafe_network() exposes unstable internals. "
            "Anything accessed through it may break between versions.",
            stacklevel=2,
        )
        return self._net


# ---------------------------------------------------------------------------
# Plugin Manager
# ---------------------------------------------------------------------------

class PluginManager:
    """
    Discovers, loads, and manages the lifecycle of all plugins.

    Typical usage inside the main window's ``__init__``::

        from . import plugin_manager
        self._plugin_manager = plugin_manager.PluginManager(self, my_network)
        self._plugin_manager.discover()
        self._plugin_manager.load_all()
    """

    def __init__(self, main_window, network_ref):
        self._win = main_window
        self._net = network_ref
        self._bus = _EventBus()
        self._plugins: Dict[str, PluginRecord] = {}
        self._load_order: List[str] = []
        self._disabled: Set[str] = set()

        # Create the API instance (shared — caller is swapped per-plugin)
        self.api = PluginAPI(main_window, network_ref, self._bus)

        # Ensure user plugin directory exists
        try:
            DEFAULT_PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # read-only filesystem, etc.

        # Resolve package-relative plugin directory
        # e.g.  .../site-packages/nettracer3d/plugins/
        self._package_plugin_dir = self._find_package_plugin_dir()

        # Load disabled-plugin list from config
        self._load_config()

    @staticmethod
    def _find_package_plugin_dir() -> Optional[Path]:
        """
        Locate a ``plugins/`` folder next to the installed nettracer3d
        package.  Returns None if the package can't be found.
        """
        try:
            # Try using the package's own __file__
            import nettracer3d
            pkg_dir = Path(nettracer3d.__file__).resolve().parent
            candidate = pkg_dir / "plugins"
            return candidate
        except Exception:
            pass

        # Fallback: walk sys.path looking for a nettracer3d directory
        for p in sys.path:
            candidate = Path(p) / "nettracer3d" / "plugins"
            if candidate.is_dir():
                return candidate

        return None

    # ------------------------------------------------------------------
    #  Configuration persistence
    # ------------------------------------------------------------------

    def _load_config(self):
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE) as f:
                    cfg = json.load(f)
                self._disabled = set(cfg.get("disabled", []))
        except Exception:
            self._disabled = set()

    def _save_config(self):
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump({"disabled": sorted(self._disabled)}, f, indent=2)

    # ------------------------------------------------------------------
    #  Discovery
    # ------------------------------------------------------------------

    def discover(self):
        """Scan all sources for plugin candidates."""
        scan_locations = []

        # 1. User home directory:  ~/.nettracer3d/plugins/
        self._discover_directory(DEFAULT_PLUGIN_DIR)
        scan_locations.append(str(DEFAULT_PLUGIN_DIR))

        # 2. Package-relative:  .../site-packages/nettracer3d/plugins/
        if self._package_plugin_dir is not None:
            self._discover_directory(self._package_plugin_dir)
            scan_locations.append(str(self._package_plugin_dir))

        # 3. NETTRACER3D_PLUGIN_PATH environment variable
        self._discover_env_paths()
        env = os.environ.get("NETTRACER3D_PLUGIN_PATH", "")
        if env:
            scan_locations.extend(env.split(os.pathsep))

        # 4. Pip entry points (nettracer3d.plugins group)
        self._discover_entry_points()

        # Report what was found
        print(f"[Plugins] Scanned directories:")
        for loc in scan_locations:
            exists = Path(loc).is_dir()
            marker = "  ✓" if exists else "  ✗ (not found)"
            print(f"  {loc}{marker}")
        print(f"[Plugins] Discovered {len(self._plugins)} plugin(s)"
              + (f": {', '.join(self._plugins.keys())}"
                 if self._plugins else ""))

    def _discover_directory(self, directory: Path):
        """Scan a directory for .py files and sub-packages."""
        if not directory.is_dir():
            return

        for item in sorted(directory.iterdir()):
            if item.name.startswith(("_", ".")):
                continue

            if item.is_file() and item.suffix == ".py":
                name = item.stem
                self._plugins.setdefault(name, PluginRecord(
                    name=name, module_path=str(item)))

            elif item.is_dir() and (item / "__init__.py").exists():
                name = item.name
                self._plugins.setdefault(name, PluginRecord(
                    name=name, module_path=str(item)))

    def _discover_env_paths(self):
        """Check NETTRACER3D_PLUGIN_PATH environment variable."""
        env = os.environ.get("NETTRACER3D_PLUGIN_PATH", "")
        if not env:
            return
        for p in env.split(os.pathsep):
            path = Path(p)
            if path.is_dir():
                self._discover_directory(path)

    def _discover_entry_points(self):
        """Check installed packages for the nettracer3d.plugins entry point."""
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups / dict;
            # earlier versions return a dict of lists.
            if hasattr(eps, "select"):
                group = eps.select(group=ENTRY_POINT_GROUP)
            else:
                group = eps.get(ENTRY_POINT_GROUP, [])

            for ep in group:
                name = ep.name
                self._plugins.setdefault(name, PluginRecord(
                    name=name, module_path=ep.value))
        except Exception:
            logger.debug("Entry point discovery failed:\n%s",
                         traceback.format_exc())

    # ------------------------------------------------------------------
    #  Loading
    # ------------------------------------------------------------------

    def load_all(self):
        """Load every discovered plugin, respecting disabled list and deps."""
        # Sort by dependency order (simple: no deps first)
        ordered = self._topological_sort()

        for name in ordered:
            if name in self._disabled:
                self._plugins[name].status = PluginStatus.DISABLED
                continue
            self._load_one(name)

    def _load_one(self, name: str) -> bool:
        rec = self._plugins.get(name)
        if rec is None:
            return False
        if rec.status == PluginStatus.LOADED:
            return True

        try:
            # ── Resolve plugin directory and inject vendor deps ──────
            plugin_dir = self._get_plugin_dir(rec)
            if plugin_dir:
                rec.requirements_file = self._find_requirements(plugin_dir)
                self._inject_vendor_path(plugin_dir)

            # ── Try importing ────────────────────────────────────────
            try:
                module = self._import_plugin(rec)
            except ImportError as imp_err:
                # Import failed — check if there's a requirements file
                missing_module = getattr(imp_err, "name", None) or str(imp_err)
                
                if rec.requirements_file and not is_frozen():
                    # pip environment: mark as needing deps
                    missing = self._check_requirements(rec.requirements_file)
                    rec.missing_deps = missing if missing else [missing_module]
                    rec.status = PluginStatus.NEEDS_DEPS
                    rec.error = (
                        f"Import failed: {imp_err}\n\n"
                        f"Missing packages: {', '.join(rec.missing_deps)}\n"
                        f"Use the Extensions panel to install dependencies.")
                    print(f"[Plugins] '{name}' needs dependencies: "
                          f"{', '.join(rec.missing_deps)}")
                    return False

                elif is_frozen():
                    # PyInstaller: vendor path didn't help
                    rec.status = PluginStatus.FAILED
                    rec.error = (
                        f"Import failed: {imp_err}\n\n"
                        f"This is a compiled (PyInstaller) environment — "
                        f"pip install is not available.\n"
                        f"Please download the bundled version of this plugin "
                        f"that includes a '_vendor/' folder with pre-compiled "
                        f"dependencies.")
                    return False
                else:
                    raise  # re-raise for the outer except to handle

            rec.module = module

            # Read PLUGIN_INFO
            info = getattr(module, "PLUGIN_INFO", {})
            rec.info = info
            if not info.get("name"):
                info["name"] = name

            # Check API version compatibility
            required_api = info.get("api_version", (1, 0))
            if required_api[0] > API_VERSION[0]:
                rec.status = PluginStatus.INCOMPATIBLE
                rec.error = (
                    f"Requires API v{required_api[0]}.{required_api[1]}, "
                    f"but current API is v{API_VERSION[0]}.{API_VERSION[1]}")
                logger.warning("Plugin '%s' incompatible: %s", name, rec.error)
                return False

            # Check inter-plugin dependencies
            for dep in info.get("requires", []):
                if dep not in self._plugins:
                    rec.status = PluginStatus.FAILED
                    rec.error = f"Missing required plugin: '{dep}'"
                    logger.warning("Plugin '%s' missing dep '%s'", name, dep)
                    return False
                if self._plugins[dep].status != PluginStatus.LOADED:
                    # Try loading the dependency first
                    if not self._load_one(dep):
                        rec.status = PluginStatus.FAILED
                        rec.error = f"Dependency '{dep}' failed to load"
                        return False

            # Call register()
            register_fn = getattr(module, "register", None)
            if register_fn is None:
                rec.status = PluginStatus.FAILED
                rec.error = "Module has no register(api) function"
                return False

            self.api._caller = name
            register_fn(self.api)
            self.api._caller = None

            rec.status = PluginStatus.LOADED
            rec.missing_deps = []
            self._load_order.append(name)
            logger.info("Loaded plugin '%s'", name)
            self._bus.emit("plugin_loaded", name)
            return True

        except Exception:
            rec.status = PluginStatus.FAILED
            rec.error = traceback.format_exc()
            self.api._caller = None
            logger.error("Failed to load plugin '%s':\n%s", name, rec.error)
            return False

    # ------------------------------------------------------------------
    #  Dependency management
    # ------------------------------------------------------------------

    @staticmethod
    def _get_plugin_dir(rec: PluginRecord) -> Optional[Path]:
        """Return the directory containing a plugin's files."""
        path = Path(rec.module_path)
        if path.is_dir():
            return path                     # package-style plugin
        if path.is_file():
            return path.parent              # single-file plugin
        return None

    @staticmethod
    def _find_requirements(plugin_dir: Path) -> Optional[Path]:
        """Look for a requirements file in the plugin's directory."""
        for name in ("requirements.txt", "plugin_requirements.txt"):
            candidate = plugin_dir / name
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _inject_vendor_path(plugin_dir: Path):
        """
        If the plugin ships a ``_vendor/`` folder (pre-bundled deps for
        PyInstaller), prepend it to sys.path so imports resolve there.
        """
        vendor = plugin_dir / "_vendor"
        if vendor.is_dir():
            vendor_str = str(vendor)
            if vendor_str not in sys.path:
                sys.path.insert(0, vendor_str)
                print(f"[Plugins] Injected vendor path: {vendor_str}")

    @staticmethod
    def _check_requirements(req_file: Path) -> List[str]:
        """
        Parse a requirements.txt and return package names that are NOT
        currently installed.  Skips comments, blank lines, and pip flags
        like ``--extra-index-url``.
        """
        from importlib.metadata import distributions

        installed = {d.metadata["Name"].lower().replace("-", "_")
                     for d in distributions()}

        missing = []
        for raw_line in req_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Strip version specifiers:  cellpose>=3.0  →  cellpose
            pkg = line.split(">=")[0].split("<=")[0].split("==")[0] \
                      .split("!=")[0].split("<")[0].split(">")[0] \
                      .split(";")[0].split("[")[0].strip()
            if not pkg:
                continue
            if pkg.lower().replace("-", "_") not in installed:
                missing.append(line)  # keep original spec for pip
        return missing

    def install_plugin_deps(self, name: str, parent_widget=None,
                            gpu_option: str = "auto") -> bool:
        """
        Attempt to pip-install a plugin's dependencies.

        Parameters
        ----------
        name : str
            Plugin name.
        parent_widget : QWidget, optional
            Parent for message boxes.
        gpu_option : str
            "auto" — try to detect CUDA,
            "cpu"  — force CPU-only torch,
            "cu118" / "cu121" / "cu124" — specific CUDA version.

        Returns True if installation succeeded and the plugin loaded.
        """
        rec = self._plugins.get(name)
        if rec is None or rec.requirements_file is None:
            return False

        if is_frozen():
            QMessageBox.warning(
                parent_widget, "Cannot Install",
                "This is a compiled (PyInstaller) build.\n"
                "pip is not available — please use the bundled version "
                "of this plugin that includes pre-compiled dependencies "
                "in a '_vendor/' folder.")
            return False

        req_file = rec.requirements_file

        # Build pip command
        cmd = [sys.executable, "-m", "pip", "install",
               "-r", str(req_file)]

        # Handle torch/GPU index URL
        torch_url = self._resolve_torch_index(gpu_option)
        if torch_url:
            cmd.extend(["--extra-index-url", torch_url])

        # Confirm with user
        req_text = req_file.read_text(encoding="utf-8")
        msg = (f"The following will be installed for plugin "
               f"'{rec.info.get('name', name)}':\n\n"
               f"{req_text}\n\n"
               f"Command:\n{' '.join(cmd)}\n\n"
               f"Proceed?")

        reply = QMessageBox.question(
            parent_widget, "Install Plugin Dependencies", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply != QMessageBox.StandardButton.Yes:
            return False

        # Run pip
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                QMessageBox.critical(
                    parent_widget, "Installation Failed",
                    f"pip returned exit code {result.returncode}:\n\n"
                    f"{result.stderr[-2000:]}")
                rec.error = result.stderr
                return False

            print(f"[Plugins] pip install succeeded for '{name}'")

            # Clear cached module and retry load
            rec.status = PluginStatus.DISCOVERED
            rec.module = None
            rec.missing_deps = []
            rec.error = None
            if name in sys.modules:
                del sys.modules[name]

            success = self._load_one(name)
            if success:
                QMessageBox.information(
                    parent_widget, "Success",
                    f"Dependencies installed and plugin "
                    f"'{rec.info.get('name', name)}' loaded successfully.")
            return success

        except subprocess.TimeoutExpired:
            QMessageBox.critical(
                parent_widget, "Timeout",
                "pip install timed out after 10 minutes.")
            return False
        except Exception as exc:
            QMessageBox.critical(
                parent_widget, "Error",
                f"Failed to run pip:\n{exc}")
            return False

    @staticmethod
    def _resolve_torch_index(gpu_option: str) -> Optional[str]:
        """
        Return the PyTorch ``--extra-index-url`` for the given option,
        or None if not needed.
        """
        base = "https://download.pytorch.org/whl"
        mapping = {
            "cpu":   f"{base}/cpu",
            "cu118": f"{base}/cu118",
            "cu121": f"{base}/cu121",
            "cu124": f"{base}/cu124",
            "cu126": f"{base}/cu126",
        }

        if gpu_option in mapping:
            return mapping[gpu_option]

        if gpu_option == "auto":
            # Try to detect CUDA
            cuda_ver = _detect_cuda_version()
            if cuda_ver:
                # Map detected version to closest supported wheel
                major, minor = cuda_ver
                if major >= 12 and minor >= 4:
                    return mapping["cu124"]
                elif major >= 12:
                    return mapping["cu121"]
                elif major >= 11 and minor >= 8:
                    return mapping["cu118"]
            # No CUDA detected or unrecognised — let pip default
            return None

        return None

    def _import_plugin(self, rec: PluginRecord):
        """Import a plugin module from its path or dotted name."""
        path = rec.module_path

        # File path → use spec_from_file_location
        if os.path.exists(path):
            p = Path(path)
            if p.is_dir():
                # Package
                init = p / "__init__.py"
                spec = importlib.util.spec_from_file_location(
                    rec.name, str(init),
                    submodule_search_locations=[str(p)])
            else:
                spec = importlib.util.spec_from_file_location(rec.name, path)

            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create module spec for {path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[rec.name] = module
            spec.loader.exec_module(module)
            return module

        # Dotted path → normal import (entry-point style, e.g. "pkg.mod:attr")
        if ":" in path:
            mod_path, attr = path.split(":", 1)
            mod = importlib.import_module(mod_path)
            return getattr(mod, attr)

        return importlib.import_module(path)

    def _topological_sort(self) -> List[str]:
        """Simple topological sort based on 'requires' fields."""
        visited: Set[str] = set()
        order: List[str] = []

        def visit(name):
            if name in visited:
                return
            visited.add(name)
            rec = self._plugins.get(name)
            if rec:
                info = getattr(rec.module, "PLUGIN_INFO", rec.info) or {}
                for dep in info.get("requires", []):
                    visit(dep)
            order.append(name)

        for name in self._plugins:
            visit(name)
        return order

    # ------------------------------------------------------------------
    #  Unloading / toggling
    # ------------------------------------------------------------------

    def unload_one(self, name: str):
        rec = self._plugins.get(name)
        if rec is None or rec.status != PluginStatus.LOADED:
            return

        # Call unregister() if available
        if rec.module:
            unregister_fn = getattr(rec.module, "unregister", None)
            if unregister_fn:
                try:
                    self.api._caller = name
                    unregister_fn(self.api)
                    self.api._caller = None
                except Exception:
                    logger.warning("unregister() for '%s' raised:\n%s",
                                   name, traceback.format_exc())

        # Clean up view items this plugin added
        for item in rec.registered_view_items:
            try:
                self._win.view.removeItem(item)
            except Exception:
                pass

        # Unsubscribe all event listeners owned by this plugin
        self._bus.unsubscribe_owner(name)

        rec.status = PluginStatus.DISABLED
        rec.registered_menus.clear()
        rec.registered_hooks.clear()
        rec.registered_events.clear()
        rec.registered_view_items.clear()
        self._bus.emit("plugin_unloaded", name)

    def disable_plugin(self, name: str):
        self.unload_one(name)
        self._disabled.add(name)
        self._plugins[name].status = PluginStatus.DISABLED
        self._save_config()

    def enable_plugin(self, name: str):
        self._disabled.discard(name)
        self._save_config()
        self._load_one(name)

    def reload_plugin(self, name: str):
        """Unload then re-import and re-load a plugin (for development)."""
        self.unload_one(name)
        rec = self._plugins.get(name)
        if rec and rec.module:
            # Remove from sys.modules to force re-import
            mod_name = rec.module.__name__
            sys.modules.pop(mod_name, None)
            rec.module = None
        self._load_one(name)

    # ------------------------------------------------------------------
    #  Event emission helpers (called from main app code)
    # ------------------------------------------------------------------

    def emit(self, event: str, data: Any = None):
        self._bus.emit(event, data)

    # ------------------------------------------------------------------
    #  Introspection
    # ------------------------------------------------------------------

    def get_all_plugins(self) -> Dict[str, PluginRecord]:
        return dict(self._plugins)

    def get_loaded_plugins(self) -> List[str]:
        return [n for n, r in self._plugins.items()
                if r.status == PluginStatus.LOADED]

    def get_failed_plugins(self) -> Dict[str, str]:
        return {n: r.error for n, r in self._plugins.items()
                if r.status == PluginStatus.FAILED and r.error}


# ---------------------------------------------------------------------------
#  Extensions Panel  (UI for managing plugins)
# ---------------------------------------------------------------------------

class ExtensionsPanel(QDialog):
    """
    Dialog that lists all discovered plugins, their status, and lets the
    user enable / disable / reload them.
    """

    def __init__(self, manager: PluginManager, parent=None):
        super().__init__(parent)
        self.mgr = manager
        self.setWindowTitle("Extensions")
        self.setMinimumSize(700, 450)
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: plugin list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.plugin_list = QListWidget()
        self.plugin_list.currentItemChanged.connect(self._on_selection)
        left_layout.addWidget(self.plugin_list)

        btn_row = QHBoxLayout()
        self.btn_enable = QPushButton("Enable")
        self.btn_disable = QPushButton("Disable")
        self.btn_install = QPushButton("Install Deps")
        self.btn_reload = QPushButton("Reload")
        self.btn_refresh = QPushButton("Rescan")
        self.btn_enable.clicked.connect(self._enable)
        self.btn_disable.clicked.connect(self._disable)
        self.btn_install.clicked.connect(self._install_deps)
        self.btn_reload.clicked.connect(self._reload)
        self.btn_refresh.clicked.connect(self._rescan)
        btn_row.addWidget(self.btn_enable)
        btn_row.addWidget(self.btn_disable)
        btn_row.addWidget(self.btn_install)
        btn_row.addWidget(self.btn_reload)
        btn_row.addWidget(self.btn_refresh)
        left_layout.addLayout(btn_row)

        # Show plugin directory paths so users know where to put files
        dirs_box = QGroupBox("Plugin Directories")
        dirs_layout = QVBoxLayout(dirs_box)
        dirs_layout.setContentsMargins(6, 6, 6, 6)

        user_dir = str(DEFAULT_PLUGIN_DIR)
        user_exists = DEFAULT_PLUGIN_DIR.is_dir()
        user_label = QLabel(
            f"<b>User plugins:</b><br>"
            f"<code>{user_dir}</code>"
            f"{'  ✓' if user_exists else '  (will be created)'}")
        user_label.setWordWrap(True)
        user_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        dirs_layout.addWidget(user_label)

        pkg_dir = self.mgr._package_plugin_dir
        if pkg_dir is not None:
            pkg_exists = pkg_dir.is_dir()
            pkg_label = QLabel(
                f"<b>Package plugins:</b><br>"
                f"<code>{pkg_dir}</code>"
                f"{'  ✓' if pkg_exists else '  (create this folder to use)'}")
            pkg_label.setWordWrap(True)
            pkg_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            dirs_layout.addWidget(pkg_label)

        open_dir_btn = QPushButton("Open User Plugin Folder")
        open_dir_btn.clicked.connect(
            lambda: self._open_folder(DEFAULT_PLUGIN_DIR))
        dirs_layout.addWidget(open_dir_btn)

        left_layout.addWidget(dirs_box)

        # Right: detail view
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.detail_label = QLabel("Select a plugin")
        self.detail_label.setWordWrap(True)
        right_layout.addWidget(self.detail_label)

        self.error_box = QGroupBox("Error Log")
        error_layout = QVBoxLayout(self.error_box)
        self.error_text = QTextEdit()
        self.error_text.setReadOnly(True)
        error_layout.addWidget(self.error_text)
        self.error_box.hide()
        right_layout.addWidget(self.error_box)

        self.registrations_box = QGroupBox("Registrations")
        reg_layout = QVBoxLayout(self.registrations_box)
        self.reg_text = QTextEdit()
        self.reg_text.setReadOnly(True)
        reg_layout.addWidget(self.reg_text)
        self.registrations_box.hide()
        right_layout.addWidget(self.registrations_box)

        right_layout.addStretch()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

    # -- list management ---------------------------------------------------

    def _refresh_list(self):
        self.plugin_list.clear()
        for name, rec in sorted(self.mgr.get_all_plugins().items()):
            item = QListWidgetItem(self._format_item_text(name, rec))
            item.setData(Qt.ItemDataRole.UserRole, name)

            # Color-code by status
            colors = {
                PluginStatus.LOADED: QColor(180, 255, 180),
                PluginStatus.FAILED: QColor(255, 180, 180),
                PluginStatus.DISABLED: QColor(220, 220, 220),
                PluginStatus.INCOMPATIBLE: QColor(255, 220, 180),
                PluginStatus.DISCOVERED: QColor(255, 255, 220),
                PluginStatus.NEEDS_DEPS: QColor(180, 200, 255),
            }
            item.setBackground(colors.get(rec.status, QColor(255, 255, 255)))
            self.plugin_list.addItem(item)

    def _format_item_text(self, name, rec):
        status = rec.status.name.lower().replace("_", " ").title()
        display_name = rec.info.get("name", name)
        version = rec.info.get("version", "")
        ver_str = f" v{version}" if version else ""
        return f"{display_name}{ver_str}  [{status}]"

    # -- detail view -------------------------------------------------------

    def _on_selection(self, current, previous):
        if current is None:
            return
        name = current.data(Qt.ItemDataRole.UserRole)
        rec = self.mgr.get_all_plugins().get(name)
        if rec is None:
            return

        info = rec.info or {}
        lines = [
            f"<b>{info.get('name', name)}</b>",
            f"Version: {info.get('version', 'N/A')}",
            f"Author: {info.get('author', 'N/A')}",
            f"Category: {info.get('category', 'N/A')}",
            f"Status: {rec.status.name}",
            f"<br>{info.get('description', '')}",
            f"<br>Source: <code>{rec.module_path}</code>",
        ]
        if rec.missing_deps:
            lines.append(
                f"<br><b style='color: #0066cc;'>Missing packages:</b> "
                f"{', '.join(rec.missing_deps)}")
            if not is_frozen():
                lines.append(
                    "<i>Click 'Install Deps' to install automatically.</i>")
            else:
                lines.append(
                    "<i>Compiled build — needs bundled '_vendor/' folder.</i>")
        if rec.requirements_file:
            lines.append(
                f"<br>Requirements: <code>{rec.requirements_file}</code>")
        self.detail_label.setText("<br>".join(lines))

        # Error log
        if rec.error:
            self.error_box.show()
            self.error_text.setPlainText(rec.error)
        else:
            self.error_box.hide()

        # Registrations
        menus = rec.registered_menus
        hooks = rec.registered_hooks
        events = rec.registered_events
        if menus or hooks or events:
            self.registrations_box.show()
            parts = []
            if menus:
                parts.append("Menu items:\n  " + "\n  ".join(menus))
            if hooks:
                parts.append("Display hooks:\n  " + "\n  ".join(hooks))
            if events:
                parts.append("Event subscriptions:\n  " + "\n  ".join(events))
            self.reg_text.setPlainText("\n\n".join(parts))
        else:
            self.registrations_box.hide()

    # -- actions -----------------------------------------------------------

    def _get_selected_name(self) -> Optional[str]:
        item = self.plugin_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _enable(self):
        name = self._get_selected_name()
        if name:
            self.mgr.enable_plugin(name)
            self._refresh_list()

    def _disable(self):
        name = self._get_selected_name()
        if name:
            self.mgr.disable_plugin(name)
            self._refresh_list()

    def _reload(self):
        name = self._get_selected_name()
        if name:
            self.mgr.reload_plugin(name)
            self._refresh_list()

    def _install_deps(self):
        name = self._get_selected_name()
        if name is None:
            return
        rec = self.mgr.get_all_plugins().get(name)
        if rec is None:
            return

        if rec.status != PluginStatus.NEEDS_DEPS:
            QMessageBox.information(
                self, "No Dependencies Needed",
                "This plugin either has no missing dependencies "
                "or is not in a state that supports installation.")
            return

        if is_frozen():
            QMessageBox.warning(
                self, "Compiled Build",
                "This is a compiled (PyInstaller) build — pip is not "
                "available.\n\nPlease download the bundled version of "
                "this plugin that includes a '_vendor/' folder with "
                "pre-compiled dependencies.")
            return

        # Check if requirements mention torch/pytorch — offer GPU options
        gpu_option = "auto"
        if rec.requirements_file:
            req_text = rec.requirements_file.read_text(
                encoding="utf-8").lower()
            if "torch" in req_text:
                gpu_option = self._ask_gpu_option()
                if gpu_option is None:
                    return  # user cancelled

        success = self.mgr.install_plugin_deps(
            name, parent_widget=self, gpu_option=gpu_option)
        self._refresh_list()

    def _ask_gpu_option(self) -> Optional[str]:
        """Show a dialog asking the user about GPU/CUDA preference."""
        dialog = QDialog(self)
        dialog.setWindowTitle("PyTorch GPU Configuration")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel(
            "<b>This plugin requires PyTorch.</b><br><br>"
            "PyTorch can run on CPU only, or accelerated with an "
            "NVIDIA GPU via CUDA. Select the option that matches "
            "your system:"))

        # Detect CUDA
        detected = _detect_cuda_version()
        if detected:
            detect_text = f"Detected CUDA {detected[0]}.{detected[1]}"
        else:
            detect_text = "No CUDA installation detected"

        layout.addWidget(QLabel(f"<i>{detect_text}</i>"))

        group = QButtonGroup(dialog)
        options = [
            ("auto", "Auto-detect (recommended)" + (
                f" — will use CUDA {detected[0]}.{detected[1]}"
                if detected else " — will default to CPU")),
            ("cpu", "CPU only (no GPU acceleration)"),
            ("cu118", "CUDA 11.8"),
            ("cu121", "CUDA 12.1"),
            ("cu124", "CUDA 12.4"),
            ("cu126", "CUDA 12.6"),
        ]
        radios = {}
        for value, label in options:
            radio = QRadioButton(label)
            radios[value] = radio
            group.addButton(radio)
            layout.addWidget(radio)

        radios["auto"].setChecked(True)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Continue")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        for value, radio in radios.items():
            if radio.isChecked():
                return value
        return "auto"

    def _rescan(self):
        self.mgr.discover()
        self.mgr.load_all()
        self._refresh_list()

    @staticmethod
    def _open_folder(path: Path):
        """Open a folder in the system file manager."""
        path.mkdir(parents=True, exist_ok=True)
        import subprocess, platform
        folder = str(path)
        system = platform.system()
        try:
            if system == "Windows":
                os.startfile(folder)
            elif system == "Darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            print(f"Could not open folder: {exc}")
