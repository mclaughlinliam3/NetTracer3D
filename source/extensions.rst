.. _extensions:

========================
NetTracer3D Plugin System
========================

NetTracer3D supports a plugin system that allows both developers and users
to extend the application with new analysis tools, processing methods,
visualisations, and integrations — without modifying the core codebase.

Plugins are discovered automatically at startup and managed through the
**Extensions** panel in the menu bar.

.. contents:: On this page
   :local:
   :depth: 2


-----------
User Guide
-----------

What Are Plugins?
=================

Plugins are small Python modules that add functionality to NetTracer3D.
They can add new menu items, new right-click options, new analysis tabs,
custom overlays, or entirely new dialog windows.  Several plugins ship
with NetTracer3D by default (e.g. **Cellpose Segmentation**,
**Cell Preview Grid**, **Channel Expansion**), and you can install
additional ones or write your own.

Where Do Plugins Live?
======================

NetTracer3D searches for plugins in several locations, in order:

1. **User plugin directory**
   ::

      ~/.nettracer3d/plugins/

   On Windows this is typically ``C:\Users\<you>\.nettracer3d\plugins\``.
   Place any ``.py`` file or plugin folder here and NetTracer3D will find
   it on the next launch.  The Extensions panel has an
   **Open User Plugin Folder** button that opens this directory in your
   file manager.

2. **Package plugin directory**
   ::

      <python>/Lib/site-packages/nettracer3d/plugins/

   Plugins that ship with the ``pip install nettracer3d`` package live
   here.  You generally do not need to touch this directory — it is
   populated automatically by the installer.

3. **Environment variable**

   Set ``NETTRACER3D_PLUGIN_PATH`` to a colon-separated (or
   semicolon-separated on Windows) list of directories containing
   additional plugins.

4. **Pip entry points**

   Plugins distributed as their own pip packages can declare the entry
   point group ``nettracer3d.plugins`` and be discovered automatically
   after ``pip install``.


The Extensions Panel
====================

Open the Extensions panel from the menu bar:

  **Extensions → Manage Extensions...**

The panel shows every discovered plugin, colour-coded by status:

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Colour
     - Meaning
   * - Green
     - **Loaded** — the plugin is active and its menu items / hooks are
       registered.
   * - Blue
     - **Needs Deps** — the plugin was found but could not be imported
       because one or more Python packages are missing.  Click
       **Install Deps** to install them automatically via pip.
   * - Red
     - **Failed** — the plugin raised an error during import or
       registration.  Select it to see the error traceback.
   * - Orange
     - **Incompatible** — the plugin requires a newer version of the
       plugin API than this build of NetTracer3D provides.
   * - Grey
     - **Disabled** — you manually disabled this plugin.  Click
       **Enable** to re-activate it.

Available buttons:

- **Enable** — re-enable a disabled plugin and attempt to load it.
- **Disable** — unload the plugin and prevent it from loading on future
  launches.
- **Install Deps** — (pip environments only) reads the plugin's
  ``requirements.txt``, asks about GPU / CUDA preferences if PyTorch is
  involved, and runs ``pip install`` in the current environment.
- **Reload** — unload and re-import the plugin without restarting
  NetTracer3D.  Useful during development.
- **Rescan** — re-scan all plugin directories for new files and attempt
  to load any newly discovered plugins.


Installing Plugin Dependencies
==============================

When a plugin is marked **Needs Deps** (blue):

1. Select it in the list.
2. Click **Install Deps**.
3. If the plugin requires PyTorch, a dialog appears asking which GPU /
   CUDA version you have.  The manager will try to auto-detect your CUDA
   installation.  Choose **Auto-detect** unless you know you need a
   specific version.
4. A confirmation dialog shows exactly which packages will be installed
   and the full pip command.  Click **Yes** to proceed.
5. pip runs in the background.  When it finishes, the plugin is
   automatically loaded.

.. note::

   If you are running the compiled (PyInstaller / installer) version of
   NetTracer3D, pip is not available.  Plugins for the compiled version
   must be distributed with a ``_vendor/`` folder containing
   pre-compiled dependencies.  See the Developer Guide below for details.


Built-In Plugins
================

Cellpose Segmentation
---------------------

Integrates the `Cellpose <https://cellpose.readthedocs.io/>`_ instance
segmentation pipeline directly into NetTracer3D.

- **Menu**: Extensions → Cellpose → Open Cellpose Panel...
- Choose which channel to segment and optionally a secondary context
  channel (e.g. a nuclear stain).
- Select a model (built-in or custom ``.pth`` file).
- Adjust parameters: diameter, flow threshold, cell probability
  threshold, minimum size, stitch threshold.
- Enable **Chunked Processing** to segment large images in pieces that
  fit in GPU memory.
- Dimensionality (2-D vs 3-D) is auto-detected from the input data.
- The segmented mask is written to the channel of your choice.
- **Requires**: ``cellpose>=3.0`` (installed via the Extensions panel).


---------------
Developer Guide
---------------

This section explains how to write, package, and distribute your own
NetTracer3D plugins.


Plugin Structure
================

Please reference the built in Cellpose plugin _init_.py and requirements.txt files for a clear example of how to integrate a plugin

A plugin is either a single ``.py`` file or a folder (Python package).

**Single file** (no dependencies beyond NetTracer3D base)::

    my_plugin.py

**Folder / package** (has its own dependencies or bundled assets)::

    my_plugin/
    ├── __init__.py          # plugin code
    ├── requirements.txt     # pip dependencies
    └── _vendor/             # (optional) bundled deps for PyInstaller

Every plugin must expose two things at module level:

1. ``PLUGIN_INFO`` — a dictionary of metadata.
2. ``register(api)`` — a function called once when the plugin loads.

Optionally:

3. ``unregister(api)`` — called when the plugin is disabled or the app
   closes.


PLUGIN_INFO
===========

.. code-block:: python

   PLUGIN_INFO = {
       'name': 'My Plugin',              # display name
       'version': '1.0.0',               # semver string
       'author': 'Your Name',
       'description': 'What it does.',
       'api_version': (1, 0),            # minimum API version required
       'requires': [],                    # other plugin names (inter-plugin deps)
       'category': 'analysis',           # analysis | processing | visualization | io | other
   }


register() and unregister()
===========================

.. code-block:: python

   _api = None

   def register(api):
       """Called once when the plugin is loaded."""
       global _api
       _api = api

       # Register menu items, event listeners, display hooks, etc.
       api.register_menu_action(
           "Extensions/My Plugin/Do Something",
           my_callback)

       api.on("slice_changed", on_slice_changed)

   def unregister(api):
       """Called when the plugin is disabled or the app closes."""
       # Clean up any resources.
       pass


Plugin API Reference
====================

The ``api`` object passed to ``register()`` is an instance of
``PluginAPI``.  All methods listed below are part of the **stable public
API** and will not change without a major version bump.

Data Access — Read
------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Method
     - Description
   * - ``api.get_channel_data(index) → ndarray | None``
     - Return a reference to channel data (0 = Nodes, 1 = Edges,
       2 = Overlay 1, 3 = Overlay 2).
   * - ``api.get_channel_names() → list[str]``
     - Return the four channel names.
   * - ``api.get_active_channel() → int``
     - Index of the currently selected channel.
   * - ``api.get_current_slice() → int``
     - Current Z-slice index.
   * - ``api.get_shape() → tuple | None``
     - ``(Z, Y, X)`` shape of the loaded data, or ``None``.
   * - ``api.get_selection() → dict``
     - Deep copy of ``{'nodes': [...], 'edges': [...]}``.
   * - ``api.get_highlight_overlay() → ndarray | None``
     - The current highlight overlay array.
   * - ``api.get_network() → nx.Graph | None``
     - The networkx Graph object.
   * - ``api.get_network_lists() → list | None``
     - ``[node_a_list, node_b_list, edge_list]``.
   * - ``api.get_node_centroids() → dict | None``
     - ``{node_id: [z, y, x], ...}``.
   * - ``api.get_edge_centroids() → dict | None``
     - ``{edge_id: [z, y, x], ...}``.
   * - ``api.get_node_identities() → dict | None``
     - ``{node_id: [identity, ...], ...}``.
   * - ``api.get_communities() → dict | None``
     - ``{node_id: community_id, ...}``.
   * - ``api.get_xy_scale() → float``
     - Physical pixel size in XY.
   * - ``api.get_z_scale() → float``
     - Physical pixel size in Z.
   * - ``api.get_visible_channels() → list[int]``
     - Indices of currently visible channels.

Data Access — Write
-------------------

Write methods validate inputs, update the UI, and trigger display
refreshes automatically.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Method
     - Description
   * - ``api.set_channel_data(index, array)``
     - Replace channel data.  Handles shape validation, undo snapshot,
       button/slider state, and display refresh.
   * - ``api.set_highlight(node_indices, edge_indices)``
     - Update the highlight overlay from index lists.
   * - ``api.set_communities(dict)``
     - Replace the community partition.
   * - ``api.set_node_identities(dict)``
     - Replace node identities.
   * - ``api.set_node_centroids(dict)``
     - Replace node centroids.
   * - ``api.set_xy_scale(float)``
     - Update the XY physical scale.
   * - ``api.set_z_scale(float)``
     - Update the Z physical scale.

UI Output
---------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Method
     - Description
   * - ``api.add_table(title, dataframe)``
     - Add a pandas DataFrame as a tab in the upper-right data panel.
   * - ``api.add_table_from_dict(data, metric, value, title)``
     - Format a Python dict into a table tab.
   * - ``api.add_widget_tab(title, widget)``
     - Add an arbitrary QWidget as a tab.
   * - ``api.show_message(title, text, level)``
     - Show a message box.  ``level``: ``"info"``, ``"warning"``,
       or ``"error"``.
   * - ``api.print(msg)``
     - Print a message to the console.

Menu & Context Registration
---------------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Method
     - Description
   * - ``api.register_menu_action(menu_path, callback, tooltip="")``
     - Add a menu item.  ``menu_path`` uses ``/`` separators, e.g.
       ``"Extensions/My Plugin/Run"``.  Intermediate menus are created
       automatically.
   * - ``api.register_context_action(label, callback)``
     - Add an entry to the image right-click context menu.
       ``callback`` receives ``{'x': int, 'y': int, 'z': int}``.

Display Hooks
-------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Method
     - Description
   * - ``api.register_display_hook(callback)``
     - Register a function called at the end of every display update.
       Signature: ``callback(view, current_slice, view_range)`` where
       ``view`` is the pyqtgraph ``ViewBox``.
   * - ``api.add_view_item(item)``
     - Add a pyqtgraph graphics item to the image view.
   * - ``api.remove_view_item(item)``
     - Remove a previously added graphics item.

Events
------

Subscribe to application events with ``api.on(event, callback)``.
The callback receives a single ``data`` argument whose type depends on
the event.

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Event
     - Data
     - Fired When
   * - ``slice_changed``
     - ``int`` (new slice index)
     - User navigates to a different Z slice.
   * - ``selection_changed``
     - ``dict`` (clicked_values)
     - User clicks or rectangle-selects nodes/edges.
   * - ``channel_loaded``
     - ``int`` (channel index)
     - A channel's data is loaded or replaced.
   * - ``channel_deleted``
     - ``int`` (channel index)
     - A channel is deleted.
   * - ``network_changed``
     - ``None``
     - The network graph is recalculated.
   * - ``communities_changed``
     - ``dict``
     - The community partition is updated.
   * - ``identities_changed``
     - ``dict``
     - Node identities are updated.
   * - ``centroids_changed``
     - ``dict``
     - Node centroids are updated.
   * - ``session_loaded``
     - ``str`` (directory path)
     - A previous session is loaded.
   * - ``session_saved``
     - ``str`` (save name)
     - The current session is saved.
   * - ``plugin_loaded``
     - ``str`` (plugin name)
     - Another plugin finishes loading.
   * - ``plugin_unloaded``
     - ``str`` (plugin name)
     - A plugin is unloaded.
   * - ``display_updated``
     - ``None``
     - The display finishes a full redraw.

Utilities
---------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Method
     - Description
   * - ``api.refresh_display()``
     - Request a full display redraw.
   * - ``api.navigate_to_slice(z)``
     - Change the current Z slice.
   * - ``api.api_version → (int, int)``
     - The current API version as a ``(major, minor)`` tuple.

Unsafe Escape Hatches
---------------------

.. warning::

   These methods return direct references to internal objects.  Anything
   accessed through them may be renamed, removed, or restructured in any
   future release without notice.  Use only for prototyping or accessing
   functionality not yet in the public API.  **Do not ship published
   plugins that depend on internals obtained this way.**

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Method
     - Description
   * - ``api.get_unsafe_window()``
     - Returns the ``ImageViewerWindow`` instance.
   * - ``api.get_unsafe_network()``
     - Returns the ``Network_3D`` object (``my_network``).


Dependency Management
=====================

Plugins declare their Python dependencies in a ``requirements.txt`` file
placed alongside the plugin code.

Pip environments (``pip install nettracer3d``)
----------------------------------------------

When the plugin manager cannot import a plugin due to a missing package,
it checks for ``requirements.txt`` in the plugin's directory.  If found,
the plugin is marked **Needs Deps** instead of **Failed**.  The user can
then click **Install Deps** in the Extensions panel.

The ``requirements.txt`` uses standard pip format:

.. code-block:: text

   cellpose>=3.0
   some-other-package

You do **not** need to list transitive dependencies — pip resolves them
automatically.  For example, listing ``cellpose>=3.0`` is sufficient;
torch and all of cellpose's other dependencies are pulled in
automatically.

If your plugin requires PyTorch, the plugin manager will detect this
from the requirements file and present a GPU / CUDA selection dialog
before running pip.  It auto-detects the installed CUDA version via
``nvidia-smi``, ``nvcc``, or an existing torch installation, and adds
the appropriate ``--extra-index-url`` to the pip command.

PyInstaller / compiled builds
-----------------------------

In a frozen (PyInstaller) environment, pip is not available.  Plugins
must bundle their dependencies in a ``_vendor/`` folder:

::

    my_plugin/
    ├── __init__.py
    ├── requirements.txt       # still included for reference
    └── _vendor/
        ├── cellpose/
        ├── torch/
        └── ...

The plugin manager detects ``sys.frozen``, prepends ``_vendor/`` to
``sys.path`` before importing the plugin, and the bundled packages
resolve normally.

To create a ``_vendor/`` folder:

.. code-block:: bash

   pip install --target my_plugin/_vendor cellpose

   # For GPU support:
   pip install --target my_plugin/_vendor cellpose torch \
       --extra-index-url https://download.pytorch.org/whl/cu124

.. note::

   ``_vendor/`` folders can be very large (>1 GB with PyTorch + CUDA).
   Consider distributing CPU-only and GPU versions separately.


Packaging for PyPI
==================

If you want your plugin to be installable via pip and auto-discovered:

1. Create a Python package with an entry point:

   .. code-block:: toml

      # pyproject.toml
      [project.entry-points."nettracer3d.plugins"]
      my_plugin = "my_package.my_plugin"

2. Your module must expose ``PLUGIN_INFO`` and ``register(api)`` at the
   top level of the entry point target.

3. After ``pip install my-plugin-package``, NetTracer3D will discover it
   automatically on the next launch.

Alternatively, for plugins bundled *inside* the ``nettracer3d`` package
itself (i.e. shipped with the default install), place the plugin folder
in ``nettracer3d/plugins/`` and add a ``package-data`` directive to
``pyproject.toml``:

.. code-block:: toml

   [tool.setuptools.package-data]
   "nettracer3d.plugins" = [
       "*/requirements.txt",
       "*/_vendor/**/*",
   ]

This ensures that ``requirements.txt`` files and ``_vendor/`` contents
are included in the wheel alongside the Python code.


Minimal Example Plugin
======================

.. code-block:: python

   """
   Example plugin that adds a menu item to count objects in the
   active channel.
   """

   import numpy as np

   PLUGIN_INFO = {
       "name": "Object Counter",
       "version": "0.1.0",
       "author": "Your Name",
       "description": "Count unique non-zero labels in the active channel.",
       "api_version": (1, 0),
       "requires": [],
       "category": "analysis",
   }

   _api = None

   def register(api):
       global _api
       _api = api
       api.register_menu_action(
           "Extensions/Object Counter/Count Objects",
           _count_objects,
       )

   def _count_objects():
       channel = _api.get_active_channel()
       data = _api.get_channel_data(channel)
       if data is None:
           _api.show_message("No Data", "Active channel is empty.", "warning")
           return
       unique = np.unique(data)
       n = len(unique) - (1 if 0 in unique else 0)
       _api.show_message(
           "Object Count",
           f"Channel {channel}: {n} unique objects",
           "info",
       )


Complete Plugin Checklist
=========================

.. code-block:: text

   ✓  PLUGIN_INFO dict with name, version, api_version, category
   ✓  register(api) function
   ✓  unregister(api) function (optional but recommended)
   ✓  requirements.txt if any non-base dependencies
   ✓  _vendor/ folder if distributing for PyInstaller
   ✓  Menu items under "Extensions/<Your Plugin>/" namespace
   ✓  Console output prefixed with [YourPlugin] for debuggability
   ✓  Error handling — plugins should not crash the host application
   ✓  No direct access to internals (use api methods; get_unsafe_*
      only as a last resort with the understanding it may break)
