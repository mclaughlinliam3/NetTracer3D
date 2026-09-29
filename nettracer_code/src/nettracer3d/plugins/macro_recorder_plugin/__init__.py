"""
NetTracer3D Macro Recorder
==========================

Records GUI interactions (menu-driven analyses, sub-dialog field entry,
button clicks and save actions) and replays them across many NetTracer3D
sessions in a folder, writing each session's outputs to its own
``<session>`` directory inside a timestamped batch folder.

Drop this file into ``~/.nettracer3d/plugins/`` (or any discovered plugin
directory) and it will appear under the ``Extensions`` menu.

How it works
------------
* **Recording** watches the menu bar, instruments every sub-dialog as it
  opens (connecting to its widgets' signals), and *wraps* the app's save
  methods so a "save" is captured as an intent rather than a hard-coded
  path. A recorded macro is written out as a plain, editable Python file
  (a ``EVENTS = [...]`` list).

* **Playback** is an asynchronous, timer-driven queue. Each step schedules
  the *next* step before it runs, so a step that opens a **modal** dialog
  (``dialog.exec()``) is still driven forward by timer callbacks firing
  inside the modal's nested event loop. Modeless dialogs work the same way.

* **Saving in batch** intercepts ``QFileDialog`` and redirects every save
  into ``<parent>/MacroBatch_<timestamp>/<session>/``, so repeat runs never
  overwrite earlier results. Message boxes are auto-answered so the run
  never blocks waiting for a click.

Recommended workflow
--------------------
1. Load one representative session (``File -> Load``).
2. Open this dialog, press **Start Recording**.
3. Run your analysis via the menus / sub-dialogs, then save the output
   table(s) as you normally would (right-click table -> Save As).
4. Press **Stop Recording** and save the macro.
5. Press **Run Macro on Folder**, pick the parent folder that holds your
   session sub-folders, and let it iterate.

Notes / limitations
--------------------
* Third-party plugins work without any changes on their part. Because the
  recorder hooks Qt itself (the menu bar, an app-wide event filter,
  QMenu/QMessageBox/QFileDialog) rather than any specific NetTracer3D class,
  a plugin's menu entry, its dialog fields and buttons, its right-click
  entries, and its own save dialogs are all recorded and replayed. A plugin
  export made through ``QFileDialog.getSaveFileName`` is redirected into the
  session's output folder and given the session suffix like anything else.
* Message boxes never block a run. Notices (``information``/``warning``/
  ``critical``, e.g. "Please select spreadsheet...") are dismissed with their
  default button and are not recorded at all, so a notice that appears in
  only some sessions cannot desync anything.
* Confirmations are answered **Yes** during playback (``AUTO_ANSWER_YES``).
  This covers throwaway boxes built inline as
  ``msg = QMessageBox(); msg.setStandardButtons(Yes | No); msg.exec()`` --
  they have no class of their own, no addressable buttons, and their text is
  often session-specific, so matching a recorded answer to them is
  unreliable. Answering affirmatively means ``if msg.exec() == Yes:`` always
  takes the branch that does the work the macro was recorded to do. Boxes
  that offer no Yes are answered with whatever they do offer (Ok, Save,
  Retry, ... in that preference order), including custom ``addButton``
  buttons, so nothing can sit waiting on a nested event loop. Set
  ``AUTO_ANSWER_YES = False`` to go back to replaying recorded answers.
* Output folders are created lazily: ``<session>`` (and the batch folder
  itself) appear the first time that session actually writes a file. A macro
  that only computes and never saves leaves no empty folders behind. The
  folder is named after the session it came from; set ``OUTPUT_SUFFIX`` to
  ``"_Output"`` for the older ``<session>_Output`` naming.
* Saving a whole **NetTracer3D session** (``save_network_3d`` /
  ``save_pickled_net`` -- the same kind of folder the batch loads in the
  first place) is handled as a unit rather than as an ordinary file save.
  The destination is the session's own output folder, whichever dialog the
  save routine asks with, and the component files inside it keep the exact
  names the save routine gave them -- they are *not* given the per-file
  session suffix, because renaming them is what stops a saved session from
  loading back with all of its sub-properties. Set
  ``SESSION_SAVE_SUBFOLDER = True`` to nest the saved session one level
  deeper, at ``<session>/<session>/``, if the same macro also writes tables
  or screenshots you would rather keep out of it.
* Tool windows that are plain ``QWidget`` top-level windows rather than
  ``QDialog`` subclasses (e.g. the Histogram Selector in ``histos.py``) are
  recorded like any other dialog. A nested ``QApplication.exec()`` started by
  such a window is suppressed during playback, since the app is already
  running an event loop; matplotlib's blocking ``show()`` is suppressed too.
* On-screen controls are recorded: channel visibility toggles, the scalebar
  toggle, the home/reset button, the active-image selector, the highlight
  overlay toggle, and the camera (screenshot) button. Screenshots are written
  into the session's output folder as ``screenshot_<session>.<ext>``, using
  the file type chosen while recording.
* Checkable controls replay to the *state* that was recorded rather than
  blindly re-clicking, so a session that already loaded with (say) the
  scalebar on ends up in the same state as every other session. A control
  that is disabled for a given session is logged and skipped.
* The zoom, pan, 3D, popout and pen buttons are deliberately ignored -- they
  are view/interaction modes that do not carry meaning between sessions.
* Right-click (context menu) actions ARE recorded, on both the image display
  window and the data tables. They are stored by their menu label path
  (e.g. ``Show Identity > ID: neuron``) and replayed by rebuilding the same
  context menu and firing that entry -- no cursor position is involved.
* Because they are matched by label, a context entry that does not exist in a
  given session (an identity or community absent from that dataset, a submenu
  that only appears for 2D/3D data) is logged and skipped; the rest of the
  macro continues.
* Context entries whose meaning depends on the pixel you clicked or on the
  current selection -- measurement points, "Show Neighbors", "Selection >
  ..." -- are recorded but replay against whatever is selected at that point
  in the macro, which may differ per session. Prefer selection-independent
  entries (identities, communities, Select All, Sort, Save As) for batch work.
* Left-click selection on the image canvas, and direct manipulation
  (pan/zoom/paint), are not recorded.
* The recorder never records itself.
"""

from __future__ import annotations

import os
import sys
import types
import contextlib
import gc
import traceback
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer, QObject, QEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget,
    QListWidgetItem, QFileDialog, QMessageBox, QApplication, QGroupBox,
    QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QRadioButton, QSlider, QAbstractSpinBox,
    QToolButton, QWidget, QProgressBar, QMenu,
)


# ---------------------------------------------------------------------------
# Safe monkey-patching of Qt (sip) methods
# ---------------------------------------------------------------------------
# ``QMenu.exec`` and friends live in the class dict as a sip
# ``methoddescriptor``. Reading one with ``getattr(QMenu, "exec")`` hands back
# an *unbound* ``builtin_function_or_method`` which does NOT implement the
# descriptor protocol. Assigning that object back onto the class therefore
# looks like a clean restore but silently destroys binding: ``menu.exec(pos)``
# then calls ``exec(pos)`` with no ``self``, and PyQt raises
#     TypeError: arguments did not match any overloaded call:
#       exec(self): first argument of unbound method must have type 'QMenu'
# The pristine descriptor must be captured from ``cls.__dict__`` instead, and
# put back from there. Note that ``delattr`` is NOT a valid alternative: sip
# does not lazily regenerate the slot, so deleting it removes the method
# outright.
_QT_PRISTINE = {}


def _patch_qt_method(cls, name, replacement):
    """Patch a Qt instance method; return the original callable.

    Records the pristine sip descriptor the first time each method is
    patched so it can be restored faithfully later.
    """
    key = (cls, name)
    if key not in _QT_PRISTINE:
        _QT_PRISTINE[key] = cls.__dict__.get(name)
    original = getattr(cls, name)
    setattr(cls, name, replacement)
    return original


def _restore_qt_method(cls, name, original=None):
    """Undo :func:`_patch_qt_method` so the method still binds to ``self``."""
    key = (cls, name)
    if key in _QT_PRISTINE:
        desc = _QT_PRISTINE[key]
        if desc is not None:
            try:
                setattr(cls, name, desc)
                return
            except Exception:
                pass
        else:
            # The method was inherited (e.g. QMessageBox.exec comes from
            # QDialog); dropping our shadow lets the base class resurface.
            try:
                delattr(cls, name)
                if getattr(cls, name, None) is not None:
                    return
            except Exception:
                pass
    # Fallback: a plain Python function *is* a descriptor, so wrapping the
    # unbound callable restores correct binding even without the original.
    if original is None:
        return

    def shim(self, *a, **k):
        return original(self, *a, **k)

    shim.__name__ = name
    try:
        setattr(cls, name, shim)
    except Exception:
        pass


def _verify_qt_methods():
    """Belt-and-braces sweep: repair any Qt method left in a non-binding state.

    Called at the end of every unpatch pass so that a failure partway through
    can never leave the host GUI with a broken context menu.
    """
    for (cls, name) in list(_QT_PRISTINE):
        current = cls.__dict__.get(name)
        if current is not None and not hasattr(type(current), "__get__"):
            _restore_qt_method(cls, name)


PLUGIN_INFO = {
    "name": "Macro Recorder",
    "version": "1.10.0",
    "author": "NetTracer3D community",
    "description": "Record GUI interactions and replay them across many "
                   "sessions for batch analysis, saving each session's "
                   "outputs to a matching output folder.",
    "api_version": (1, 0),
    "requires": [],
    "category": "processing",
}

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Master output folder for one batch run: <parent>/<prefix><timestamp>/
BATCH_PREFIX = "MacroBatch_"

# Per-session folder inside the batch root: <batch>/<session><suffix>/.
# Empty means the folder is named exactly after the session it came from, so a
# batch of session saves comes out as a set of folders with the same names as
# the inputs. Set to "_Output" to restore the old <session>_Output naming.
OUTPUT_SUFFIX = ""

# Main-window methods that write a whole NetTracer3D session (a *folder* of
# component files) rather than a single output file. These are special-cased
# throughout: they are named after the session, their contents are never
# renamed, and they are never given the per-file session suffix.
SESSION_SAVE_METHODS = ("save_network_3d", "save_pickled_net")

# Where a session save lands.
#   False -> straight into <batch>/<session>/, so the per-session output folder
#            *is* the saved session and can be re-loaded (or re-batched) as-is.
#   True  -> nested one level down, <batch>/<session>/<session>/, which keeps
#            the saved session separate from any tables/screenshots the same
#            macro writes.
SESSION_SAVE_SUBFOLDER = False

# How many further playback steps a name reserved by a gesture stays available
# for, when the gesture returned before its save actually fired. Long enough to
# cover a save that runs on a queued callback or from a dialog the gesture
# opened; short enough that a gesture which never saves at all cannot lend its
# name to an unrelated save later in the macro.
DEFERRED_SAVE_WINDOW = 8

# Run a session save with the working directory set to the destination folder.
# Network_3D.dump() wraps all eleven of its component saves in a single bare
# ``except:`` whose fallback re-runs every one of them with *no* directory
# argument -- which writes them relative to the process working directory. One
# component raising therefore scatters the rest wherever NetTracer3D happens to
# have been started from, leaving a half-populated session folder behind. This
# cannot be fixed from the plugin side, but pointing the working directory at
# the destination means the fallback lands in the right folder anyway.
SESSION_SAVE_CHDIR = True

# Keep the host window's "last saved" location pointing at the batch output for
# the duration of a run. Without this a recorded plain *Save* (as opposed to
# *Save As*) replays through ``save_network_3d(False)``, which opens no dialog
# at all -- there is nothing for the redirect to intercept, and the save lands
# straight back on top of the source session the batch just loaded.
REDIRECT_PLAIN_SAVE = True

# Replace Network_3D.dump() with a per-component version for the duration of a
# batch run. The stock dump() saves all eleven components inside a single try
# block, so the first one to raise aborts every component after it -- and its
# fallback re-runs the whole list with no directory, which cannot succeed for
# the component that raised in the first place. Saving each one independently
# means a single failing property (an empty or oddly-shaped communities dict is
# the usual culprit) costs you that property alone, and the exception is written
# to the macro log instead of being swallowed.
ROBUST_SESSION_DUMP = True

# Delete dialogs left over from a session before starting the next one. A batch
# re-opens every dialog the macro touches once per round, so a dialog class that
# is parented to the main window and lacks WA_DeleteOnClose accumulates one live
# instance per session -- each still holding whatever channel data it captured.
# Closing such a dialog only hides it; only deletion frees it.
REAP_DIALOGS_BETWEEN_SESSIONS = True

# Close matplotlib figures between sessions. pyplot's figure manager holds every
# figure ever created until it is explicitly closed, which is invisible to any
# Qt-side accounting.
CLOSE_FIGURES_BETWEEN_SESSIONS = True

# Log resident memory and live-dialog counts after each session, so a leak can
# be attributed to the batch runner or ruled out of it.
LOG_MEMORY_PER_SESSION = True

# After a session save, check that each component saver actually wrote
# something. Several of them swallow their own exceptions (save_json has a bare
# except that can leave a truncated file behind, and save_singval_iden_dict
# catches everything and only prints), so a component can fail without raising.
VERIFY_SESSION_COMPONENTS = True

# Component savers called by Network_3D.dump(), in order, with the keyword each
# one is given there. A name missing from a given build is skipped.
_DUMP_COMPONENTS = (
    ("save_nodes", {"compression": "zlib"}),
    ("save_edges", {"compression": "zlib"}),
    ("save_node_centroids", {}),
    ("save_search_region", {"compression": "zlib"}),
    ("save_network", {}),
    ("save_node_identities", {}),
    ("save_edge_centroids", {}),
    ("save_scaling", {}),
    ("save_communities", {}),
    ("save_network_overlay", {"compression": "zlib"}),
    ("save_id_overlay", {"compression": "zlib"}),
)

STEP_DELAY_MS = 120        # pause between playback steps (also aids visibility)
DIALOG_WAIT_MS = 60        # poll interval while waiting for a dialog to appear
DIALOG_WAIT_RETRIES = 60   # ~3.6s max wait for a dialog to appear

# Widget types we treat as "inputs" (order within each type is what matters).
# Enumerated identically at record- and playback-time so indices line up.
_INPUT_TYPES = [
    QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QRadioButton, QSlider, QGroupBox,
]
_INPUT_TYPE_BY_NAME = {t.__name__: t for t in _INPUT_TYPES}

# Button texts that usually mean "accept / run this dialog".
_ACCEPT_HINTS = {
    "ok", "run", "apply", "yes", "accept", "confirm", "calculate",
    "compute", "start", "go", "done", "finish", "generate", "continue",
    "proceed", "save",
}

# During playback, answer every confirmation affirmatively instead of using
# the answer that was recorded. Inline boxes (``msg = QMessageBox(); ...;
# msg.exec()``) are not classes, expose no addressable buttons, and usually
# carry session-specific text, so a recorded answer cannot be matched to them
# reliably -- and a "No" answers away the very work the macro exists to do.
# Set to False to replay recorded answers instead (older behaviour).
AUTO_ANSWER_YES = True

# Preferred replies for a *question* box, most affirmative first. The first
# button the box actually offers wins; the tail entries only ever apply to a
# box that offers no affirmative answer at all, where the goal is simply to
# close it so the run continues.
_QUESTION_BUTTON_ORDER = (
    "Yes", "YesToAll", "Ok", "Apply", "Save", "SaveAll", "Open", "Retry",
    "Close", "Ignore", "Discard", "Cancel", "No", "NoToAll", "Abort",
)

# Preferred replies for a *notice* (information/warning/critical). These are
# dismissals, so Ok comes first -- but a notice built with Yes/No is still a
# question in disguise and is answered Yes.
_NOTICE_BUTTON_ORDER = (
    "Ok", "Yes", "Close", "Ignore", "Discard", "Cancel", "No", "Abort",
    "Retry",
)


def _clean(text: str) -> str:
    """Normalise menu/button text for matching (drop mnemonics/whitespace)."""
    return (text or "").replace("&", "").strip()


def _is_inside(widget, types_tuple) -> bool:
    """True if *widget* is a descendant of any widget of the given types."""
    p = widget.parent()
    while p is not None:
        if isinstance(p, types_tuple):
            return True
        p = p.parent()
    return False


def _typed_widgets(container, cls):
    """Deterministic, filtered list of *cls* descendants of *container*.

    QLineEdit is filtered to exclude the internal editors that live inside
    spin boxes / combo boxes, so indices remain stable and meaningful.
    """
    widgets = container.findChildren(cls)
    if cls is QLineEdit:
        widgets = [w for w in widgets
                   if not _is_inside(w, (QAbstractSpinBox, QComboBox))]
    return widgets


def _buttons(container):
    """Deterministic list of clickable push/tool buttons in *container*."""
    return (list(container.findChildren(QPushButton))
            + list(container.findChildren(QToolButton)))


def _label_hint(widget) -> str:
    """A short human hint for a widget (for readable macro scripts)."""
    try:
        if isinstance(widget, (QCheckBox, QRadioButton, QGroupBox)):
            return _clean(widget.title() if isinstance(widget, QGroupBox)
                          else widget.text())
        name = widget.objectName()
        if name:
            return name
    except Exception:
        pass
    return ""


def _get_value(widget):
    """Return ``(kind, value)`` for a supported input widget, else None."""
    try:
        if isinstance(widget, QCheckBox):
            return ("bool", bool(widget.isChecked()))
        if isinstance(widget, QRadioButton):
            return ("bool", bool(widget.isChecked()))
        if isinstance(widget, QGroupBox):
            if widget.isCheckable():
                return ("bool", bool(widget.isChecked()))
            return None
        if isinstance(widget, QComboBox):
            return ("combo", [int(widget.currentIndex()),
                              str(widget.currentText())])
        if isinstance(widget, QDoubleSpinBox):
            return ("float", float(widget.value()))
        if isinstance(widget, QSpinBox):
            return ("int", int(widget.value()))
        if isinstance(widget, QSlider):
            return ("int", int(widget.value()))
        if isinstance(widget, QLineEdit):
            return ("text", str(widget.text()))
        if isinstance(widget, QPlainTextEdit):
            return ("ptext", str(widget.toPlainText()))
        if isinstance(widget, QTextEdit):
            return ("ptext", str(widget.toPlainText()))
    except Exception:
        return None
    return None


def _set_value(widget, kind, value) -> None:
    """Apply a recorded ``(kind, value)`` to a live widget."""
    if kind == "bool":
        widget.setChecked(bool(value))
    elif kind == "combo":
        idx, text = value
        if isinstance(idx, int) and 0 <= idx < widget.count():
            widget.setCurrentIndex(idx)
        else:
            j = widget.findText(str(text))
            if j >= 0:
                widget.setCurrentIndex(j)
            elif widget.isEditable():
                widget.setCurrentText(str(text))
    elif kind in ("int", "float"):
        widget.setValue(value)
    elif kind == "text":
        widget.setText(str(value))
        # Nudge any editingFinished-based logic in the dialog.
        try:
            widget.editingFinished.emit()
        except Exception:
            pass
    elif kind == "ptext":
        widget.setPlainText(str(value))


# ===========================================================================
# The recorder / player / batch controller, wrapped in the dialog.
# ===========================================================================

class MacroRecorderDialog(QDialog):
    """Single-instance dialog that records, saves, loads and runs macros."""

    def __init__(self, api, parent=None):
        super().__init__(parent)
        self._macro_internal = True          # so we never record ourselves
        self.api = api
        self.win = api.get_unsafe_window()
        self.gui_mod = sys.modules.get(type(self.win).__module__)

        self.setWindowTitle("Macro Recorder")
        self.setMinimumSize(560, 460)

        # --- recording state ------------------------------------------------
        self.recording = False
        self.events = []                     # events captured for current take
        self.current_events = []             # the "loaded" macro to run
        self.current_macro_name = None
        self._internal_msg = False           # guard: suppress recording our own boxes
        self._baseline = {}                  # per-dialog baseline field values

        # --- playback state -------------------------------------------------
        self._playing = False
        self._pb = None                      # active playback context
        self._pb_dialog = None               # currently open analysis dialog
        self._pending_save_path = None       # redirect target for next save
                                             # (one-shot: consumed on use)
        self._deferred_save_path = None      # name reserved by a gesture whose
                                             # save had not fired by the time
                                             # the gesture returned
        self._deferred_save_step = -1        # playback step it was deferred at
        self._save_stem_hint = None          # stem for extra, unnamed writes
                                             # belonging to the armed save
        self._session_save_target = None     # folder an in-flight session save
                                             # must write into
        self._session_save_step = -1         # -1 while the save is still the
                                             # step in progress
        self._reserved_paths = set()         # every path handed to a save
                                             # dialog this run (see _unique_path)
        self._notag_paths = set()            # files a session save wrote; their
                                             # names must survive verbatim
        self._session_dirs = set()           # folders that ARE saved sessions
        self._current_output_dir = None      # planned path; may not exist yet
        self._made_dirs = set()              # output folders actually created
        self._batch = None

        # --- save-method wrapping bookkeeping -------------------------------
        self._orig = {}                      # restore handles for wrapped saves
        self._orig_dialogs = {}              # restore handles for patched Qt statics
        self._msg_fifo = {}                  # recorded message-box answers (playback)
        self._open_fifo = {}                 # recorded file-open paths (playback)
        self._msg_by_key = {}                # recorded answers keyed by message text
        self._filesave_fifo = []             # recorded save-dialog names
        self._save_depth = {}                # re-entrancy guard for wrapped saves
        self._orig_dump = None               # (class, method) restore handle
        self._last_rss = None                # RSS after the previous session
        self._ctx_pending = None             # which widget is opening a menu
        self._last_save_path = None          # last path chosen in a save dialog
        self._ctrl_conns = []                # recording hooks on GUI controls
        self._aborted = False                # set when the window goes away
        self._shutting_down = False
        self._pb_ctx_target = None           # label path to trigger (playback)
        self._pb_ctx_hit = False

        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Record an analysis on one loaded session, then replay it across "
            "every session folder inside a parent directory. Each session's "
            "outputs are written to a matching <b>&lt;session&gt;</b> folder "
            "inside a timestamped batch folder.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Recording controls
        rec_box = QGroupBox("Record")
        rec_row = QHBoxLayout(rec_box)
        self.btn_start = QPushButton("\u25CF  Start Recording")
        self.btn_stop = QPushButton("\u25A0  Stop Recording")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        rec_row.addWidget(self.btn_start)
        rec_row.addWidget(self.btn_stop)
        layout.addWidget(rec_box)

        # Load / run controls
        run_box = QGroupBox("Load & Run")
        run_row = QHBoxLayout(run_box)
        self.btn_load = QPushButton("Load Macro\u2026")
        self.btn_run = QPushButton("\u25B6  Run Macro on Folder\u2026")
        self.btn_load.clicked.connect(self.load_macro)
        self.btn_run.clicked.connect(self.run_macro)
        run_row.addWidget(self.btn_load)
        run_row.addWidget(self.btn_run)
        layout.addWidget(run_box)

        self.status_label = QLabel("No macro loaded.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addWidget(QLabel("Log:"))
        self.log_list = QListWidget()
        layout.addWidget(self.log_list, 1)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear Log")
        close_btn = QPushButton("Close")
        clear_btn.clicked.connect(self.log_list.clear)
        close_btn.clicked.connect(self.close)
        btn_row.addStretch(1)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._update_status()

    def _log(self, msg: str):
        # NB: never call QApplication.processEvents() here. During playback,
        # _log runs inside timer callbacks that may themselves be executing
        # within a modal dialog's nested event loop; pumping events there
        # causes re-entrant recursion. The list repaints on the next normal
        # return to the event loop, which is sufficient.
        stamp = datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{stamp}] {msg}")
        self.log_list.addItem(item)
        self.log_list.scrollToBottom()

    def _update_status(self):
        if self.recording:
            self.status_label.setText(
                f"<b>Recording\u2026</b> {len(self.events)} step(s) captured.")
        elif self.current_events:
            name = self.current_macro_name or "(unsaved)"
            self.status_label.setText(
                f"Loaded macro: <b>{name}</b> \u2014 "
                f"{len(self.current_events)} step(s).")
        else:
            self.status_label.setText("No macro loaded.")

    def _info(self, title, text):
        """Message box that doesn't get recorded."""
        self._internal_msg = True
        try:
            QMessageBox.information(self, title, text)
        finally:
            self._internal_msg = False

    def _warn(self, title, text):
        self._internal_msg = True
        try:
            QMessageBox.warning(self, title, text)
        finally:
            self._internal_msg = False

    # ================================================================ RECORD

    def start_recording(self):
        if self.recording:
            return
        if self._playing:
            self._warn("Busy", "Cannot record while a macro is running.")
            return
        self.events = []
        self._baseline = {}
        self._aborted = False
        self.recording = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)

        # Watch the menu bar for triggered analyses.
        try:
            self.win.menuBar().triggered.connect(self._on_menu_triggered)
        except Exception:
            pass
        # Instrument dialogs as they appear.
        QApplication.instance().installEventFilter(self)
        # Wrap the save methods so saves are captured semantically.
        self._wrap_saves_for_recording()

        self._log("Recording started. Perform your analysis, then Stop.")
        self._update_status()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        try:
            self.win.menuBar().triggered.disconnect(self._on_menu_triggered)
        except Exception:
            pass
        QApplication.instance().removeEventFilter(self)
        self._unwrap_saves()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)

        self.current_events = list(self.events)
        self._log(f"Recording stopped. {len(self.current_events)} step(s).")

        if not self.current_events:
            self._warn("Empty Macro",
                       "No steps were recorded. Nothing to save.")
            self._update_status()
            return

        # Prompt to save the macro as a python script.
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Macro As", "macro.py", "Python Macro (*.py)")
        if path:
            if not path.endswith(".py"):
                path += ".py"
            try:
                self._write_macro_script(path, self.current_events)
                self.current_macro_name = os.path.basename(path)
                self._log(f"Saved macro to {path}")
            except Exception as e:
                self._warn("Save Failed", f"Could not write macro:\n{e}")
        self._update_status()

    # -- recording: event filter (dialog discovery) ----------------------

    def eventFilter(self, obj, event):
        try:
            etype = event.type()
            if etype == QEvent.Type.Show:
                if self.recording:
                    self._maybe_instrument_dialog(obj)
                if self._playing:
                    self._maybe_track_playback_dialog(obj, opened=True)
            elif etype in (QEvent.Type.Hide, QEvent.Type.Close):
                if self._playing:
                    self._maybe_track_playback_dialog(obj, opened=False)
        except Exception:
            pass
        return False  # never consume

    def _is_recordable_dialog(self, obj) -> bool:
        """Windows whose widgets we instrument.

        Not every tool window in this GUI is a QDialog -- some are plain
        QWidget top-level windows created elsewhere (e.g. the Histogram
        Selector in ``histos.py``). Those are recordable too; we just have to
        exclude popups, message/file dialogs, the main window itself, and
        matplotlib's own figure windows.
        """
        if not isinstance(obj, QWidget):
            return False
        if isinstance(obj, (QFileDialog, QMessageBox, QMenu)):
            return False
        if obj is self.win:
            return False
        if getattr(obj, "_macro_internal", False):
            return False
        try:
            top = obj.window()
        except Exception:
            return False
        if getattr(top, "_macro_internal", False):
            return False
        if isinstance(obj, QDialog):
            return True
        # Non-dialog: only free-standing windows, never child widgets.
        try:
            if not obj.isWindow():
                return False
            if obj is self.win.window():
                return False
            wtype = obj.windowFlags() & Qt.WindowType.WindowType_Mask
            if wtype in (Qt.WindowType.Popup, Qt.WindowType.ToolTip,
                         Qt.WindowType.SplashScreen, Qt.WindowType.Desktop):
                return False
        except Exception:
            return False
        mod = getattr(type(obj), "__module__", "") or ""
        if mod.startswith("matplotlib") or mod.startswith("mpl_toolkits"):
            return False
        return True

    def _maybe_instrument_dialog(self, obj):
        if not self._is_recordable_dialog(obj):
            return
        if getattr(obj, "_macro_instrumented", False):
            # Re-shown: refresh baseline so only new changes are captured.
            self._snapshot_baseline(obj)
            return
        obj._macro_instrumented = True
        cls = type(obj).__name__
        self._emit({"op": "dialog_open", "cls": cls,
                    "title": _clean(obj.windowTitle())})
        self._log(f"  dialog opened: {cls}")

        # Baseline current field values (so defaults aren't emitted).
        self._snapshot_baseline(obj)

        # Connect buttons -> record snapshot + click.
        for b in _buttons(obj):
            try:
                b.clicked.connect(
                    lambda _checked=False, d=obj, btn=b:
                    self._on_dialog_button(d, btn))
            except Exception:
                pass

        # Capture Enter-to-accept (no button click) as a fallback.
        try:
            obj.finished.connect(
                lambda result, d=obj: self._on_dialog_finished(d, result))
        except Exception:
            pass

    def _snapshot_baseline(self, dialog):
        dialog._macro_closed_by_click = False
        base = {}
        for T in _INPUT_TYPES:
            for i, w in enumerate(_typed_widgets(dialog, T)):
                kv = _get_value(w)
                if kv is not None:
                    base[(T.__name__, i)] = kv
        self._baseline[id(dialog)] = base

    def _emit_changed_fields(self, dialog):
        """Emit set-events for fields that changed from their baseline."""
        base = self._baseline.get(id(dialog), {})
        cls = type(dialog).__name__
        for T in _INPUT_TYPES:
            for i, w in enumerate(_typed_widgets(dialog, T)):
                kv = _get_value(w)
                if kv is None:
                    continue
                key = (T.__name__, i)
                if base.get(key) != kv:
                    kind, value = kv
                    self._emit({
                        "op": "set", "scope": "dialog", "dialog_cls": cls,
                        "w_class": T.__name__, "w_index": i,
                        "hint": _label_hint(w), "kind": kind, "value": value,
                    })
                    base[key] = kv
        self._baseline[id(dialog)] = base

    def _on_dialog_button(self, dialog, button):
        if not self.recording:
            return
        try:
            self._emit_changed_fields(dialog)
            btns = _buttons(dialog)
            try:
                idx = btns.index(button)
            except ValueError:
                idx = None
            text = _clean(button.text())
            dialog._macro_closed_by_click = True
            self._emit({"op": "click", "scope": "dialog",
                        "dialog_cls": type(dialog).__name__,
                        "button_text": text, "button_index": idx})
            self._log(f"  click: {text or '(button %s)' % idx}")
        except Exception:
            pass

    def _on_dialog_finished(self, dialog, result):
        if not self.recording:
            return
        if result != QDialog.DialogCode.Accepted:
            return
        # A button connected in the dialog's __init__ often triggers accept()
        # *before* our own click handler runs, so we can't yet know whether a
        # button handled this. Defer the decision to the next event-loop tick,
        # by which point _on_dialog_button (if any) will have set the flag.
        QTimer.singleShot(0, lambda d=dialog: self._finalize_accept(d))

    def _finalize_accept(self, dialog):
        if not self.recording:
            return
        try:
            if getattr(dialog, "_macro_closed_by_click", False):
                return  # a real button click already recorded the accept
            # Genuine keyboard/programmatic accept: snapshot + synthetic accept.
            self._emit_changed_fields(dialog)
            accept_btn = self._guess_accept_button(dialog)
            text = _clean(accept_btn.text()) if accept_btn else ""
            idx = None
            if accept_btn is not None:
                btns = _buttons(dialog)
                idx = btns.index(accept_btn) if accept_btn in btns else None
            self._emit({"op": "click", "scope": "dialog",
                        "dialog_cls": type(dialog).__name__,
                        "button_text": text, "button_index": idx,
                        "synthetic_accept": True})
            self._log("  accept (via keyboard)")
        except Exception:
            pass

    @staticmethod
    def _guess_accept_button(dialog):
        btns = _buttons(dialog)
        for b in btns:
            if _clean(b.text()).lower() in _ACCEPT_HINTS:
                return b
        return btns[-1] if btns else None

    # -- recording: menu actions -----------------------------------------

    def _on_menu_triggered(self, action):
        if not self.recording:
            return
        try:
            menubar = self.win.menuBar()
            path = self._find_action_path(menubar, action)
            if not path:
                return
            # Never record the menu entry that opens this recorder.
            if any("macro recorder" in p.lower() for p in path):
                return
            self._emit({"op": "menu", "path": path})
            self._log("menu: " + " > ".join(path))
        except Exception:
            pass

    @staticmethod
    def _find_action_path(menubar, target):
        def rec(node, trail):
            for act in node.actions():
                sub = act.menu()
                if sub is not None:
                    r = rec(sub, trail + [_clean(act.text())])
                    if r:
                        return r
                elif act is target:
                    return trail + [_clean(act.text())]
            return None
        return rec(menubar, [])

    # -- recording: wrapping save methods --------------------------------

    def _wrap_saves_for_recording(self):
        rec = self

        # 1) Table saves (class-level; covers all current + future tables).
        CTV = getattr(self.gui_mod, "CustomTableView", None)
        if CTV is not None and hasattr(CTV, "save_table_as"):
            orig = CTV.save_table_as

            def wrapped_table_save(self_tbl, file_type, *a, **k):
                result = orig(self_tbl, file_type, *a, **k)
                try:
                    if rec.recording and not rec._internal_msg:
                        tab, kind = rec._identify_table(self_tbl)
                        rec._emit({"op": "save", "method": "save_table_as",
                                   "fmt": file_type, "tab": tab,
                                   "table_kind": kind})
                        rec._log(f"save: table '{tab or kind}' as {file_type}")
                except Exception:
                    pass
                return result

            CTV.save_table_as = wrapped_table_save
            self._orig[("cls", CTV, "save_table_as")] = orig

        # 2) Main-window saves (instance-level shadows the class method).
        for meth in ("save", "save_network_3d", "save_pickled_net"):
            if hasattr(self.win, meth):
                bound = getattr(self.win, meth)

                def make(mname, original):
                    def wrapped(*a, **k):
                        # save_network_3d's plain-Save branch calls
                        # self.save_network_3d() -- i.e. this wrapper -- when
                        # there is no previous save location. Without a depth
                        # guard one Save is recorded as two save events, and
                        # playback writes the session twice.
                        depth = rec._save_depth.get(mname, 0)
                        rec._save_depth[mname] = depth + 1
                        try:
                            result = original(*a, **k)
                        finally:
                            rec._save_depth[mname] = depth
                        try:
                            if (rec.recording and not rec._internal_msg
                                    and depth == 0):
                                ev = {"op": "save", "method": mname}
                                if mname == "save":
                                    # first positional arg is ch_index
                                    ev["ch_index"] = (a[0] if a else
                                                      k.get("ch_index"))
                                rec._emit(ev)
                                rec._log(f"save: {mname}")
                        except Exception:
                            pass
                        return result
                    return wrapped

                setattr(self.win, meth, make(meth, bound))
                self._orig[("inst", meth)] = bound

        # Record message-box answers so playback reproduces choices.
        self._wrap_message_boxes_for_recording()

        # Record file-open selections so playback reloads the same file.
        self._wrap_open_dialogs_for_recording()

        # Record right-click (context menu) actions.
        self._wrap_context_menus_for_recording()

        # Record the on-screen control buttons.
        self._connect_controls_for_recording()

    # Buttons deliberately NOT recorded: zoom, pan, 3D, popout and pen are
    # view/interaction modes that don't carry meaning between sessions.
    _CONTROL_MAP = [
        ("reset_view", "home", False),
        ("toggle_scale", "scalebar", True),
        ("high_button", "highlight", True),
    ]

    def _connect_controls_for_recording(self):
        rec = self
        self._ctrl_conns = []

        def hook(signal, slot):
            try:
                signal.connect(slot)
                self._ctrl_conns.append((signal, slot))
            except Exception:
                pass

        # Simple / checkable buttons.
        for attr, target, checkable in self._CONTROL_MAP:
            btn = getattr(self.win, attr, None)
            if btn is None:
                continue

            def make(target=target, btn=btn, checkable=checkable):
                def slot(*_a):
                    if not rec.recording:
                        return
                    ev = {"op": "ui", "target": target}
                    if checkable:
                        ev["checked"] = bool(btn.isChecked())
                    rec._emit(ev)
                    rec._log("control: " + rec._describe(ev))
                return slot

            hook(btn.clicked, make())

        # Channel visibility toggles.
        for i, btn in enumerate(getattr(self.win, "channel_buttons", []) or []):
            def make_ch(idx=i, btn=btn):
                def slot(*_a):
                    if not rec.recording:
                        return
                    ev = {"op": "ui", "target": "channel", "index": idx,
                          "checked": bool(btn.isChecked())}
                    rec._emit(ev)
                    rec._log("control: " + rec._describe(ev))
                return slot
            hook(btn.clicked, make_ch())

        # Active image selector.
        combo = getattr(self.win, "active_channel_combo", None)
        if combo is not None:
            def combo_slot(index):
                if not rec.recording:
                    return
                ev = {"op": "ui", "target": "active_image", "index": int(index)}
                rec._emit(ev)
                rec._log("control: " + rec._describe(ev))
            hook(combo.currentIndexChanged, combo_slot)

        # Camera / screenshot. snap() runs first (it was connected at GUI
        # construction), so by the time this fires the save path has already
        # been captured by the getSaveFileName wrapper below.
        cam = getattr(self.win, "cam_button", None)
        if cam is not None:
            def cam_slot(*_a):
                if not rec.recording:
                    return
                path = self._last_save_path
                ext = os.path.splitext(path)[1] if path else ""
                rec._emit({"op": "save", "method": "snap",
                           "ext": ext.lower() or ".png"})
                rec._log("control: screenshot")
                self._last_save_path = None
            hook(cam.clicked, cam_slot)

    def _disconnect_controls(self):
        for signal, slot in getattr(self, "_ctrl_conns", []) or []:
            try:
                signal.disconnect(slot)
            except Exception:
                pass
        self._ctrl_conns = []

    def _wrap_context_menus_for_recording(self):
        """Capture right-click actions on the canvas and on tables.

        Context menus are rebuilt on every right-click, so there is no stable
        object to hook. Instead we mark *which* widget is opening a menu
        (``_ctx_pending``), then patch ``QMenu.exec`` so that any menu shown
        while that mark is set reports the action the user chose.
        """
        rec = self

        # -- canvas: ImageViewerWindow.create_context_menu(event) ----------
        orig_ctx = getattr(self.win, "create_context_menu", None)
        if callable(orig_ctx):
            bound = orig_ctx

            def wrapped_ctx(event, *a, **k):
                prev = rec._ctx_pending
                try:
                    x = int(round(event.xdata))
                    y = int(round(event.ydata))
                except Exception:
                    x = y = None
                rec._ctx_pending = {"source": "canvas", "x": x, "y": y}
                try:
                    return bound(event, *a, **k)
                finally:
                    rec._ctx_pending = prev

            self.win.create_context_menu = wrapped_ctx
            self._orig[("inst", "create_context_menu")] = bound

        # -- tables: CustomTableView.show_context_menu(position) -----------
        CTV = getattr(self.gui_mod, "CustomTableView", None)
        if CTV is not None and hasattr(CTV, "show_context_menu"):
            orig_show = CTV.show_context_menu
            self._orig[("cls", CTV, "show_context_menu")] = orig_show

            def wrapped_show(self_tbl, position, *a, **k):
                prev = rec._ctx_pending
                tab, kind = rec._identify_table(self_tbl)
                rec._ctx_pending = {"source": "table", "tab": tab,
                                    "table_kind": kind}
                try:
                    return orig_show(self_tbl, position, *a, **k)
                finally:
                    rec._ctx_pending = prev

            CTV.show_context_menu = wrapped_show

        # -- QMenu.exec: observe which action the user picks ---------------
        orig_exec = getattr(QMenu, "exec")
        self._orig[("qmenu", "exec")] = orig_exec

        def wrapped_exec(menu_self, *a, **k):
            ctx = rec._ctx_pending
            if not (rec.recording and ctx) or rec._is_internal_menu(menu_self):
                return orig_exec(menu_self, *a, **k)
            # Submenus in this GUI are parented to the main window rather than
            # to their parent menu, so QMenu.triggered does not propagate. We
            # connect every menu in the tree and de-duplicate instead.
            state = {"done": False}
            conns = []
            rec._connect_menu_tree(menu_self, menu_self, dict(ctx), state, conns)
            try:
                return orig_exec(menu_self, *a, **k)
            finally:
                for m, slot in conns:
                    try:
                        m.triggered.disconnect(slot)
                    except Exception:
                        pass

        _patch_qt_method(QMenu, "exec", wrapped_exec)

    def _connect_menu_tree(self, top, menu, ctx, state, conns, depth=0):
        """Connect ``triggered`` on ``menu`` and every submenu beneath it."""
        if depth > 8:
            return
        rec = self

        def slot(act, _top=top, _ctx=ctx, _state=state):
            if _state["done"]:
                return          # already captured this pick
            _state["done"] = True
            rec._on_ctx_triggered(_top, act, _ctx)

        try:
            menu.triggered.connect(slot)
            conns.append((menu, slot))
        except Exception:
            return
        try:
            for act in menu.actions():
                sub = act.menu()
                if sub is not None:
                    self._connect_menu_tree(top, sub, ctx, state, conns,
                                            depth + 1)
        except Exception:
            pass

    def _is_internal_menu(self, menu) -> bool:
        try:
            w = menu.parentWidget()
            win = w.window() if w is not None else None
            return bool(getattr(win, "_macro_internal", False))
        except Exception:
            return False

    def _on_ctx_triggered(self, menu, action, ctx):
        """A context-menu action fired: record it by its label path."""
        if not self.recording:
            return
        try:
            path = self._action_path(menu, action)
            if not path:
                return
            ev = {"op": "ctx"}
            ev.update(ctx)
            ev["path"] = path
            self._emit(ev)
            self._log("right-click: " + " > ".join(path))
        except Exception:
            pass

    @staticmethod
    def _action_path(menu, target):
        """Depth-first search for ``target``, returning its label path."""
        def walk(m, trail):
            for act in m.actions():
                if act.isSeparator():
                    continue
                label = _clean(act.text())
                sub = act.menu()
                if sub is not None:
                    found = walk(sub, trail + [label])
                    if found:
                        return found
                elif act is target:
                    return trail + [label]
            return None
        return walk(menu, [])

    def _wrap_open_dialogs_for_recording(self):
        rec = self

        def _internal(args):
            # Skip the recorder's own file dialogs (parent is our window).
            try:
                w = args[0] if args else None
                if w is rec:
                    return True
                win = w.window() if hasattr(w, "window") else None
                return bool(getattr(win, "_macro_internal", False))
            except Exception:
                return False

        # Single-file open.
        orig_open = QFileDialog.getOpenFileName
        self._orig[("qfd", "getOpenFileName")] = orig_open

        def wrapped_open(*a, **k):
            result = orig_open(*a, **k)
            try:
                if rec.recording and not _internal(a):
                    path = result[0] if isinstance(result, tuple) else result
                    if path:
                        rec._emit({"op": "open", "method": "getOpenFileName",
                                   "path": path,
                                   "caption": _clean(a[1]) if len(a) > 1 else ""})
                        rec._log(f"open: {os.path.basename(path)}")
            except Exception:
                pass
            return result

        QFileDialog.getOpenFileName = staticmethod(wrapped_open).__func__

        # Multi-file open (getOpenFileNames).
        orig_multi = QFileDialog.getOpenFileNames
        self._orig[("qfd", "getOpenFileNames")] = orig_multi

        def wrapped_multi(*a, **k):
            result = orig_multi(*a, **k)
            try:
                if rec.recording and not _internal(a):
                    paths = result[0] if isinstance(result, tuple) else result
                    if paths:
                        rec._emit({"op": "open", "method": "getOpenFileNames",
                                   "paths": list(paths),
                                   "caption": _clean(a[1]) if len(a) > 1 else ""})
                        rec._log(f"open: {len(paths)} file(s)")
            except Exception:
                pass
            return result

        QFileDialog.getOpenFileNames = staticmethod(wrapped_multi).__func__

        # Observe (do not alter) save paths so the camera button can learn
        # which file extension the user chose for screenshots.
        orig_save = QFileDialog.getSaveFileName
        self._orig[("qfd", "getSaveFileName")] = orig_save

        def wrapped_save_obs(*a, **k):
            result = orig_save(*a, **k)
            try:
                if rec.recording:
                    path = result[0] if isinstance(result, tuple) else result
                    if path:
                        rec._last_save_path = path
                        if not _internal(a):
                            # Generic marker: lets ANY save dialog be
                            # redirected at playback, including ones inside
                            # third-party plugins that we know nothing about.
                            stem, ext = os.path.splitext(
                                os.path.basename(path))
                            rec._emit({"op": "filesave", "stem": stem,
                                       "ext": ext})
            except Exception:
                pass
            return result

        QFileDialog.getSaveFileName = staticmethod(wrapped_save_obs).__func__

    def _wrap_message_boxes_for_recording(self):
        """Record only the message boxes that carry a real choice.

        ``information``/``warning``/``critical`` are notices with a single OK
        button -- there is nothing to record, and recording them would desync
        the answer queue on sessions where the notice happens not to appear.
        They are auto-dismissed at playback instead.
        """
        rec = self

        orig_q = QMessageBox.question
        self._orig[("mbox", "question")] = orig_q

        def wrapped_q(*a, **k):
            result = orig_q(*a, **k)
            try:
                if rec.recording and not rec._internal_msg:
                    rec._emit({"op": "msgbox", "method": "question",
                               "key": rec._msg_key_from_args(a),
                               "button": (int(result)
                                          if result is not None else None)})
            except Exception:
                pass
            return result

        QMessageBox.question = staticmethod(wrapped_q).__func__

        # Instance-style boxes: msg = QMessageBox(); ...; msg.exec()
        orig_exec = QMessageBox.exec
        self._orig[("mboxi", "exec")] = orig_exec

        def wrapped_exec(box_self, *a, **k):
            result = orig_exec(box_self, *a, **k)
            try:
                if rec.recording and not rec._internal_msg:
                    rec._emit({"op": "msgbox", "method": "exec",
                               "key": rec._msg_key_from_box(box_self),
                               "button": (int(result)
                                          if result is not None else None)})
            except Exception:
                pass
            return result

        _patch_qt_method(QMessageBox, "exec", wrapped_exec)

    @staticmethod
    def _msg_key_from_args(args):
        """Identify a static message box by its title + text."""
        try:
            title = _clean(args[1]) if len(args) > 1 else ""
            text = _clean(args[2]) if len(args) > 2 else ""
        except Exception:
            title = text = ""
        return (title + "|" + text)[:200]

    @staticmethod
    def _msg_key_from_box(box):
        try:
            return (_clean(box.windowTitle()) + "|" + _clean(box.text())
                    + "|" + _clean(box.informativeText()))[:200]
        except Exception:
            return ""

    # -- message-box answering helpers -----------------------------------
    #
    # An inline ``QMessageBox()`` has no class of its own and no properties we
    # can address, so playback cannot "click" anything by name. Instead we
    # work out which standard button the box offers, decide on one, and close
    # the box with it -- these helpers do the deciding.

    @staticmethod
    def _as_standard_button(value):
        """Coerce a recorded integer back into a StandardButton, or None."""
        if value is None:
            return None
        try:
            return QMessageBox.StandardButton(int(value))
        except Exception:
            return None

    @staticmethod
    def _button_name(btn):
        """Readable name for a StandardButton, for the log."""
        try:
            return getattr(btn, "name", None) or str(btn)
        except Exception:
            return str(btn)

    @staticmethod
    def _box_offers(box, flag):
        """True if *box* actually has the given standard button."""
        try:
            return bool(box.standardButtons() & flag)
        except Exception:
            return False

    @staticmethod
    def _pick_flag_button(flags, order):
        """First button in *order* that is present in the *flags* set."""
        if flags is None:
            return None
        SB = QMessageBox.StandardButton
        for name in order:
            cand = getattr(SB, name, None)
            if cand is None:
                continue
            try:
                if flags & cand:
                    return cand
            except Exception:
                continue
        return None

    @classmethod
    def _pick_standard_button(cls, box, order):
        try:
            flags = box.standardButtons()
        except Exception:
            return None
        return cls._pick_flag_button(flags, order)

    @staticmethod
    def _buttons_from_static_args(args, kwargs):
        """The StandardButton set a static box was called with, if any.

        ``QMessageBox.question(parent, title, text, buttons, defaultButton)``
        -- so the button set is positional index 3 (or the ``buttons``
        keyword). None means the call relied on Qt's own default set.
        """
        value = kwargs.get("buttons")
        if value is None and len(args) > 3:
            value = args[3]
        if value is None:
            return None
        try:
            if isinstance(value, QMessageBox.StandardButton):
                return value
            return QMessageBox.StandardButton(int(value))
        except Exception:
            return None

    @staticmethod
    def _custom_accept_button(box):
        """An ``addButton``-style (non-standard) button meaning "go ahead".

        Boxes built entirely from custom buttons report ``NoButton`` for
        ``standardButtons()``, so they need to be answered by clicking a real
        QAbstractButton rather than by a standard-button code.
        """
        SB = QMessageBox.StandardButton
        try:
            buttons = list(box.buttons())
        except Exception:
            return None
        custom = []
        for b in buttons:
            try:
                if box.standardButton(b) == SB.NoButton:
                    custom.append(b)
            except Exception:
                pass
        if not custom:
            return None
        BR = QMessageBox.ButtonRole
        for role in (BR.YesRole, BR.AcceptRole, BR.ApplyRole, BR.ActionRole):
            for b in custom:
                try:
                    if box.buttonRole(b) == role:
                        return b
                except Exception:
                    pass
        for b in custom:
            try:
                if _clean(b.text()).lower() in _ACCEPT_HINTS:
                    return b
            except Exception:
                pass
        return custom[0]

    @staticmethod
    def _default_button_for(box):
        """Pick a safe answer for an unrecorded instance message box."""
        SB = QMessageBox.StandardButton
        try:
            d = box.defaultButton()
            if d is not None:
                sb = box.standardButton(d)
                if sb != SB.NoButton:
                    return sb
        except Exception:
            pass
        try:
            std = box.standardButtons()
            for cand in (SB.Ok, SB.Yes, SB.Close, SB.Cancel):
                if std & cand:
                    return cand
        except Exception:
            pass
        return SB.Ok

    def _unwrap_saves(self):
        self._disconnect_controls()
        for key, original in list(self._orig.items()):
            try:
                if key[0] == "cls":
                    _, cls, mname = key
                    setattr(cls, mname, original)
                elif key[0] == "inst":
                    _, mname = key
                    # Remove instance shadow -> class method resurfaces.
                    if mname in self.win.__dict__:
                        delattr(self.win, mname)
                elif key[0] == "mbox":
                    _, mname = key
                    setattr(QMessageBox, mname, original)   # statics
                elif key[0] == "mboxi":
                    _, mname = key
                    _restore_qt_method(QMessageBox, mname, original)
                elif key[0] == "qfd":
                    _, mname = key
                    setattr(QFileDialog, mname, original)
                elif key[0] == "qmenu":
                    _, mname = key
                    _restore_qt_method(QMenu, mname, original)
            except Exception:
                pass
        self._orig.clear()
        _verify_qt_methods()

    def _identify_table(self, table):
        """Return ``(tab_name, kind)`` for a CustomTableView instance."""
        try:
            tabbed = getattr(self.win, "tabbed_data", None)
            if tabbed is not None:
                for name, tbl in tabbed.tables.items():
                    if tbl is table:
                        return name, "top"
        except Exception:
            pass
        if table is getattr(self.win, "network_table", None):
            return "Network", "network"
        if table is getattr(self.win, "selection_table", None):
            return "Selection", "selection"
        return None, "other"

    # -- recording helpers -----------------------------------------------

    def _emit(self, event: dict):
        self.events.append(event)
        self._update_status()

    # ============================================================ SCRIPT I/O

    def _write_macro_script(self, path, events):
        lines = []
        lines.append('"""')
        lines.append("NetTracer3D macro (auto-generated).")
        lines.append("")
        lines.append(f"Recorded: {datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"Steps: {len(events)}")
        lines.append("")
        lines.append("This file is data, not a program: the Macro Recorder "
                     "reads the EVENTS")
        lines.append("list below and replays it. You may hand-edit values "
                     "(e.g. change a")
        lines.append("threshold) as long as the structure stays intact.")
        lines.append('"""')
        lines.append("")
        lines.append("MACRO_FORMAT = 1")
        lines.append("")
        lines.append("EVENTS = [")
        for ev in events:
            comment = self._describe(ev)
            lines.append(f"    {ev!r},"
                         + (f"  # {comment}" if comment else ""))
        lines.append("]")
        lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _describe(ev) -> str:
        op = ev.get("op")
        if op == "menu":
            return "menu: " + " > ".join(ev.get("path", []))
        if op == "dialog_open":
            return f"opens dialog {ev.get('cls')}"
        if op == "set":
            return f"set {ev.get('hint') or ev.get('w_class')} = {ev.get('value')!r}"
        if op == "click":
            return f"click '{ev.get('button_text')}'"
        if op == "save":
            if ev.get("method") == "save_table_as":
                return f"save table '{ev.get('tab')}' ({ev.get('fmt')})"
            if ev.get("method") == "snap":
                return f"screenshot ({ev.get('ext', '.png')})"
            return f"save via {ev.get('method')}"
        if op == "msgbox":
            return f"answer {ev.get('method')} dialog"
        if op == "filesave":
            return f"save file '{ev.get('stem','')}{ev.get('ext','')}'"
        if op == "open":
            if ev.get("method") == "getOpenFileNames":
                n = len(ev.get("paths") or [])
                return f"load {n} file(s)"
            import os as _os
            return f"load file '{_os.path.basename(ev.get('path') or '')}'"
        if op == "ctx":
            src = "table" if ev.get("source") == "table" else "image"
            return f"right-click ({src}): " + " > ".join(ev.get("path", []))
        if op == "ui":
            t = ev.get("target")
            state = "on" if ev.get("checked") else "off"
            if t == "home":
                return "reset view (home)"
            if t == "scalebar":
                return f"scalebar {state}"
            if t == "highlight":
                return f"highlight overlay {state}"
            if t == "channel":
                names = ["Nodes", "Edges", "Overlay 1", "Overlay 2"]
                i = ev.get("index", 0)
                nm = names[i] if 0 <= i < len(names) else f"channel {i}"
                return f"channel '{nm}' {state}"
            if t == "active_image":
                names = ["Nodes", "Edges", "Overlay 1", "Overlay 2"]
                i = ev.get("index", 0)
                nm = names[i] if 0 <= i < len(names) else str(i)
                return f"active image -> {nm}"
            return f"control: {t}"
        return ""

    def load_macro(self):
        if self.recording:
            self._warn("Busy", "Stop recording before loading a macro.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Macro", "", "Python Macro (*.py);;All Files (*)")
        if not path:
            return
        try:
            events = self._read_macro_script(path)
        except Exception as e:
            self._warn("Load Failed", f"Could not read macro:\n{e}")
            return
        if not events:
            self._warn("Empty Macro", "That file contained no EVENTS.")
            return
        self.current_events = events
        self.current_macro_name = os.path.basename(path)
        self._log(f"Loaded {len(events)} step(s) from {path}")
        self._update_status()

    @staticmethod
    def _read_macro_script(path):
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        namespace = {}
        # The macro file only defines literals (EVENTS list); exec in a
        # clean namespace and pull EVENTS out.
        exec(compile(source, path, "exec"), namespace)
        events = namespace.get("EVENTS")
        if not isinstance(events, list):
            raise ValueError("No EVENTS list found in macro file.")
        return events

    # ================================================================== RUN

    def run_macro(self):
        if self.recording:
            self._warn("Busy", "Stop recording before running.")
            return
        if self._playing:
            return
        if not self.current_events:
            self._warn("No Macro", "Record or load a macro first.")
            return

        parent = QFileDialog.getExistingDirectory(
            self, "Select Parent Folder Containing Session Sub-folders", "")
        if not parent:
            return

        # Snapshot session sub-folders up front (we'll be creating new ones).
        try:
            sessions = []
            for entry in os.scandir(parent):
                if (entry.is_dir()
                        and not entry.name.startswith(".")
                        and not entry.name.endswith("_Output")
                        and not entry.name.startswith(BATCH_PREFIX)):
                    sessions.append(entry.path)
            sessions.sort()
        except Exception as e:
            self._warn("Scan Failed", f"Could not read folder:\n{e}")
            return

        if not sessions:
            self._warn("No Sessions",
                       "No candidate session sub-folders were found.")
            return

        self._internal_msg = True
        try:
            proceed = QMessageBox.question(
                self, "Run Macro",
                f"Found {len(sessions)} folder(s) under:\n{parent}\n\n"
                f"Run the macro on each?\n\n"
                f"Outputs go to a new timestamped "
                f"'{BATCH_PREFIX}...' folder,\n"
                f"one '<name>{OUTPUT_SUFFIX}' sub-folder per session.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
        finally:
            self._internal_msg = False
        if proceed != QMessageBox.StandardButton.Yes:
            return

        self._start_batch(parent, sessions)

    # -- batch driver ----------------------------------------------------

    def _start_batch(self, parent, sessions):
        self._aborted = False
        self._playing = True
        self.btn_run.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(sessions))
        self.progress.setValue(0)

        # All outputs for this run live under one timestamped master folder so
        # repeat runs never overwrite or mix with earlier batch results. The
        # folder is *not* created here: it is created on the first actual save
        # (see _ensure_output_dir), so a macro that saves nothing leaves no
        # empty folders behind. Writability is still checked up front so the
        # failure is reported before the run rather than halfway through it.
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        batch_root = self._unique_path(
            os.path.join(parent, BATCH_PREFIX + stamp))
        if not os.access(parent, os.W_OK):
            self._warn("Cannot Write",
                       "Outputs cannot be written to:\n"
                       f"{parent}\n\nChoose a writable folder.")
            self._playing = False
            self.btn_run.setEnabled(True)
            self.btn_start.setEnabled(True)
            self.btn_load.setEnabled(True)
            self.progress.setVisible(False)
            return

        self._made_dirs = set()
        self._batch = {"parent": parent, "sessions": sessions,
                       "root": batch_root, "i": 0, "results": [],
                       "made_output": False}

        # Install the playback patches (dialogs + saves redirect) and the
        # dialog-tracking event filter.
        self._install_playback_patches()
        QApplication.instance().installEventFilter(self)

        self._log(f"=== Batch start: {len(sessions)} session(s) ===")
        self._log(f"    output folder (created on first save): "
                  f"{os.path.basename(batch_root)}")
        QTimer.singleShot(STEP_DELAY_MS, self._next_session)

    def _next_session(self):
        if self._aborted:
            return
        b = self._batch
        if b is None:
            return
        if b["i"] >= len(b["sessions"]):
            self._finish_batch()
            return

        folder = b["sessions"][b["i"]]
        name = os.path.basename(folder.rstrip("/\\"))
        outdir = os.path.join(b.get("root") or b["parent"],
                              self._safe_name(name) + OUTPUT_SUFFIX)
        self._current_output_dir = outdir
        # Names are reserved per session, not per batch: two sessions writing
        # 'network_table_<session>.csv' into their own folders must not push
        # the second one to '..._2'.
        self._reserved_paths = set()
        self._notag_paths = set()
        self._session_dirs = set()
        self._deferred_save_path = None
        self._deferred_save_step = -1
        self._save_stem_hint = None
        self._session_save_target = None
        self._session_save_step = -1
        self.progress.setValue(b["i"])
        self._log(f"--- Session {b['i'] + 1}/{len(b['sessions'])}: {name} ---")

        # Load the session; failures are non-fatal. Note that *outdir* is only
        # a planned path at this point -- nothing is created on disk until a
        # save actually happens (_ensure_output_dir).
        try:
            self._prepare_and_load(folder)
            if not self._session_loaded_ok():
                raise RuntimeError("session did not load (incompatible folder)")
        except Exception as e:
            self._log(f"  LOAD FAILED: {e}")
            b["results"].append((name, "LOAD FAILED", str(e)))
            b["i"] += 1
            QTimer.singleShot(STEP_DELAY_MS, self._next_session)
            return

        # A plain Save writes to wherever the window was last saved, with no
        # dialog to intercept -- and loading a session sets that to the source
        # folder. Repoint it at this session's output folder before the macro
        # runs so nothing can write back over the input.
        self._redirect_plain_save(outdir)

        # Refresh recorded answers/paths so every session replays identically.
        self._reset_playback_fifos()

        # Replay the macro on this session, then advance.
        self._play_events(self.current_events, self._on_session_done)

    def _redirect_plain_save(self, outdir):
        """Aim the host window's implicit save target at the batch output.

        ``save_network_3d(False)`` -- the plain *Save* -- does not open a
        dialog: it writes to ``last_saved``/``last_save_name`` directly. Those
        are set to the source folder when a session is loaded, so a recorded
        plain Save would overwrite the very session the batch is iterating
        over. Setting them here means such a save lands in the output folder
        like everything else. They are deliberately *not* restored afterwards:
        leaving them pointed at the output is the safe direction to fail in.
        """
        if not REDIRECT_PLAIN_SAVE or not outdir:
            return
        target = (os.path.join(outdir, self._safe_name(self._session_basename()))
                  if SESSION_SAVE_SUBFOLDER else outdir)
        try:
            if hasattr(self.win, "last_saved"):
                self.win.last_saved = os.path.dirname(target)
            if hasattr(self.win, "last_save_name"):
                self.win.last_save_name = os.path.basename(target)
        except Exception as e:
            self._log(f"  [note] could not repoint plain Save: {e}")

    def _prepare_and_load(self, folder):
        # Clear stale output tables so we never save another session's data.
        try:
            tabbed = getattr(self.win, "tabbed_data", None)
            if tabbed is not None:
                tabbed.clear_all_tabs()
        except Exception:
            pass
        # Load the session (explicit directory -> no dialog).
        self.win.load_from_network_obj(folder)

    def _session_loaded_ok(self) -> bool:
        try:
            net = self.api.get_network()
            if net is not None:
                return True
        except Exception:
            pass
        try:
            for ch in self.win.channel_data:
                if ch is not None:
                    return True
        except Exception:
            pass
        return False

    def _on_session_done(self, status, msg):
        b = self._batch
        if b is None:
            return
        name = os.path.basename(b["sessions"][b["i"]].rstrip("/\\"))
        b["results"].append((name, status, msg))
        # Some save routines are connected at GUI construction and name their
        # own files (e.g. the quickload pickles reached via right-click), so
        # sweep the output folder and tag anything still untagged. Files we
        # named ourselves already carry the suffix and are skipped by
        # _tag_new_files -- passing an empty 'before' set here would otherwise
        # re-tag the whole folder on every session.
        try:
            outdir = self._current_output_dir
            # Nothing to sweep if this session never saved: the folder was
            # never created.
            if outdir and os.path.isdir(outdir):
                self._tag_new_files(outdir, set())
        except Exception:
            pass
        self._log(f"  session '{name}': {status}"
                  + (f" ({msg})" if msg else ""))
        b["i"] += 1
        self._close_stray_dialogs()
        try:
            self._log_session_memory(*self._reap_between_sessions())
        except Exception as e:
            self._log(f"  [note] cleanup between sessions: {e}")
        QTimer.singleShot(STEP_DELAY_MS, self._next_session)

    def _finish_batch(self):
        if self._aborted:
            return
        results = self._batch["results"] if self._batch else []
        root = (self._batch or {}).get("root")
        made_output = bool((self._batch or {}).get("made_output"))
        self.progress.setValue(self.progress.maximum())
        self._remove_playback_patches()
        QApplication.instance().removeEventFilter(self)
        self._playing = False
        self._batch = None
        self._pb = None
        self._pb_dialog = None
        self._current_output_dir = None
        self._made_dirs = set()
        self._reserved_paths = set()
        self._notag_paths = set()
        self._session_dirs = set()
        self._deferred_save_path = None
        self._deferred_save_step = -1
        self._save_stem_hint = None
        self._session_save_target = None
        self._session_save_step = -1

        self.btn_run.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.progress.setVisible(False)

        ok = sum(1 for _, s, _ in results if s == "OK")
        self._log(f"=== Batch complete: {ok}/{len(results)} succeeded ===")
        for name, status, msg in results:
            if status != "OK":
                self._log(f"    \u2717 {name}: {status} {msg}")
        if not made_output:
            self._log("    no files were saved -- no output folder created")
        self._info("Batch Complete",
                   f"Processed {len(results)} session(s).\n"
                   f"{ok} succeeded, {len(results) - ok} had problems.\n\n"
                   + (f"Outputs: {root}\n\n" if (root and made_output)
                      else "This macro saved no files, so no output "
                           "folder was created.\n\n")
                   + "See the log for details.")

    # -- playback engine (async, modal-safe) -----------------------------

    # Ops that can make the application run its own save routine.
    _GESTURE_OPS = ("ctx", "menu", "click", "ui")
    # Passive markers the recorder writes between a gesture and its save.
    _MARKER_OPS = ("filesave", "dialog_open", "msgbox", "open")

    @classmethod
    def _pair_saves(cls, events):
        """Work out which 'save' events are echoes of a recorded gesture.

        One "Save As > CSV" right-click is recorded as *three* events: the
        'ctx' gesture, a 'filesave' marker, and a 'save' event emitted by the
        wrapped ``save_table_as``. Replaying the gesture already makes the
        application save, so replaying the save event as well writes the file
        a second time. Instead the save event is used only to *name* the write
        the gesture is about to perform.

        Returns a dict with:
          ``save_for``   gesture index -> the save event it produced
          ``skip``       indices of save events not to execute on their own
          ``claimed``    indices of filesave markers consumed by such a pair
        """
        plan = {"save_for": {}, "skip": set(), "claimed": set()}

        # save_network_3d(False) falls back to calling self.save_network_3d()
        # when nothing has been saved yet -- and that attribute is the *wrapped*
        # method, so one Save is recorded as two identical save events. Replaying
        # both writes the session twice. Drop the outer echo.
        for idx in range(1, len(events)):
            ev, prev = events[idx], events[idx - 1]
            if (ev.get("op") == "save" and prev.get("op") == "save"
                    and ev.get("method") == prev.get("method")
                    and ev.get("method") in SESSION_SAVE_METHODS):
                plan["skip"].add(idx)

        for idx, ev in enumerate(events):
            if ev.get("op") != "save" or idx in plan["skip"]:
                continue
            markers, j = [], idx - 1
            while j >= 0 and events[j].get("op") in cls._MARKER_OPS:
                if events[j].get("op") == "filesave":
                    markers.append(j)
                j -= 1
            if j < 0 or events[j].get("op") not in cls._GESTURE_OPS:
                continue                      # a save with no gesture: replay it
            if j in plan["save_for"]:
                continue                      # gesture already owns a save
            plan["save_for"][j] = ev
            plan["skip"].add(idx)
            plan["claimed"].update(markers)
        return plan

    def _play_events(self, events, on_finished):
        self._pb = {"events": events, "i": 0, "on_finished": on_finished,
                    "retries": 0, "done": False,
                    "plan": self._pair_saves(events)}
        QTimer.singleShot(0, self._pb_step)

    def _pb_finish(self, status, msg):
        pb = self._pb
        self._pb = None
        if pb and not pb["done"]:
            pb["done"] = True
            cb = pb.get("on_finished")
            if cb:
                cb(status, msg)

    def _pb_step(self):
        if self._aborted:
            return
        pb = self._pb
        if pb is None or pb["done"]:
            return
        if pb["i"] >= len(pb["events"]):
            self._pb_finish("OK", "")
            return

        idx = pb["i"]
        ev = pb["events"][idx]
        op = ev.get("op")

        # A save that is merely the echo of the gesture on the line above is
        # not replayed: the gesture itself does the writing (see _pair_saves).
        if idx in pb["plan"]["skip"]:
            pb["i"] += 1
            QTimer.singleShot(0, self._pb_step)
            return

        # Dialog-scoped ops need a live dialog; wait for it to appear.
        if op in ("set", "click") and ev.get("scope", "dialog") == "dialog":
            if self._pb_dialog is None or not self._pb_dialog.isVisible():
                if pb["retries"] < DIALOG_WAIT_RETRIES:
                    pb["retries"] += 1
                    QTimer.singleShot(DIALOG_WAIT_MS, self._pb_step)
                    return
                self._log(f"  [skip] no dialog for {op} "
                          f"({ev.get('hint') or ev.get('button_text', '')})")
                pb["retries"] = 0
                pb["i"] += 1
                QTimer.singleShot(STEP_DELAY_MS, self._pb_step)
                return
        pb["retries"] = 0
        pb["i"] += 1

        # Schedule the continuation *before* executing, so that if this step
        # opens a modal dialog (exec()), the next steps still run inside its
        # nested event loop.
        QTimer.singleShot(STEP_DELAY_MS, self._pb_step)
        # If this gesture is known to trigger a save, decide the filename now
        # so the patched save dialog can hand it straight back.
        finish_save = None
        save_ev = pb["plan"]["save_for"].get(idx)
        if save_ev is not None:
            try:
                finish_save = self._arm_save(save_ev)
            except Exception as e:
                self._log(f"  [error] arming save: {e}")
        # A gesture that triggers a whole-session save runs with the working
        # directory at the destination, so that dump()'s directory-less
        # fallback still lands in the right folder (see SESSION_SAVE_CHDIR).
        guard = (self._chdir(self._session_save_target)
                 if self._session_save_target else contextlib.nullcontext())
        with guard:
            try:
                self._pb_execute(ev)
            except Exception as e:
                self._log(f"  [error] {op}: {e}")
            finally:
                if finish_save is not None:
                    try:
                        finish_save()
                    except Exception:
                        pass

    def _pb_execute(self, ev):
        op = ev.get("op")
        if op == "menu":
            self._exec_menu(ev)
        elif op == "set":
            self._exec_set(ev)
        elif op == "click":
            self._exec_click(ev)
        elif op == "save":
            self._exec_save(ev)
        elif op == "ctx":
            self._exec_ctx(ev)
        elif op == "ui":
            self._exec_ui(ev)
        # dialog_open / msgbox / open: passive markers. The real file-open
        # happens when the triggering action calls the patched getOpenFileName.

    @staticmethod
    def _action_path_lookup(menu, path):
        """Find the action at ``path`` (list of labels) within ``menu``."""
        node = menu
        for depth, label in enumerate(path):
            match = None
            for act in node.actions():
                if act.isSeparator():
                    continue
                if _clean(act.text()) == label:
                    match = act
                    break
            if match is None:
                return None
            sub = match.menu()
            if depth == len(path) - 1:
                # Final component must be a leaf action, not a submenu.
                return match if sub is None else None
            if sub is None:
                return None
            node = sub
        return None

    def _exec_ctx(self, ev):
        """Replay a right-click action by rebuilding its context menu.

        The GUI builds these menus inside ``create_context_menu`` /
        ``show_context_menu`` and then blocks in ``QMenu.exec``. During
        playback ``exec`` is patched to look up ``_pb_ctx_target`` in the
        menu, trigger it, and return immediately -- so no menu is displayed
        and nothing depends on real cursor position.
        """
        path = ev.get("path") or []
        if not path:
            return
        self._pb_ctx_target = path
        self._pb_ctx_hit = False
        label = " > ".join(path)
        try:
            if ev.get("source") == "table":
                self._open_table_ctx(ev)
            else:
                self._open_canvas_ctx(ev)
        except Exception as e:
            self._log(f"  [skip] right-click '{label}': {e}")
            return
        finally:
            self._pb_ctx_target = None

        if self._pb_ctx_hit:
            self._log("  right-click: " + label)
        else:
            self._log(f"  [skip] right-click item not available: {label}")

    def _open_canvas_ctx(self, ev):
        fn = getattr(self.win, "create_context_menu", None)
        if not callable(fn):
            raise RuntimeError("canvas context menu unavailable")

        class _SynthEvent:
            pass

        e = _SynthEvent()
        # Coordinates only matter for point/measurement items; anything else
        # (identities, communities, select-all) ignores them.
        e.xdata = ev.get("x") if ev.get("x") is not None else 0
        e.ydata = ev.get("y") if ev.get("y") is not None else 0
        e.inaxes = getattr(self.win, "ax", None)
        e.button = 3
        fn(e)

    def _open_table_ctx(self, ev):
        table = self._resolve_table(ev)
        if table is None:
            raise RuntimeError("table not found: %s" % ev.get("tab"))
        fn = getattr(table, "show_context_menu", None)
        if not callable(fn):
            raise RuntimeError("table context menu unavailable")
        # show_context_menu needs a position over a valid cell.
        pos = None
        try:
            model = table.model()
            if model is not None and model.rowCount() > 0:
                idx = model.index(0, 0)
                pos = table.visualRect(idx).center()
        except Exception:
            pos = None
        if pos is None:
            from PyQt6.QtCore import QPoint
            pos = QPoint(1, 1)
        fn(pos)

    def _exec_ui(self, ev):
        """Replay an on-screen control.

        Checkable controls are driven to the *state* that was recorded rather
        than blindly clicked, because a new session may start with different
        toggles already set (e.g. a channel that loaded visible).
        """
        target = ev.get("target")
        label = self._describe(ev)
        try:
            if target == "home":
                btn = getattr(self.win, "reset_view", None)
                if btn is None:
                    raise RuntimeError("no reset button")
                btn.click()

            elif target == "active_image":
                combo = getattr(self.win, "active_channel_combo", None)
                if combo is None:
                    raise RuntimeError("no active-image selector")
                idx = int(ev.get("index", 0))
                if idx >= combo.count():
                    self._log(f"  [skip] {label}: only {combo.count()} images")
                    return
                combo.setCurrentIndex(idx)

            elif target == "channel":
                btns = getattr(self.win, "channel_buttons", None) or []
                idx = int(ev.get("index", 0))
                if idx >= len(btns):
                    self._log(f"  [skip] {label}: no such channel")
                    return
                if not self._set_checked(btns[idx], ev.get("checked"), label):
                    return

            elif target in ("scalebar", "highlight"):
                attr = "toggle_scale" if target == "scalebar" else "high_button"
                btn = getattr(self.win, attr, None)
                if btn is None:
                    raise RuntimeError(f"no {target} button")
                if not self._set_checked(btn, ev.get("checked"), label):
                    return
            else:
                self._log(f"  [skip] unknown control: {target}")
                return
        except Exception as e:
            self._log(f"  [skip] {label}: {e}")
            return
        self._log("  " + label)

    def _set_checked(self, btn, want, label):
        """Click ``btn`` only if its state differs from ``want``."""
        if not btn.isEnabled():
            self._log(f"  [skip] {label}: control unavailable this session")
            return False
        want = bool(want)
        if bool(btn.isChecked()) == want:
            return True     # already in the recorded state
        btn.click()
        return True

    def _exec_menu(self, ev):
        path = ev.get("path", [])
        action = self._resolve_menu(path)
        if action is None:
            self._log(f"  [skip] menu not found: {' > '.join(path)}")
            return
        self._log("  menu: " + " > ".join(path))
        action.trigger()

    def _resolve_menu(self, path):
        if not path:
            return None
        node = self.win.menuBar()
        for title in path[:-1]:
            found = None
            for act in node.actions():
                if act.menu() is not None and _clean(act.text()) == title:
                    found = act.menu()
                    break
            if found is None:
                return None
            node = found
        for act in node.actions():
            if act.menu() is None and _clean(act.text()) == path[-1]:
                return act
        return None

    def _exec_set(self, ev):
        dlg = self._pb_dialog
        if dlg is None:
            return
        T = _INPUT_TYPE_BY_NAME.get(ev.get("w_class"))
        if T is None:
            return
        widgets = _typed_widgets(dlg, T)
        i = ev.get("w_index", -1)
        if not (0 <= i < len(widgets)):
            self._log(f"  [skip] field {ev.get('w_class')}[{i}] missing")
            return
        _set_value(widgets[i], ev.get("kind"), ev.get("value"))

    def _exec_click(self, ev):
        dlg = self._pb_dialog
        if dlg is None:
            return
        btns = _buttons(dlg)
        target = None
        wanted = ev.get("button_text") or ""
        if wanted:
            for b in btns:
                if _clean(b.text()) == wanted:
                    target = b
                    break
        if target is None:
            idx = ev.get("button_index")
            if isinstance(idx, int) and 0 <= idx < len(btns):
                target = btns[idx]
        if target is None:
            self._log(f"  [skip] button '{wanted}' not found")
            return
        self._log(f"  click: {wanted or '(button)'}")
        target.click()

    def _ensure_output_dir(self):
        """Create (once, on demand) and return this session's output folder.

        Output folders are deliberately lazy. ``_current_output_dir`` is only
        a *planned* path until something is actually written, so a session --
        or a whole batch -- that performs no save leaves no empty
        ``<session>`` directory, and no ``MacroBatch_...`` folder
        either. Every code path that is about to write must call this first;
        returns None if the folder cannot be created.
        """
        outdir = self._current_output_dir
        if not outdir:
            return None
        if outdir in self._made_dirs:
            return outdir
        try:
            # Creates the batch root too, since it is the parent.
            os.makedirs(outdir, exist_ok=True)
        except Exception as e:
            self._log(f"  [error] could not create output folder: {e}")
            return None
        self._made_dirs.add(outdir)
        if self._batch is not None:
            self._batch["made_output"] = True
        self._log(f"  created output folder: {os.path.basename(outdir)}")
        return outdir

    _EXT_BY_FMT = {"csv": ".csv", "xlsx": ".xlsx", "gexf": ".gexf",
                   "graphml": ".graphml", "net": ".net"}
    _CHANNEL_NAMES = {0: "labelled_nodes", 1: "labelled_edges",
                      2: "overlay_1", 3: "overlay_2",
                      4: "highlighted_element"}

    def _session_save_path(self, outdir):
        """Where a whole-session save (a folder of components) should land.

        Named after the session it came from, never after the method, and
        never given the per-file session suffix -- the folder name *is* the
        session name, so appending it again would produce ``sess1_sess1``.
        """
        if not SESSION_SAVE_SUBFOLDER:
            # The per-session output folder is itself the saved session.
            return outdir
        return self._unique_path(
            os.path.join(outdir, self._safe_name(self._session_basename())))

    def _plan_save_path(self, ev, outdir):
        """Full path this save event should be written to, or None.

        None means the filename is not ours to choose (an unknown method).
        """
        method = ev.get("method")
        if method in SESSION_SAVE_METHODS:
            return self._session_save_path(outdir)
        if method == "save_table_as":
            fmt = ev.get("fmt", "csv")
            ext = self._EXT_BY_FMT.get(fmt, "." + fmt)
            base = ev.get("tab") or ev.get("table_kind") or "table"
        elif method == "save":
            ext = ".tif"
            ch = ev.get("ch_index")
            base = self._CHANNEL_NAMES.get(ch, f"channel_{ch}")
        elif method == "snap":
            ext = ev.get("ext") or ".png"
            if not ext.startswith("."):
                ext = "." + ext
            base = "screenshot"
        else:
            return None
        base = self._with_session(self._safe_name(base))
        return self._unique_path(os.path.join(outdir, base + ext))

    def _arm_save(self, ev):
        """Pre-name the write a gesture is about to perform.

        Returns a callable to run once the gesture has finished, or None if
        this save event cannot be armed (in which case nothing is reserved
        and the gesture's save falls back to the recorded-name queue).
        """
        outdir = self._ensure_output_dir()
        if not outdir:
            return None
        method = ev.get("method")
        is_session = method in SESSION_SAVE_METHODS

        target = self._plan_save_path(ev, outdir)
        if not target:
            return None

        # A session save writes a folder full of components, and it may reach
        # for *either* dialog to find out where. Both are covered: the folder
        # dialog hands back this path directly (fake_existing_dir), and the
        # file dialog consumes it as the pending target.
        self._pending_save_path = target
        self._save_stem_hint = (os.path.splitext(os.path.basename(target))[0]
                                if not is_session else None)
        self._session_save_target = target if is_session else None
        self._session_save_step = -1
        before = self._dir_snapshot(target if is_session else outdir)
        name = os.path.basename(target)

        def done():
            if is_session:
                # Contents are the session's own component files. Record them
                # so the end-of-session sweep leaves their names alone.
                self._protect_session_files(target, before)
                n = max(0, len(self._dir_snapshot(target)) - len(before))
                if n:
                    self._log(f"  saved session -> {name}{os.sep} "
                              f"({n} item(s))")
                    self._session_save_target = None
                else:
                    # Nothing on disk yet: the save is still in flight. Keep
                    # the destination reserved for a few more steps so the
                    # components land in the session folder rather than being
                    # scattered under generic names.
                    self._session_save_step = self._pb_index()
                    self._log(f"  (deferred) session folder held for {name}")
            elif self._pending_save_path is None:
                self._log(f"  saved -> {name}")
            else:
                # The gesture has not reached a save dialog *yet*. It may still
                # do so a step or two later -- a menu entry that opens its own
                # dialog, or a save that runs on a queued callback. Dropping
                # the name here is what left those writes unnamed, so they fell
                # through to the 'plugin_output...dat' fallback. Demote it
                # instead: still available to the next unnamed save, but no
                # longer claiming to be the one for *this* step.
                self._deferred_save_path = self._pending_save_path
                self._deferred_save_step = self._pb_index()
                self._pending_save_path = None
                self._log(f"  (deferred) name held for {name}")
        return done

    def _pb_index(self):
        """Current playback step, or -1 when nothing is playing."""
        pb = self._pb
        return pb["i"] if pb else -1

    @contextlib.contextmanager
    def _chdir(self, target):
        """Run a session save with the working directory at its destination.

        See ``SESSION_SAVE_CHDIR``: ``Network_3D.dump()`` falls back to writing
        its components with no directory argument if any one of them raises, so
        the working directory decides where the remainder of a half-failed save
        ends up. Pointing it at the destination turns that scatter into a
        second, complete write into the correct folder.
        """
        if not SESSION_SAVE_CHDIR:
            yield
            return
        try:
            os.makedirs(target, exist_ok=True)
            previous = os.getcwd()
        except Exception:
            yield
            return
        try:
            os.chdir(target)
        except Exception:
            yield
            return
        try:
            yield
        finally:
            try:
                os.chdir(previous)
            except Exception:
                pass

    def _protect_session_files(self, target, before):
        """Remember a session save's component files so nothing renames them.

        A saved session is only re-loadable if its parts keep the names the
        save routine gave them, so the session-suffix sweep must skip them.
        """
        try:
            for fname in self._dir_snapshot(target) - before:
                self._notag_paths.add(os.path.join(target, fname))
            # When the session save lands directly in the output folder, the
            # folder itself must not be touched either.
            self._notag_paths.add(target)
            # A save that finishes late (on a queued callback, or from a dialog
            # the gesture opened) writes its components after this runs, so the
            # destination is remembered as a whole: anything that turns up
            # inside it later is part of the session too.
            self._session_dirs.add(os.path.normpath(target))
        except Exception:
            pass

    def _session_save_fresh(self):
        """Is a session save still the write we should be redirecting?

        True while the save is the step in progress, and for a short window
        afterwards so that components written on a queued callback still reach
        the session folder. After that the reservation is dropped, so an
        unrelated save later in the macro cannot land inside it.
        """
        if not self._session_save_target:
            return False
        if self._session_save_step < 0:
            return True
        if self._pb_index() - self._session_save_step <= DEFERRED_SAVE_WINDOW:
            return True
        self._session_save_target = None
        self._session_save_step = -1
        return False

    def _is_protected(self, path):
        """Is this a saved session's own file, which must not be renamed?"""
        if path in self._notag_paths:
            return True
        norm = os.path.normpath(path)
        for d in self._session_dirs:
            if norm == d or norm.startswith(d + os.sep):
                return True
        return False

    def _exec_save(self, ev):
        """Replay a save event that has no recorded gesture of its own."""
        method = ev.get("method")
        outdir = self._ensure_output_dir()
        if not outdir:
            return
        if method not in SESSION_SAVE_METHODS:
            # A session folder still reserved from an earlier step must not
            # swallow this save.
            self._session_save_target = None
            self._session_save_step = -1
        try:
            if method == "save_table_as":
                self._save_table(ev, outdir)
            elif method == "save":
                self._save_channel(ev, outdir)
            elif method in SESSION_SAVE_METHODS:
                self._save_session(method, outdir)
            elif method == "snap":
                target = self._plan_save_path(ev, outdir)
                self._pending_save_path = target
                try:
                    self.win.snap()
                finally:
                    self._pending_save_path = None
                self._log(f"  saved screenshot -> {os.path.basename(target)}")
        except Exception as e:
            self._log(f"  [error] save {method}: {e}")

    def _save_session(self, method, outdir):
        """Write a whole NetTracer3D session (a folder of components).

        The destination is the session's own folder, whichever dialog the save
        routine reaches for, and its contents keep the names the routine gave
        them so the result can be loaded straight back in.
        """
        fn = getattr(self.win, method, None)
        if fn is None:
            self._log(f"  [skip] {method} not available in this build")
            return
        target = self._session_save_path(outdir)
        before = self._dir_snapshot(target)
        self._pending_save_path = target
        self._session_save_target = target
        self._session_save_step = -1
        try:
            os.makedirs(target, exist_ok=True)
        except Exception:
            pass
        # Call with no arguments wherever the signature allows it. This is not
        # just caution about unknown builds: ``save_network_3d(asbool=True)``
        # defaults to *Save As*, the branch that opens a dialog, which is the
        # branch the redirect can actually intercept. Passing an argument would
        # only be guessing at what it means.
        args = ()
        try:
            import inspect
            required = [q for q in inspect.signature(fn).parameters.values()
                        if q.kind in (q.POSITIONAL_ONLY,
                                      q.POSITIONAL_OR_KEYWORD)
                        and q.default is q.empty]
            if required:
                args = (True,)
        except Exception:
            pass
        with self._chdir(target):
            try:
                fn(*args)
            finally:
                self._pending_save_path = None
                self._session_save_target = None
                self._session_save_step = -1
        self._protect_session_files(target, before)
        n = max(0, len(self._dir_snapshot(target)) - len(before))
        self._log(f"  saved session -> {os.path.basename(target)}"
                  f"{os.sep} ({n} item(s))")

    @staticmethod
    def _dir_snapshot(outdir):
        try:
            return set(os.listdir(outdir))
        except Exception:
            return set()

    def _tag_new_files(self, outdir, before):
        """Append the session name to files the GUI just created itself.

        Used where the save routine chooses its own filenames (e.g. the
        quickload pickles), so those outputs get the same suffix as the ones
        we name directly.
        """
        count = 0
        try:
            after = self._dir_snapshot(outdir)
        except Exception:
            return 0
        suffix = self._safe_name(self._session_basename())
        for fname in sorted(after - before):
            src = os.path.join(outdir, fname)
            try:
                # A saved session's component files must keep the exact names
                # the save routine gave them, or the session will not load
                # back in with all of its sub-properties.
                if self._is_protected(src):
                    continue
                # Anything already carrying this session's suffix was named by
                # us and must be left alone.
                if self._is_tagged(fname, suffix):
                    continue
                stem, ext = os.path.splitext(fname)
                tagged = self._with_session(stem) + ext
                if tagged == fname:
                    continue
                dst = self._unique_path(os.path.join(outdir, tagged))
                if os.path.isdir(src):
                    os.rename(src, dst)
                else:
                    os.replace(src, dst)
                count += 1
            except Exception:
                # Leave the file where it is rather than losing it.
                count += 1
        return count

    @staticmethod
    def _is_tagged(base: str, suffix: str) -> bool:
        """Has this name already been given the session suffix?

        Testing only ``endswith`` is not enough: ``_unique_path`` may have
        turned ``report_patient_1`` into ``report_patient_1_2``, and a save
        routine may have appended its own extension, so the suffix often sits
        in the middle. Tagging such a name a second time is what produced
        ``report_patient_1_2_patient_1``.
        """
        return bool(suffix) and (base == suffix or ("_" + suffix) in base)

    def _with_session(self, base: str) -> str:
        """Append the session folder name so outputs stay unique if pooled.

        Every session writes into its own ``<name>`` folder, but the
        filenames inside would otherwise be identical across sessions, which
        makes them collide as soon as they are moved into one directory.
        Whole-session saves are exempt -- see ``_session_save_path``.
        """
        suffix = self._safe_name(self._session_basename())
        if not suffix or self._is_tagged(base, suffix):
            return base
        return f"{base}_{suffix}"

    def _save_table(self, ev, outdir):
        fmt = ev.get("fmt", "csv")
        table = self._resolve_table(ev)
        if table is None:
            self._log(f"  [skip] table '{ev.get('tab')}' not present")
            return
        target = self._plan_save_path(ev, outdir)
        self._pending_save_path = target
        try:
            table.save_table_as(fmt)
        finally:
            self._pending_save_path = None
        self._log(f"  saved table -> {os.path.basename(target)}")

    def _save_channel(self, ev, outdir):
        ch = ev.get("ch_index")
        target = self._plan_save_path(ev, outdir)
        self._pending_save_path = target
        try:
            self.win.save(ch, True)
        finally:
            self._pending_save_path = None
        self._log(f"  saved image -> {os.path.basename(target)}")

    def _resolve_table(self, ev):
        kind = ev.get("table_kind")
        if kind == "network":
            return getattr(self.win, "network_table", None)
        if kind == "selection":
            return getattr(self.win, "selection_table", None)
        tab = ev.get("tab")
        tabbed = getattr(self.win, "tabbed_data", None)
        if tabbed is not None and tab is not None:
            table = tabbed.tables.get(tab)
            if table is not None:
                return table
            # Fall back to the currently active top table.
            cur = tabbed.get_current_table()
            if cur is not None:
                return cur
        return None

    def _session_basename(self):
        b = self._batch
        if b and 0 <= b["i"] < len(b["sessions"]):
            return os.path.basename(b["sessions"][b["i"]].rstrip("/\\"))
        return "session"

    @staticmethod
    def _safe_name(name):
        keep = "-_. ()[]"
        cleaned = "".join(c for c in str(name) if c.isalnum() or c in keep)
        return cleaned.strip() or "output"

    def _unique_path(self, path):
        """A path nothing has been written to *and* nothing else was promised.

        Checking ``os.path.exists`` alone is not enough. A save routine often
        does not write to the exact path it was handed: it appends its own
        extension, or treats the name as a stem and writes ``<stem>_nodes.tif``,
        ``<stem>_edges.tif``, ... Nothing then exists at the reserved path, so
        the next caller in the same save is handed *the same* name and its
        files land on top of the previous component's -- which is how a saved
        session ends up missing some of its sub-properties. Remembering every
        path we hand out closes that hole.
        """
        reserved = getattr(self, "_reserved_paths", None)
        if reserved is None:
            reserved = self._reserved_paths = set()

        def taken(p):
            return os.path.exists(p) or p in reserved

        if not taken(path):
            reserved.add(path)
            return path
        root, ext = os.path.splitext(path)
        n = 2
        while taken(f"{root}_{n}{ext}"):
            n += 1
        out = f"{root}_{n}{ext}"
        reserved.add(out)
        return out

    # -- playback: dialog tracking + Qt static patches -------------------

    def _maybe_track_playback_dialog(self, obj, opened):
        if not self._is_recordable_dialog(obj):
            return
        if opened:
            self._pb_dialog = obj
        else:
            if obj is self._pb_dialog:
                self._pb_dialog = None

    def _build_msg_fifo(self, events):
        # Two lookups: by message identity (title/text), which survives a
        # notice appearing in some sessions but not others, and by method as
        # a fallback for macros recorded before keys existed.
        self._msg_fifo = {}
        self._msg_by_key = {}
        for ev in events:
            if ev.get("op") == "msgbox":
                self._msg_fifo.setdefault(ev.get("method"), []).append(
                    ev.get("button"))
                key = ev.get("key")
                if key:
                    self._msg_by_key.setdefault(key, []).append(
                        ev.get("button"))

    def _msg_answer(self, method, key):
        """Recorded answer for this box, or None to use a safe default."""
        try:
            if key and self._msg_by_key.get(key):
                return self._msg_by_key[key].pop(0)
            fifo = self._msg_fifo.get(method)
            if fifo:
                return fifo.pop(0)
        except Exception:
            pass
        return None

    def _build_filesave_fifo(self, events):
        # Only the markers that do *not* belong to a gesture+save pair go in:
        # the paired ones are named directly from their save event, so leaving
        # them here would push every later name one slot out of step.
        claimed = self._pair_saves(events)["claimed"]
        self._filesave_fifo = [
            {"stem": ev.get("stem"), "ext": ev.get("ext")}
            for i, ev in enumerate(events)
            if ev.get("op") == "filesave" and i not in claimed]

    @staticmethod
    def _ext_from_filter(args, kwargs):
        """Best-effort extension from a save dialog's filter string."""
        try:
            filt = kwargs.get("filter")
            if filt is None and len(args) > 3:
                # getSaveFileName(parent, caption, directory, filter, ...):
                # the filter is the *fourth* argument, not the fifth. Reading
                # args[4] meant a four-argument call -- which is the common
                # case -- never found its filter and always fell through to
                # the '.dat' default below.
                filt = args[3]
            if filt:
                import re as _re
                m = _re.search(r"\*(\.[A-Za-z0-9]+)", str(filt))
                if m:
                    return m.group(1).lower()
        except Exception:
            pass
        return ".dat"

    def _build_open_fifo(self, events):
        # Recorded file-open selections, in order, per method. The FIFO is
        # rebuilt for every session so the same file is reloaded each time.
        self._open_fifo = {}
        for ev in events:
            if ev.get("op") == "open":
                m = ev.get("method", "getOpenFileName")
                if m == "getOpenFileNames":
                    self._open_fifo.setdefault(m, []).append(
                        list(ev.get("paths") or []))
                else:
                    self._open_fifo.setdefault(m, []).append(ev.get("path"))

    def _reset_playback_fifos(self):
        # Consumable answers must be refreshed at the start of each session,
        # otherwise later sessions run with empty queues (defaults only).
        self._build_msg_fifo(self.current_events)
        self._build_open_fifo(self.current_events)
        self._build_filesave_fifo(self.current_events)

    @staticmethod
    def _dir_stamp(directory):
        """Name -> (size, mtime) for everything directly in a folder."""
        out = {}
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    try:
                        st = entry.stat()
                        out[entry.name] = (st.st_size, st.st_mtime_ns)
                    except OSError:
                        pass
        except OSError:
            pass
        return out

    @staticmethod
    def _suspect_files(directory, after, before):
        """Files a component just wrote that look wrong.

        Empty files, and .json files that do not parse or that hold nothing but
        ``null``. Both are reachable without any exception escaping: save_json
        writes incrementally, so a mid-dump failure leaves a truncated file that
        its bare ``except`` then retries over, and passing it a property that is
        None writes a literal ``null`` quite successfully.
        """
        import json as _json
        bad = []
        for fname, stat in after.items():
            if before.get(fname) == stat:
                continue
            path = os.path.join(directory, fname)
            if stat[0] == 0:
                bad.append(f"{fname}: empty")
                continue
            if fname.lower().endswith(".json"):
                try:
                    with open(path) as fh:
                        data = _json.load(fh)
                except Exception as e:
                    bad.append(f"{fname}: unreadable JSON ({e})")
                    continue
                if data is None:
                    bad.append(f"{fname}: contains null")
        return bad

    def _network_class(self):
        """The Network_3D class, however this build exposes it."""
        try:
            net = self.api.get_network()
            if net is not None and hasattr(type(net), "dump"):
                return type(net)
        except Exception:
            pass
        for mod in list(sys.modules.values()):
            cls = getattr(mod, "Network_3D", None)
            if isinstance(cls, type) and hasattr(cls, "dump"):
                return cls
        return None

    def _install_dump_patch(self):
        """Save a session one component at a time instead of all-or-nothing.

        The stock ``dump`` puts every component save in one ``try`` and, on any
        failure, re-runs the whole list with no directory -- so one bad property
        both truncates the session and scatters the retry. Here each component
        is saved on its own, into the folder that was asked for, and a failure
        is reported rather than silently swallowed.
        """
        if not ROBUST_SESSION_DUMP:
            return
        cls = self._network_class()
        if cls is None:
            self._log("  [note] Network_3D.dump not found; using stock save")
            return
        rec = self
        original = cls.dump

        def robust_dump(net_self, directory=None, parent_dir=None, name=None):
            try:
                target = directory
                if not target:
                    target = os.path.join(parent_dir or os.getcwd(),
                                          name or "my_network")
                os.makedirs(target, exist_ok=True)
            except Exception as e:
                rec._log(f"  [error] could not prepare session folder: {e}")
                return original(net_self, directory=directory,
                                parent_dir=parent_dir, name=name)

            saved, failed, silent, quiet = 0, [], [], []
            for meth, kwargs in _DUMP_COMPONENTS:
                fn = getattr(net_self, meth, None)
                if fn is None:
                    continue
                stamp = rec._dir_stamp(target)
                try:
                    fn(target, **kwargs)
                    saved += 1
                except TypeError:
                    # Older signature without the compression keyword.
                    try:
                        fn(target)
                        saved += 1
                    except Exception as e:
                        failed.append((meth, e))
                        continue
                except Exception as e:
                    failed.append((meth, e))
                    continue
                # Returning normally is not proof of success. save_json wraps
                # its write in a bare except and save_singval_iden_dict catches
                # everything and only prints, so a component can fail in total
                # silence. Compare the folder before and after instead.
                if VERIFY_SESSION_COMPONENTS:
                    after = rec._dir_stamp(target)
                    if after == stamp:
                        # Not necessarily an error: a saver whose property is
                        # unset may legitimately write nothing.
                        quiet.append(meth)
                    else:
                        for note in rec._suspect_files(target, after, stamp):
                            silent.append(f"{meth} -> {note}")
            for meth, e in failed:
                rec._log(f"  [warn] {meth} raised: {type(e).__name__}: {e}")
            for note in silent:
                rec._log(f"  [warn] {note}")
            if quiet:
                rec._log(f"  [note] wrote no file: {', '.join(quiet)}")
            rec._log(f"  session components: {saved} saved, "
                     f"{len(failed)} raised, {len(silent)} suspect")
            return target

        cls.dump = robust_dump
        self._orig_dump = (cls, original)

    def _remove_dump_patch(self):
        if not self._orig_dump:
            return
        cls, original = self._orig_dump
        try:
            cls.dump = original
        except Exception:
            pass
        self._orig_dump = None

    def _install_playback_patches(self):
        rec = self

        self._install_dump_patch()

        # Redirect saves: getSaveFileName / getExistingDirectory / getOpenFileName
        self._orig_dialogs["getSaveFileName"] = QFileDialog.getSaveFileName
        self._orig_dialogs["getExistingDirectory"] = \
            QFileDialog.getExistingDirectory
        self._orig_dialogs["getOpenFileName"] = QFileDialog.getOpenFileName
        self._orig_dialogs["getOpenFileNames"] = QFileDialog.getOpenFileNames

        def fake_save(*a, **k):
            # A session save asks once for a destination and then writes its
            # components; hand back the session folder every time it asks, and
            # never let it fall through to the generic naming below.
            if rec._session_save_fresh():
                return (rec._session_save_target, "")
            # A path reserved for this very step wins. It is consumed on use,
            # so a second, unexpected save cannot silently land on the same
            # name -- it falls through and gets named from the queue instead.
            target = rec._pending_save_path
            if target:
                rec._pending_save_path = None
                return (target, "")
            # A name a gesture reserved but had not used by the time it
            # returned. The save it belongs to is arriving now, late, so use
            # it rather than inventing a generic one.
            target = rec._deferred_save_path
            if target:
                fresh = (rec._pb_index() - rec._deferred_save_step
                         <= DEFERRED_SAVE_WINDOW)
                rec._deferred_save_path = None
                if fresh:
                    rec._log(f"  save (late) -> {os.path.basename(target)}")
                    return (target, "")
                rec._log(f"  [skip] stale reserved name "
                         f"{os.path.basename(target)} released")
            # Unrecognised save (e.g. a plugin's own export): build a path in
            # this session's output folder from what was recorded. Only these
            # unnamed saves draw on the queue, which is why the queue holds
            # exactly the recorded saves we could not name (see
            # _build_filesave_fifo). This is a real save, so the folder gets
            # created now.
            entry = rec._filesave_fifo.pop(0) if rec._filesave_fifo else None
            outdir = rec._ensure_output_dir()
            if not outdir:
                return ("", "")
            # Fall back to the stem of the save this step is part of before
            # resorting to a generic name, so an extra component written by a
            # known save is still recognisable.
            stem = ((entry or {}).get("stem") or rec._save_stem_hint
                    or "plugin_output")
            ext = (entry or {}).get("ext") or rec._ext_from_filter(a, k)
            base = rec._with_session(rec._safe_name(stem))
            target = rec._unique_path(os.path.join(outdir, base + ext))
            rec._log(f"  save -> {os.path.basename(target)}")
            return (target, "")

        def fake_existing_dir(*a, **k):
            # A session save that asks for a folder gets the same destination
            # a session save asking for a filename would get.
            if rec._session_save_fresh():
                try:
                    os.makedirs(rec._session_save_target, exist_ok=True)
                except Exception:
                    pass
                return rec._session_save_target
            # The caller is about to write into whatever we hand back, so the
            # folder has to exist by the time it does.
            return rec._ensure_output_dir() or ""

        def fake_open(*a, **k):
            # Reload the same file the user picked while recording. If it no
            # longer exists, return "" so the caller's `if filename:` guard
            # skips the load and the session continues.
            fifo = rec._open_fifo.get("getOpenFileName")
            if fifo:
                path = fifo.pop(0)
                if path and os.path.exists(path):
                    rec._log(f"  open -> {os.path.basename(path)}")
                    return (path, "")
                rec._log(f"  [skip] open: file not found ({path})")
                return ("", "")
            rec._log("  [note] unrecorded file-open suppressed in batch")
            return ("", "")

        def fake_open_multi(*a, **k):
            fifo = rec._open_fifo.get("getOpenFileNames")
            if fifo:
                paths = fifo.pop(0) or []
                existing = [p for p in paths if p and os.path.exists(p)]
                if existing:
                    rec._log(f"  open -> {len(existing)} file(s)")
                    return (existing, "")
                rec._log("  [skip] open: file(s) not found")
                return ([], "")
            return ([], "")

        QFileDialog.getSaveFileName = staticmethod(fake_save).__func__
        QFileDialog.getExistingDirectory = \
            staticmethod(fake_existing_dir).__func__
        QFileDialog.getOpenFileName = staticmethod(fake_open).__func__
        QFileDialog.getOpenFileNames = staticmethod(fake_open_multi).__func__

        # Context menus: never display during playback. If a recorded action
        # path is pending, resolve it inside the freshly-built menu and fire
        # it; otherwise the menu is simply suppressed so nothing blocks.
        self._orig_dialogs["QMenu.exec"] = QMenu.exec

        def fake_menu_exec(menu_self, *a, **k):
            target = rec._pb_ctx_target
            if not target:
                return None
            act = rec._action_path_lookup(menu_self, target)
            if act is None:
                return None
            rec._pb_ctx_hit = True
            act.trigger()
            return act

        _patch_qt_method(QMenu, "exec", fake_menu_exec)

        # Auto-answer message boxes so nothing ever blocks the run. Notices
        # ("Please select spreadsheet...", error alerts) are dismissed with
        # their default button; real questions use the recorded answer.
        for name in ("question", "information", "warning", "critical"):
            self._orig_dialogs[("mbox", name)] = getattr(QMessageBox, name)

            def make(mname):
                is_question = (mname == "question")
                order = (_QUESTION_BUTTON_ORDER if is_question
                         else _NOTICE_BUTTON_ORDER)
                default = (QMessageBox.StandardButton.Yes if is_question
                           else QMessageBox.StandardButton.Ok)

                def fake(*a, **k):
                    # Consume any queued answer so the FIFO stays aligned.
                    recorded = rec._msg_answer(mname,
                                               rec._msg_key_from_args(a))
                    offered = rec._buttons_from_static_args(a, k)

                    btn = None
                    if not AUTO_ANSWER_YES and is_question:
                        cand = rec._as_standard_button(recorded)
                        if cand is not None and (offered is None
                                                 or (offered & cand)):
                            btn = cand
                    if btn is None:
                        btn = rec._pick_flag_button(offered, order)
                    if btn is None:
                        # The call relied on Qt's default button set, which
                        # always contains our fallback (Yes / Ok).
                        btn = default

                    try:
                        title = _clean(a[1]) if len(a) > 1 else ""
                    except Exception:
                        title = ""
                    rec._log(f"  [auto] {mname} -> {rec._button_name(btn)}"
                             + (f": {title}" if title else ""))
                    return btn
                return staticmethod(fake).__func__

            setattr(QMessageBox, name, make(name))

        # Instance-style boxes (msg = QMessageBox(); msg.exec()). These are
        # not covered by the statics above and would otherwise block forever.
        self._orig_dialogs[("mboxi", "exec")] = QMessageBox.exec
        self._orig_dialogs[("mboxi", "open")] = QMessageBox.open

        def fake_box_exec(box_self, *a, **k):
            """Answer an inline ``QMessageBox()`` without ever showing it.

            ``msg.exec()`` starts a nested event loop and sits there until a
            human clicks. In a batch run there is no human, so the answer is
            decided here and the box is closed with ``done()``, which sets the
            value ``exec()`` would have returned.

            With AUTO_ANSWER_YES (the default) a confirmation is answered
            Yes, so ``return msg.exec() == QMessageBox.StandardButton.Yes``
            proceeds. This is deliberate rather than recorded: these boxes are
            anonymous (no class, no addressable buttons) and their informative
            text is frequently session-specific, so keying a recorded answer
            to them is unreliable -- and answering No would skip the very
            branch the macro was recorded to exercise.
            """
            # Consume one queued answer regardless of whether we use it, so
            # the recorded FIFO stays aligned with the rest of the macro.
            recorded = rec._msg_answer("exec", rec._msg_key_from_box(box_self))

            try:
                label = (_clean(box_self.text())
                         or _clean(box_self.informativeText())
                         or _clean(box_self.windowTitle()))[:60]
            except Exception:
                label = ""

            btn = None
            if not AUTO_ANSWER_YES:
                cand = rec._as_standard_button(recorded)
                # Only trust a recorded answer if this box really offers it.
                if cand is not None and rec._box_offers(box_self, cand):
                    btn = cand

            if btn is None:
                btn = rec._pick_standard_button(box_self,
                                                _QUESTION_BUTTON_ORDER)

            if btn is None:
                # No standard buttons: the box was built with addButton().
                custom = rec._custom_accept_button(box_self)
                if custom is not None:
                    rec._log("  [auto] message box -> "
                             + (_clean(custom.text()) or "(custom button)")
                             + (f": {label}" if label else ""))
                    try:
                        # click() sets clickedButton() and calls done() with
                        # the code exec() would have returned.
                        custom.click()
                    except Exception:
                        pass
                    try:
                        if box_self.isVisible():
                            box_self.done(0)
                    except Exception:
                        pass
                    try:
                        return int(box_self.result())
                    except Exception:
                        return 0
                btn = rec._default_button_for(box_self)

            rec._log(f"  [auto] message box -> {rec._button_name(btn)}"
                     + (f": {label}" if label else ""))

            # Prefer clicking the real button so clickedButton() is correct
            # for callers that check it; fall back to done() otherwise.
            handled = False
            try:
                target = box_self.button(btn)
            except Exception:
                target = None
            if target is not None:
                try:
                    target.click()
                    handled = True
                except Exception:
                    handled = False
            if not handled:
                try:
                    box_self.done(int(btn))
                except Exception:
                    pass
            return int(btn)

        def fake_box_open(box_self, *a, **k):
            fake_box_exec(box_self)
            return None

        _patch_qt_method(QMessageBox, "exec", fake_box_exec)
        _patch_qt_method(QMessageBox, "open", fake_box_open)

        # A nested QApplication.exec() (used by some tool windows, e.g. the
        # histogram selector) would never return while the main loop owns the
        # app -- suppress it; we are already inside a running event loop.
        self._orig_dialogs["QApplication.exec"] = QApplication.exec

        def fake_app_exec(*a, **k):
            rec._log("  [note] nested QApplication.exec() suppressed")
            return 0

        QApplication.exec = fake_app_exec

        # matplotlib's blocking show() would stall the batch.
        plt = sys.modules.get("matplotlib.pyplot")
        if plt is not None and hasattr(plt, "show"):
            self._orig_dialogs[("plt", "show")] = plt.show

            def fake_plt_show(*a, **k):
                return None

            plt.show = fake_plt_show

    def _remove_playback_patches(self):
        for key, original in list(self._orig_dialogs.items()):
            try:
                if isinstance(key, tuple) and key[0] == "mboxi":
                    _restore_qt_method(QMessageBox, key[1], original)
                elif isinstance(key, tuple) and key[0] == "mbox":
                    setattr(QMessageBox, key[1], original)   # statics
                elif isinstance(key, tuple) and key[0] == "plt":
                    plt = sys.modules.get("matplotlib.pyplot")
                    if plt is not None:
                        setattr(plt, key[1], original)
                elif key == "QMenu.exec":
                    _restore_qt_method(QMenu, "exec", original)
                elif key == "QApplication.exec":
                    QApplication.exec = original
                else:
                    setattr(QFileDialog, key, original)
            except Exception:
                pass
        self._orig_dialogs.clear()
        self._remove_dump_patch()
        _verify_qt_methods()

    def _close_stray_dialogs(self):
        """Close any lingering analysis dialogs between sessions."""
        try:
            for w in QApplication.topLevelWidgets():
                if not w.isVisible():
                    continue
                # Message boxes never appear during playback (they are
                # auto-answered), but close any that slipped through.
                if isinstance(w, QMessageBox):
                    try:
                        w.close()
                    except Exception:
                        pass
                    continue
                if not self._is_recordable_dialog(w):
                    continue
                try:
                    if isinstance(w, QDialog):
                        w.reject()
                    else:
                        w.close()
                except Exception:
                    try:
                        w.close()
                    except Exception:
                        pass
        except Exception:
            pass
        self._pb_dialog = None

    def _rss_mb(self):
        """Resident set size in MB, or None where it cannot be read."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            pass
        try:
            with open("/proc/self/statm") as fh:
                pages = int(fh.read().split()[1])
            import resource
            return pages * resource.getpagesize() / (1024 * 1024)
        except Exception:
            return None

    def _live_dialogs(self):
        """Every QDialog still owned by the main window, visible or not."""
        try:
            return self.win.findChildren(QDialog)
        except Exception:
            return []

    def _reap_between_sessions(self):
        """Delete what the previous round left behind.

        ``_close_stray_dialogs`` only closes *visible* dialogs, and closing a
        dialog parented to the main window merely hides it -- the C++ object
        stays in the child list, holding whatever the dialog captured. Over a
        long batch that is one live copy per dialog per session. Deleting them
        here is the difference between hiding and freeing.
        """
        freed = 0
        if REAP_DIALOGS_BETWEEN_SESSIONS:
            for dlg in self._live_dialogs():
                try:
                    if dlg.isVisible():
                        continue
                    dlg.setParent(None)
                    dlg.deleteLater()
                    freed += 1
                except Exception:
                    pass
        figs = 0
        if CLOSE_FIGURES_BETWEEN_SESSIONS:
            plt = sys.modules.get("matplotlib.pyplot")
            if plt is not None:
                try:
                    figs = len(plt.get_fignums())
                    plt.close("all")
                except Exception:
                    figs = 0
        # Let deleteLater actually run before the collector looks.
        try:
            QApplication.processEvents()
            QApplication.sendPostedEvents(None, 0)
        except Exception:
            pass
        collected = gc.collect()
        return freed, figs, collected

    def _log_session_memory(self, freed, figs, collected):
        if not LOG_MEMORY_PER_SESSION:
            return
        parts = []
        if freed or figs:
            parts.append(f"reaped {freed} dialog(s), {figs} figure(s)")
        if collected:
            parts.append(f"gc freed {collected} object(s)")
        live = len(self._live_dialogs())
        parts.append(f"{live} dialog(s) still live")
        rss = self._rss_mb()
        if rss is not None:
            delta = ("" if self._last_rss is None
                     else f", {rss - self._last_rss:+.0f} MB")
            self._last_rss = rss
            parts.append(f"RSS {rss:.0f} MB{delta}")
        self._log("  " + "; ".join(parts))

    # -- lifecycle -------------------------------------------------------

    def _shutdown(self, reason="closed"):
        """Undo every global change this plugin made. Safe to call twice.

        This must run on *any* path that takes the window away, otherwise the
        monkeypatches (QMenu.exec, the QFileDialog/QMessageBox statics, the
        wrapped save methods) stay installed pointing at a dead recorder and
        the whole GUI misbehaves until NetTracer3D is restarted.
        """
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        try:
            # 1. Stop any in-flight batch/playback before unpatching, so no
            #    queued timer step runs against a half-restored app.
            self._aborted = True
            self._batch = None
            self._pb = None
            self._pb_dialog = None
            self._pb_ctx_target = None
            self._pb_ctx_hit = False
            self._current_output_dir = None
            self._made_dirs = set()
            self._pending_save_path = None
            self._deferred_save_path = None
            self._deferred_save_step = -1
            self._save_stem_hint = None
            self._session_save_target = None
            self._session_save_step = -1
            self._reserved_paths = set()
            self._notag_paths = set()
            self._session_dirs = set()
            self._ctx_pending = None

            # 2. Recording hooks.
            if self.recording:
                try:
                    self.win.menuBar().triggered.disconnect(
                        self._on_menu_triggered)
                except Exception:
                    pass
            self.recording = False
            self._unwrap_saves()

            # 3. Playback hooks.
            self._remove_playback_patches()
            self._playing = False

            # 4. The app-wide event filter is used by both paths.
            try:
                app = QApplication.instance()
                if app is not None:
                    app.removeEventFilter(self)
            except Exception:
                pass

            # 5. Restore the UI in case the window is shown again.
            try:
                self.btn_run.setEnabled(True)
                self.btn_start.setEnabled(True)
                self.btn_load.setEnabled(True)
                self.progress.setVisible(False)
            except Exception:
                pass
        except Exception:
            pass
        finally:
            self._shutting_down = False

    def closeEvent(self, event):
        self._shutdown("closed")
        super().closeEvent(event)

    def done(self, result):
        # QDialog.done() is what Escape/reject() and accept() go through --
        # neither of which sends a closeEvent, so clean up here too.
        self._shutdown("dismissed")
        super().done(result)

    def reject(self):
        self._shutdown("dismissed")
        super().reject()


# ===========================================================================
# Plugin entry points
# ===========================================================================

_dialog_singleton = {"dlg": None}


def _open_dialog(api):
    dlg = _dialog_singleton.get("dlg")
    if dlg is not None:
        # If Qt destroyed the underlying C++ object (e.g. the parent window
        # was rebuilt), any access raises RuntimeError -- rebuild instead.
        try:
            dlg.isVisible()
        except RuntimeError:
            dlg = None
            _dialog_singleton["dlg"] = None
    if dlg is None:
        try:
            parent = api.get_unsafe_window()
        except Exception:
            parent = None
        dlg = MacroRecorderDialog(api, parent)
        _dialog_singleton["dlg"] = dlg
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()


def register(api):
    api.register_menu_action(
        "Extensions/Macro Recorder/Open Macro Recorder\u2026",
        lambda *args: _open_dialog(api),
        tooltip="Record GUI actions and replay them across many sessions.")


def unregister(api):
    dlg = _dialog_singleton.get("dlg")
    if dlg is not None:
        try:
            if dlg.recording:
                dlg.stop_recording()
        except Exception:
            pass
        try:
            dlg.close()
        except Exception:
            pass
        _dialog_singleton["dlg"] = None
