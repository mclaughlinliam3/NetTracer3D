.. _extensions:

=========================
NetTracer3D Plugin System and Instructions on Using Prebundled Plugins
=========================

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

See :ref:`cellpose-extension` for a full walkthrough of the panel,
including chunk padding and seam handling.


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


.. _bundled-extensions:

--------------------
Bundled Extensions
--------------------

This section documents the extensions that ship with NetTracer3D in
detail — what each control does and how to get sensible results out of
it.  For the one-line summary of what is bundled, see
`Built-In Plugins`_ in the User Guide above.

Extensions documented here are installed automatically with
NetTracer3D, but they are ordinary plugins: they appear in the
Extensions panel like any other, can be disabled, and may need their
dependencies installed on first use.


.. _cellpose-extension:

Cellpose Segmentation
=====================

Runs `Cellpose <https://cellpose.readthedocs.io/>`_ instance
segmentation on a NetTracer3D channel and writes the resulting label
mask back into a channel of your choice.

**Requires**: ``cellpose>=3.0``.  On first use the extension will be
marked **Needs Deps** — select it in the Extensions panel and click
**Install Deps**.  Accept the CUDA prompt if you want GPU support.

Opening the panel
-----------------

  **Extensions → Cellpose → Open Cellpose Panel...**

There is also **Extensions → Cellpose → Quick Launch Cellpose GUI**,
which launches the standalone Cellpose Qt application in a separate
process.  That GUI is independent of NetTracer3D — it does not see your
loaded channels and does not write back to them.  Use it for exploring
models interactively; use the panel for everything else.

Choosing channels
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Control
     - Meaning
   * - Image to segment
     - The channel Cellpose runs on.  Must contain data.
   * - Secondary context image
     - Optional second channel stacked as an extra input channel — for
       example a nuclear stain alongside a cytoplasmic one.  Must have
       the same Z/Y/X dimensions as the primary image.
   * - Send output to
     - Where the label mask is written.  **This overwrites whatever is
       in that channel**, so point it at an empty channel unless you
       mean to replace something.

The output dtype is chosen automatically from the number of objects
found (``uint8``, ``uint16``, or ``uint32``).

Model selection
---------------

The dropdown lists the built-in Cellpose models (``cyto3``, ``nuclei``,
``tissuenet_cp3`` and so on).  **Load Custom Model...** adds a trained
``.pth`` file to the list, marked with a ✦.  Custom models are
remembered for the lifetime of the panel.

Segmentation parameters
-----------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Meaning
   * - Diameter
     - Expected object diameter in pixels.  ``0`` lets Cellpose
       estimate it.  This is the single most important parameter — if
       results are poor, fix this before touching anything else.
   * - Flow threshold
     - Maximum allowed error of the flow field.  Raise it to accept
       more objects, lower it for stricter segmentation.
   * - Cell probability threshold
     - Pixels above this probability are considered part of an object.
       Lower it to pick up dimmer objects.
   * - Minimum object size
     - Objects smaller than this many pixels are discarded by Cellpose.
   * - Stitch threshold
     - IoU threshold for stitching 2-D masks across Z.  Only used for
       3-D images when **Native 3D** is off.  ``0`` disables stitching
       and leaves every slice independent.
   * - Use GPU
     - Uses CUDA if available and falls back to CPU otherwise.
   * - Native 3D
     - Uses true volumetric convolutions instead of the default
       slice-by-slice-then-stitch approach.  Much slower and far more
       memory-hungry, but can do better on roughly spherical objects.

Dimensionality is detected automatically.  NetTracer3D stores images as
``(Z, Y, X)`` even when there is only one slice, so a single-slice
channel is treated as 2-D and the 3-D-only options are ignored.

Chunked processing
------------------

Large volumes often will not fit in GPU memory.  **Enable** under
Chunked Processing splits the image into sub-volumes, segments each
one, and reassembles the result.  The number of chunks is a target —
the plugin picks a Z×Y×X grid whose chunks are as close to cubic as it
can manage, so the actual count may be slightly higher (asking for 8
on a flat stack typically gives a 1×3×3 grid, i.e. 9 chunks).

Padding
~~~~~~~

Chunking naively means objects sitting on a chunk boundary get cut in
half by an artificial image edge, and Cellpose's flow field near that
edge is wrong.  **Use padding** (on by default) gives each chunk an
overlapping border of surrounding voxels when it is sent to Cellpose,
then discards that border when the volume is put back together.  Each
chunk therefore contributes only its central region, but that region
was segmented with its real surroundings visible.

**XY padding** and **Z padding** are set separately because data is
often anisotropic — with thick slices you usually want much less
padding in Z.  Both are auto-filled from the image size and the chunk
count, and re-estimated whenever you change either.  Typing your own
value switches that field to manual; **Auto** hands it back to the
estimator.

A good manual value is roughly one object diameter.  Note that padding
enlarges the block actually handed to Cellpose, so it raises peak VRAM
per chunk — if you run out of memory, reduce the padding or increase
the chunk count.

Seam handling
~~~~~~~~~~~~~

**Keep objects crossing chunk seams whole** (on by default) lets the
first chunk that owns an object's centre also claim the parts spilling
past its boundary, so an object straddling a seam stays one label
instead of being split into two.  Turn it off for a strict crop, where
each chunk contributes only what falls inside its own core region.

**Filter seam artifacts** (on by default) removes labels far smaller
than a typical object after reassembly.  These are usually thin slivers
along a seam, left where two chunks segmented the same object and
disagreed about its boundary by a voxel or two.  The threshold is a
percentage of the *median* object volume — the median rather than the
mean, because fragments are numerous and tiny enough to drag a mean
down toward themselves, and because object sizes are right-skewed so
plenty of legitimate objects sit below the mean.  10% is conservative;
raise it if fragments survive, lower it if real objects vanish.

.. note::

   If you disable the artifact filter, expect to clean up small
   fragments along the chunk borders yourself downstream.

Reading the console
-------------------

The extension logs to the console with a ``[Cellpose]`` prefix, which
is the quickest way to see what it actually did::

   [Cellpose] Detected 3D image: (48, 512, 512) (Z, Y, X)
   [Cellpose] Mode: slice-by-slice + stitch (stitch_threshold=0.5)
   [Cellpose] Loading model: cyto3, GPU=True
   [Cellpose] Chunked into 8 pieces (2x2x2), padding XY=26, Z=26
   [Cellpose] Segmenting chunk 1/8, shape=(50, 282, 282)...
   [Cellpose] Removed 3 fragment(s) smaller than 38 voxels
   [Cellpose] Segmentation complete: 214 objects

Worth checking on the first run of a new dataset: that the
dimensionality is what you expected, and that the artifact filter is
not removing more than a handful of objects.

Troubleshooting
---------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Symptom
     - Try
   * - Out of GPU memory
     - Enable chunking, raise the chunk count, or reduce the padding.
       Turn off **Native 3D**, which is by far the most memory-hungry
       mode.
   * - Chunking seems to do nothing
     - Check the console for the chunk grid line.  A chunk count of 1
       is ignored.
   * - Objects split along chunk borders
     - Increase the padding and confirm seam recovery is enabled.
   * - Small fragments along chunk borders
     - Enable the artifact filter, or raise its percentage.
   * - Real objects disappearing
     - Lower the artifact filter percentage or disable it, and check
       **Minimum object size**.
   * - Too few / too many objects
     - Fix **Diameter** first, then adjust flow and cell probability
       thresholds.
   * - 3-D objects fragmented across Z
     - Raise **Stitch threshold** toward 1.0 for stricter matching, or
       lower it to merge more readily.  Consider **Native 3D**.

.. _macro-recorder-extension:

Macro Recorder
==============

Records what you do in the GUI and replays it across many saved
sessions, so a workflow you have worked out on one dataset can be run
unattended over a whole folder of them.

**Requires**: nothing beyond NetTracer3D itself.

Opening the panel
-----------------

  **Extensions → Macro Recorder → Open Macro Recorder...**

The panel has four buttons — **Start Recording**, **Stop Recording**,
**Load Macro** and **Run Macro** — plus a progress bar and a log that
reports every step as it happens.

Recommended workflow
--------------------

1. Load one representative session and get it into the state you want
   to start from.
2. Click **Start Recording**.
3. Carry out your analysis exactly as you normally would.
4. Click **Stop Recording**.  You will be prompted to save the macro as
   a ``.py`` file.
5. Click **Run Macro** and choose the *parent* folder that contains
   your session folders.

The macro is replayed on each session folder in turn.  Recording a
macro against a session that is representative of the batch matters —
a dialog field or right-click entry that only exists for some datasets
will simply be skipped on the sessions that lack it.

.. note::

   Only the *first* session needs to be loaded by hand.  **Run Macro**
   loads each session itself, so do not include a session load in the
   recording.

What gets recorded
------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Interaction
     - Notes
   * - Menu bar actions
     - Recorded by menu path and re-triggered on replay.
   * - Dialog fields and buttons
     - Only fields you actually changed are stored, followed by the
       button you clicked.  Works for both modal and modeless dialogs.
   * - Right-click actions
     - On the image display and on the data tables.  Stored by menu
       label path, e.g. ``Show Identity > ID: neuron``.
   * - On-screen controls
     - Channel visibility toggles, scalebar, home/reset, the
       **Active Image** selector, the highlight overlay toggle, and the
       camera (screenshot) button.
   * - File loads
     - A file chosen through a load dialog is re-loaded from the same
       path on every session.
   * - Saves
     - Tables, image channels, Network3D dumps, quickload pickles and
       screenshots.
   * - Message box answers
     - Replayed so the run never stops waiting for a click.

Checkable controls are replayed to the *state* that was recorded rather
than blindly re-clicked.  If you recorded "scalebar on" and a session
happens to load with the scalebar already on, it is left alone — every
session ends in the same state regardless of where it started.

What is not recorded
--------------------

The zoom, pan, 3D, popout and pen buttons are deliberately ignored:
they are view and interaction modes that carry no meaning from one
session to the next.  Left-click selection on the image canvas, and
direct canvas manipulation such as panning, zooming and painting, are
likewise not recorded.

Anything that depends on *where* you clicked is the general limitation
here.  Right-click entries whose meaning follows from the current
selection — **Show Neighbors**, the **Selection** submenu, measurement
points — are recorded, but on replay they act on whatever happens to be
selected at that moment, which may differ per session.  For batch work
prefer the selection-independent entries: identities, communities,
Select All, Sort and Save As.

Where the output goes
---------------------

Each run creates a new timestamped folder inside the parent folder you
selected, containing one sub-folder per session::

   YourParentFolder/
   ├── Mouse_01/                          <- your sessions, untouched
   ├── Mouse_02/
   └── MacroBatch_2026-08-11_01-10-12/
       ├── Mouse_01_Output/
       │   ├── Proximity_Mouse_01.csv
       │   └── screenshot_Mouse_01.png
       └── Mouse_02_Output/
           ├── Proximity_Mouse_02.csv
           └── screenshot_Mouse_02.png

Because the folder is timestamped, running the same macro again never
overwrites an earlier batch.  Session folders themselves are never
written to.

Every output file also carries its session name at the end of the
filename.  This is what makes the results poolable: you can copy the
contents of every ``*_Output`` folder into a single directory for
downstream analysis without anything colliding.

Folders whose name starts with ``MacroBatch_`` are skipped when
scanning for sessions, so previous results are never mistaken for
input.

When something goes wrong
-------------------------

Errors are contained to the session that caused them.  A folder that is
corrupt, is not a NetTracer3D session, or is simply incompatible with
the macro is logged and skipped, and the run carries on with the next
one.  The summary at the end reports how many sessions succeeded, and
the log lists what failed for each one that did not.

Steps that cannot be applied to a particular session — a dialog field
that is not present, a right-click entry that dataset does not produce,
a channel toggle that is disabled, a load file that has been moved — are
logged as skipped and the rest of the macro continues.

Editing a macro by hand
-----------------------

Saved macros are plain, readable Python.  Each recorded step is one
entry in an ``EVENTS`` list with a comment describing it::

   MACRO_FORMAT = 1

   EVENTS = [
       {'op': 'menu', 'path': ['Analyze', 'Proximity Analysis...']},   # menu: Analyze > Proximity Analysis...
       {'op': 'set', 'w_class': 'QDoubleSpinBox', 'w_index': 0, ...},  # set search distance
       {'op': 'ui', 'target': 'scalebar', 'checked': True},            # scalebar on
       {'op': 'save', 'method': 'save_table_as', 'fmt': 'csv', ...},   # save table 'Proximity' (csv)
   ]

You can delete steps, reorder them, or tweak a recorded value in a text
editor rather than re-recording the whole run.  Keep the ``EVENTS``
name and the dictionary structure intact.

Troubleshooting
---------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Symptom
     - Try
   * - Every session reports LOAD FAILED
     - Check you selected the *parent* folder containing the session
       folders, not a session folder itself.
   * - A step is skipped on some sessions
     - The dialog field or menu entry does not exist for that dataset.
       Record against a representative session, or split the macro.
   * - Results differ between sessions
     - Something in the macro depends on the current selection or on a
       clicked position.  Prefer selection-independent actions.
   * - A recorded file load is skipped
     - The file has been moved or renamed.  Macros store the absolute
       path chosen at record time.
   * - Nothing is saved
     - The macro contains no save step.  Saves must be performed while
       recording for the run to produce output.
