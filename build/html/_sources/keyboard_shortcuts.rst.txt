.. _keyboard_shortcuts:

============================
Available Keyboard Shortcuts
============================


Default Viewer Mode
-------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Shortcut
     - Action
   * - Mouse wheel
     - Zoom in / out.
   * - ``Z``
     - Toggle zoom mode. Left click zooms in, right click zooms out, and
       :kbd:`Shift` + right click zooms all the way out. Left click and drag
       zooms in on a specific region.
   * - ``X``
     - Toggle highlight overlay visibility.
   * - Middle mouse
     - Quick pan mode.
   * - ``Shift`` + scroll wheel
     - Move through the image stack.
   * - ``Ctrl`` + ``Shift`` + scroll wheel
     - Move rapidly through the image stack.
   * - ``Shift`` + ``F``
     - Search for a node or edge in the image channels.
   * - ``Ctrl`` + left click
     - Select or deselect an object in the image viewer window without
       deselecting others already selected.
   * - Left click + drag
     - Group select objects in the image viewer window. Add :kbd:`Ctrl` to
       preserve the current selection, or hold :kbd:`Shift` while dragging to
       crop a specific region instead.
   * - ``Delete``
     - Delete any selected nodes or edges from the image and from their
       respective networks.
   * - ``Ctrl`` + ``S``
     - Save the Network3D object. If none has been saved this session, you are
       prompted to create a new folder; otherwise the current state is saved to
       the last-saved location, avoiding the save menu.
   * - ``Ctrl`` + ``L``
     - Load a Network3D object. If none has been loaded this session, you are
       prompted to find a folder; otherwise the same directory is reused, making
       it easy to reload the baseline data after experimenting.
   * - ``B``
     - Enter flood click mode. Set a distance from the mouse, then click on
       segmented nodes or edges images to select the group of voxels within that
       distance. The distance is measured by how far the selection can flood
       through contiguous foreground rather than through space, so
       non-contiguous nearby foreground is not selected. This is useful for
       punching out arbitrary regions of a mask.


Network and Tabulated Data Widgets
----------------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Shortcut
     - Action
   * - ``Ctrl`` + ``F``
     - Find a specific table entry.


Paint Mode
----------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Shortcut
     - Action
   * - ``Ctrl`` + scroll wheel
     - Resize the brush.
   * - Click / right click
     - Draw / erase.
   * - ``D``
     - Toggle 3D mode for both the paintbrush and the fill can, applying
       operations across multiple planes at once. While using the 3D brush,
       :kbd:`Alt` + scroll wheel resizes the number of planes affected.
   * - ``F``
     - Toggle fill can mode. While using the fill can only, :kbd:`Ctrl` +
       :kbd:`Z` undoes the last fill can action — the only undo option in the
       program.
   * - ``Ctrl`` + ``Z``
     - Undo the last paint action. This does not work once the display has been
       refreshed.


Machine-Learning Segmenter
--------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Shortcut
     - Action
   * - ``A``
     - Toggle between segmenting foreground and background.
   * - ``T``
     - Train the model with the current settings.
   * - ``Ctrl`` + ``Z``
     - Undo the last paint action. This does not work once the display has been
       refreshed.

Brush mode uses the same shortcuts as paint mode, excluding the fill can and 3D
drawing.


Next Steps
----------

This concludes the tutorial section. The remaining chapters cover every algorithm
in NetTracer3D and its associated parameters in detail, in a reference rather
than tutorial style. For questions about a particular function, locate it in the
corresponding section guide.
