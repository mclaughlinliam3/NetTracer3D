.. _keyboard_shortcuts:

============
Available Keyboard Shortcuts
============


In the default viewer mode:
----------------------------

* 'Mousewheel' = zoom in/out.
* 'Z' = toggle zoom mode.
    * Clicking will zoom in.
    * Right clicking will zoom out.
    * Shift + Right Click will zoom all the way out.
    * Left clicking + Dragging will in zoom mode will zoom in on specific regions.
* 'X' = toggle highlight overlay visibility.
* 'Middle Mouse' = Quick Pan Mode.
* 'Shift + Scroll Wheel' = Move Through Image Stack.
* 'Ctrl + Shift + Scroll Wheel' = Rapidly move through image stack.
* 'Shift + F' = Search for a node or an edge in the image channels.
* 'Ctrl + Left Click in the image viewer window' = Select/deselect object without deselecting others that are selected.
* 'Left Click + Drag in the image viewer window' = Group select objects. (Add ctrl to not deselect current selection)
    * And hold shift while dragging to instead crop a specific region.
* 'Delete' - Deletes any selected nodes/edges from the image and from their respective networks.
* 'Ctrl + S' = If no Network3D object has been saved this session, the user will be prompted to create a new folder to save one. If one has been created, then the current state of the data will be saved to the last-saved location of the Network3D object. (Avoids having to go through the save menu).
* 'Ctrl + L' = If no Network3D object has been loaded this session, the user will be prompted to find a folder to load one from. If one has been loaded, then the same directory will be used to load a Network3D Object. (In case the user has messed around with the current data and wants to reload it from baseline easily).

If in the upper right or lower right network/tabulated data widgets:
----------------------------------------------------

* Ctrl + F = Find specific table entry.

If in paint mode:
----------------------------

* Ctrl + Scroll Wheel = Resize the Brush.
* Click = Draw, Right Click = Erase
* D = Toggle 3D Mode for both the paintbrush and the fill-can, which makes operations occur on multiple planes at once.
    * While using the 3D brush, 'Alt + Scroll wheel' = Resize the amount of planes the 3D brush applies to.
* F = Toggle fill can mode
    * While using the fill-can only, 'Ctrl + z' will undo the last fill-can action. Note this is the only function in the entire program to feature an undo option.
* 'Ctrl + Z' = Undo Last Paint Action (Does not go back if you've refreshed the display).

If using the machine-learning segmenter:
----------------------------

* A = Toggle whether segmenting foreground/background.
* T = train model with current settings
* 'Ctrl + Z' = Undo Last Paint Action (Does not go back if you've refreshed the display).
* The 'brush mode' also uses the same shortcuts as paint mode, besides the fill can/3d drawing.


Next Steps
---------
This concludes the tutorial section about using NetTracer3D. Although I covered many network-generating options in some detail, there are a plethora of other features and functions to learn about. The rest of this guide will go over all the algorithms and associated parameters within NetTracer3D in detail, in a more informative and less tutorial-oriented style. For questions about any particular function, please locate the section in the corresponding section guide to read more about it.