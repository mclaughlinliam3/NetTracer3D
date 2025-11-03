.. _image_menu:

==========
All Image Menu Options
==========
* The image menu has options for editing the properties of the current session, modifying visualization, generating overlays, and viewing interactable 3D renders.

'Image -> Properties'
--------------------------

* The properties menu displays what properties are currently present, and is also where xy_scale and z_scale for the image should be set. These are its parameters:

1. xy_scale
    * Enter a float value here corresponding to the pixel-to-real-world-distance of your image. This property will not auto-populate when loading images, but instead should be entered here (or in functions that request it).
    * Note that NetTracer3D currently presumes your x and y scales will always be equal, and therefore does not support images that were transformed along only a single 2D axis.
2. z_scale
    * Enter a float value here corresponding to the voxel-to-real-world-depth/step-size of your image. This property will not auto-populate when loading images, but instead should be entered here (or in functions that request it).
3. Nodes status
    * This button will be enabled if something is currently loaded into the nodes channel. Disabling it and entering the new properties will empty the nodes channel. It will also purge the node_identities and node_centroids property.
    * Enabling this button when it was previously disabled will not do anything, as this menu is not for loading.
4. Edges status
    * This button will be enabled if something is currently loaded into the edges channel. Disabling it and entering the new properties will empty the edges channel. It will also purge the edge_centroids property.
    * Enabling this button when it was previously disabled will not do anything, as this menu is not for loading.
5. Overlay1 status
    * This button will be enabled if something is currently loaded into the Overlay1 channel. Disabling it and entering the new properties will empty the Overlay1 channel.
    * Enabling this button when it was previously disabled will not do anything, as this menu is not for loading.
6. Overlay2 status
    * This button will be enabled if something is currently loaded into the Overlay2 channel. Disabling it and entering the new properties will empty the Overlay2 channel.
    * Enabling this button when it was previously disabled will not do anything, as this menu is not for loading.
7. Node Search Region Status
    * This button will be enabled if something is currently loaded into the hidden Search Regions property. Disabling it and entering the new properties will purge this property.
    * This property is generated from the Connectivity Network function and represents the search space of expanded nodes. It will be saved with Network3D objects but currently cannot be loaded directly into this property.
    * However one might want to load it into the nodes property if they desire to run the Connectivity Network again and would like to skip the node search expansion step (which is the slow step).
8. Network Status
    * This button will be enabled if something is currently loaded into the Network property/table. Disabling it and entering the new properties will empty the network. It will also purge the communities property.
9. Identities status
    * Will be checked if the nodes identities property is set.
10. Enter (Erases Unchecked Properties)
    * Accepts reassigned properties (mainly just xy and z scales if altered), but anything that was unchecked will be purged from the current session.
11. Report Properties
    * Any property associated with a spreadsheet that exists in the current session will populate the upperright table.

'Image -> Adjust Brightness/Contrast'
--------------------------
* Selecting this displays a menu where brightness/contrast of images can be altered. 
* For each channel, there will be a dual-knobbed slider bar alongside min/max values.
* The user can move the left knob or enter a val 0-65534 in the left bar to set the maximum brightness. The user can move the right knob or enter a val 1-65535 to set the minimum brightness.
* If you can't see a loaded image, please use this menu to make it visible.

'Image -> Channel Colors'
--------------------------
* The default colors for each channel are light_red for nodes, light_green for edges, and white for Overlay1 and Overlay2.
* Here the user can alter those colors for each channel using a wide selection of preset colors.

'Image -> Overlays -> Create Network Overlay'
--------------------------
* If a network is present with node_centroids (the user will be prompted if no centroids), then this method draws 1-voxel thick white lines between all node centroids.
* This line-based network overlay is placed in Overlay1 and provides a convenient way to visualize network structure, especially in 3D.
* Note that the lines tend to be somewhat thin by default, so for larger images, the user may want to dilate the overlay a small amount to better see it.
* There is a param to optionally downsample while generating this, which essentially just enlarges rendered the output by the magnitude of the entered downsample.

'Image -> Overlays -> Create ID Overlay'
--------------------------
* If a network is present with node_centroids (the user will be prompted if no centroids), then this method will literally write the numerical ID of each node over its centroid.
* This ID overlay is placed in Overlay2 and provides a convenient way to visualize node labels.
* There is a param to optionally downsample while generating this, which essentially just enlarges rendered the output by the magnitude of the entered downsample.

'Image -> Overlays -> Color Nodes (or edges)'
--------------------------
* This method will create a new RGB overlay where each grayscale label in the nodes (or edges) image is assigned a unique color.
* This overlay will be placed in Overlay2, while a legend saying what node/edge corresponds to what color will be placed in the tabulated data widget.
* This is an excellent way to visualize what nodes/edges have been labeled.
* The first parameter is a dropdown menu to tell the program whether to color the nodes or the edges.
* The second parameter, down_factor, applies an internal downsample equivalent to the inputted integer on all three dimensions before drawing the overlay. This can be used to speed up processing, but note that over-downsampling small nodes may cause them to be removed from the image.

'Image -> Overlays -> Shuffle'
--------------------------
* This method can be used to swap the data within the channels. Enter the desired channels to swap in the dropdown menu, and their images will be exchanged. If one of the channels is empty, the one that is not empty will have its data moved to the empty channel.
* This method is actually rather useful when using NetTracer3D, since the system often populates outputs to specific overlay channels (which will overwrite any preexisting data there), and expects contents in certain channels for other functions. For example, labeling branches has to be done in the edges channel, while grouping branches into networks has to be done in the nodes channel. The shuffle function can be used to move each one to the correct channel without having to save and reload images.

'Image -> Select Objects'
--------------------------
* This method can be used to arbitrarily select groups of objects, and find them in the image.

Parameter Explanations
~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Type to Select:
    * Whether we want to select objects in the nodes or edges channels.
2. Select the following?:
    * Enter a list of unspaced integers seperated by commas (ie: '4,5,7,10') to have NetTracer3D select and highlight them.
    * Additionally, NetTracer3D will navigate to the Z-plane of the node that corresponds to the first integer - therefore this window can be used for arbitrary searching for nodes/edges.
    * Select the 'Import Selection from spreadsheet...' button to open the file browser. Select a .csv or .xlsx file where the integers of the desired objects have been placed in the first column to have this param autopopulate the integers from the spreadsheet. This can be used to arbitrarily select groups of objects that were pre-organized by some means.
3. Deselect the following?
    * Enter a list of unspaced integers seperated by commas (ie: '4,5,7,10') to have NetTracer3D deselect them (if they were selected).
    * Select the 'Import Selection from spreadsheet...' button to open the file browser. Select a .csv or .xlsx file where the integers of the desired objects have been placed in the first column to have this param autopopulate the integers from the spreadsheet. This can be used to arbitrarily deselect groups of objects that were pre-organized by some means.
    * Note that param 2 always overrides param 3, so selecting and deselecting the same object will result in it being selected.


'Image -> Show 3D (Napari)'
--------------------------
* At last, 3D visualization!
* Select this option to have NetTracer3D use Napari (the premiere open-source pythonic 3D image viewer) to show a 3D render of all visible images. Chi-Li Chiu, Nathan Clack, the napari community, napari: a Python Multi-Dimensional Image Viewer Platform for the Research Community, Microscopy and Microanalysis, Volume 28, Issue S1, 1 August 2022, Pages 1576–1577, https://doi.org/10.1017/S1431927622006328
* Napari will show any channels that are currently visible in the bottom control panel, so disable the visibility of any channels you do not wish to show. It will also show the highlight overlay if it is present.
* If your computer monitor is currently hooked up to your GPU, Napari will use your GPU for rendering by default. As long as the sum of your images' sizes are less than your total VRAM, Napari visual displays are quite smooth. 
    * However this visualization does not utilize image pyramids in this case. If your images' sizes exceed the VRAM of your card, please downsample it or it will lag. (This is a feature that I may implement in the future).
    * If your monitor is not currently using the GPU, this visualization will be limited to small images.
* This requires Napari to be installed in NetTracer3D's package environment.

Parameter Explanations
~~~~~~~~~~~~~~~~~~~~~~~~~~~
#. Downsample Factor:
    * Temporarily downsamples the image to speed up the 3D display. Downsampling is done in all three dimensions by the inputed factor.
#. Use cubic downsample?:
    * Enable this to use the cubic resample algorithm, which is slower but may better preserve shapes.
#. Include Bounding Box
    * Enable this to draw in a bounding box around your channels in the visualization. Note the bounding box is an equivalently sized array as the other channels so it will demand the necesarry RAM.

* Press 'Show 3D' to create the 3D display with the desired params. A new Napari window will open and show your desired channels. Note that any RGB images will be split into three seperate red, green, and blue channels.

'Image -> Cellpose'
--------------------------
* Selecting this just opens the Cellpose3 GUI (Stringer, C., Pachitariu, M. Cellpose3: one-click image restoration for improved cellular segmentation. Nat Methods 22, 592–599 (2025). https://doi.org/10.1038/s41592-025-02595-5), provided it has been installed in NetTracer3D's package environment.
* Cellpose3 is my favorite open-source tool to segment cells with, so I added this option as a suggestion to use it together with NetTracer3d.
* This requires Cellpose3 to be installed in NetTracer3D's package environment.
* If NetTracer3D has a 3D image or no image is present, the 3D-stack version of cellpose will open. If a 2D image is open in NetTracer3D, the 2D-stack version of cellpose will open.