.. _changelog:

==========
Changelog
==========

Version 0.7.0 Updates:

1. Added new function in 'Analyze -> Stats -> Cluster Analysis'
    * This function allows the user to create a ripley's K or H function to compare the relative clustering of two types of nodes, or of one type of node vs itself.

2. Added new function in 'Analyze -> Randomize -> Scramble Nodes'
    * This function randomly rearranges the node (centroids) for comparison with other centroid-using methods, as a possible way to demonstrate non-random behavior.
    * The randomize menu is likewise new and the 'Generate Equivalent Random Network' method was moved there.

3. Bug fixes.
    * Importantly fixed a bug with dt-based dilation not working in 2D, which I had accidentally introduced recently.

Version 0.7.1 Updates:
    * This was just a quick change of loosening package version requirements because I had made them too strict and it was making the package tough to install.

Version 0.7.2 Updates:
    * Added new option to the modify network qualities menu to remove node centroids with unassigned id values.
    * Bug fixes, mainly:
        * Had to fix a bug with the ripley's function that was making it always evaluate nodes of one id against themselves even when a seperate id was specified.
        * Fixed some bugs when processing 2D images.

Version 0.7.3 Updates:
    * Loosened package restrictions again because they were calling install issues.

Version 0.7.4 Updates

	* Bug fixes
	* The segmenter now has a GPU option that actually works quite a bit faster! Only available with CUDA toolkit and cupy.
	* The segmenter also now no longer leaks any memory.

Version 0.7.5 Updates

	* Bug fixes
	* The segmenter GPU option has been updated to include 2D segmentation and to also be able to load/save models.
	* A new function (Analyze -> Stats -> Calculate Generic Network Histograms has been added (gives a few histograms about the network and their corresponding tables. Previously stats mostly gave averages).
	* The function 'Analyze -> Data/Overlays -> Get Hub Information' now looks for the upper hubs unique to each separate network component. It will also ignore smaller components that have too few nodes to be reasonably considered having hubs relative to the threshold the user sets.
	* The function to split non-connected nodes has been improved a bit (its faster albeit still slowish - its a tough operation)
	* Removed python-louvain dependence, now uses networkx for Louvain partitioning. (On top of this, now the user can set the random seed they desire for partitioning for reproducibility purposes).

Version 0.7.6 Updates
    * Bug Fixes

Version 0.8.0 Updates

	* Added ability to threshold nodes by degree.
	* Improved image viewer window performance.
	* Bug fixes and a few optimizations.
	* Added ability to 'merge node identities' which just uses the nodes image as a reference for collecting 'identity' information from a group of other images - ie can use with cell nuclei (DAPI) to see what markers from the same imaging session overlap.
	* Added ability to search for specific nodes directly in the nodes image with 'shift + f' or right click.

Version 0.8.1 Updates

	* Added nearest neighbor evaluation function (Analysis -> Stats -> Avg Nearest Neighbor)
	* Added heatmap outputs for node degrees (Analysis -> Data/Overlays -> Get Degree Information).
	* Bug fixes and misc improvements.

Version 0.8.2 Updates

	* Bug Fixes.
	* Improved some of the image viewer window features.
	* New option to zoom in on specific windows by clicking + dragging while in zoom mode.
	* Added more features to UMAP/community neighborhood clustering (optional DBSCAN clustering, results more robust to node distribution)
	* Made Napari and optional rather than core dependency.
	* Added Cellpose as an optional dependency.

Version 0.9.0 Updates
	* Note that this includes updates for 0.8.3 - 0.9.0
	* Bug Fixes
	* Updated network histogram statistics menu, and moved degree distribution here
	* Added gray watershed
	* Updated binary watershed
	* Improved branch labelling method
	* Updated branch removal in skeletonization method to no longer trim branches that do not reach their node.
	* Added default branchpoint and branch-adjacency calculation options.
	* Improved speed of painting and panning.
	* Enabled the nearest neighbor method to handle non-centroid objects, for the first neighbor at least. And updated it to actually predict theoretical clustering when coloring the heatmap.
	* Improved segmenter.
	* Added centroids UMAP method.

Version 0.9.1 Updates
	* Adjusted the segment by 3D function to now show the 3D chunks in the preview mode. Previously it showed 2D segmentations in the preview which finished the current plane faster but didn't show accurate training data.
	* Adjusted the neighborhood heatmap predicted range value to now just simulate a uniform distribution rather than trying to use a mathematical algorithm. 
	* The image display window now uses image pyramids and cropping for zoom ins so it should run a lot faster on bigger images.
	* The community UMAP can now color them by neighborhood.
	* No longer zooms all the way out by default with right click in zoom mode. Now user needs to Shift + Right Click.

Versions 0.9.2 - 1.0.0 Updates
	* Tables can now be opened to the rightside upper widget if they are the right format.
	* Similarly, tables that have the format node id column:numerical values can now be used liberally to threshold the nodes, meaning most outputs of network analysis can be used to threshold nodes.
	* The overlay 2 is now cyan by default.
	* Moved some file menu options around.
	* The 'merge node id' option now offers interactive support for assisted thresholding for any new identity channels the user is trying to merge with.
	* The 'merge nodes' option now can provide centroids prior to the merge, since oftentimes objects end up on top of each other.
	* Erode can now optionally preserve object labels.
	* Added some compatibility for nodes being assigned 'multiple identities'
	* Image viewer canvas window can now be popped out into a separate window. 
	* Image pyramid calculation is more dynamic instead of using arbitrary size thresholds.
	* The 'network selection' table is now auto-populated when using the multiple-identity selector, and when using the node thresholder.
	* Bug Fixes

Versions 1.0.1 - 1.0.4 Updates
	* Heatmap theoretical distances can now be calculated based on an area constrained within a binary mask.
	* Added ability to generate violin plots using the table generated from merging node identities, showing the relative expression of markers for multiple channels for the nodes belonging to some channel or community/neighborhood
	* Other bug fixes, improvements.

Version 1.0.5 Update
	* Minor change to how the violin plots are normalized

Versions 1.0.6 - 1.1.3 Updates
	* Added handling if the user tries to load in a multichannel 3dimensional image (note this will not detect if you have a multi-channel image of 2d planes, it will think those are 3d. For now those can be split up with other software, or you can use the crop function to just isolate the channel you want as if it were a z-plane).
	* Added significance testing menu
	* Added ability to load a full sized highlight overlay from the file menu (in case you need it for a picture - on big images the highlight overlay is computed by slice so if you reload the channel its trying to highlight, it will alter the highlight overlay, but loading in the entire highlight overlay directly will stop this behavior until a new highlight is generated).
	* For the 'calculate edge < > node interaction' method - now can compute the length of nearby edges as an alternative option to just the volumes.
	* Added ability to select all nodes/edges participating in the network.
	* Can now intermittently downsample while making the network and id overlays now to make their relevant elements larger in the actual rendered output.
	* Added option when calculating branches to get some stats about them (lengths, tortuosity).
	* Added 'Clean Segmentation' function which just shows a menu grouping together some functions useful for cleaning up a segmentation.
	* Bug fixes and minor adjustments.

Versions 1.1.4 - 1.2.3 Updates
	* When using the intensity table used from interactive thresholding for multi-chan codex image (where the user assigns identities to cells based on their defined intensities) - added option to allow k-means clustering based on the results they obtained as a way to group together cells that express similar intensities of fluorescent markers. This is available through the Analyze -> Stats -> Show Identity Violins/UMAP. Should be a good way to auto detect different cell phenotypes in an image. The detected groups are saved as communities, and thereafter can be called once more through the Show Identity Violins/UMAP to comparatively assess the relative marker composition of each community, or to now generate a UMAP grouping the cells based on marker overlap but colored in the visualization by community (before this UMAP could also be shown but would color the cells based on one of their assigned identities).
	* Removed the option to input a directory for the few methods that could autosave to one when processing something (since saving is generally done elsewhere in the GUI).
	* Added 'Analyze -> Stats -> Calculate Branch Stats...' to the GUI. Before branch stats could be calculated by labeling branches, but calling this method on a channel with labeled branches in them will just let the user get the branch stats once more. Mainly added this since the user might want to manually reconfigure something about the labeled branches and re-obtain these stats afterwards.
	* Removed some parameters from a few of the functions that I felt were not very useful/were confusing to use.
	* Implemented an in-GUI tutorial available from the 'Help' menu
	* Added second 'neighbor label' option to label objects in one image based on whether they are continuous in space with labels in a second image. Previously the only option was based on rote distance to the nearest label. This one can better define things that are stemming from other things, for example I made it for first labeling branches of an 'Opened' segmentation (eroded, then dilated, which smooths out borders and can improve branch labeling), then relabeling the original branch segmentation based on the 'Opened' branches.
	* Updated the label correcting optional step for the branch labeler to assign discontinuous branches to have the label of an adjacent 'correctly labeled' branch rather than just taking on a unique label which should greatly improve the labeling schema if a broken label happens to show up.
	* Added a scalebar that can be toggled on an off in the main canvas.
	* Fixed saving of Boolean True/False arrays.
	* Fixed branch functions not working correctly when a temporary downsample was being applied
	* Added the 'filaments tracer' which can be used to improve vessel based segmentations
	* Removed options for 'cubic' downsampling during processing as this option actually made outputs worse.
	* Rearranged the 'Analyze -> Stats' menu
	* Added ability to attempt to re-merge contiguous large branches that had been split up into separate branches by the branch labeler.

Version 1.2.4-1.2.5 Update
	* Bug fixes
	* Updated license and readme files.
