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