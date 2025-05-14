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