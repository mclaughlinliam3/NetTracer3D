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