.. _analyze_menu:

==========
All Analyze Menu Options
==========

The Analyze Menu offers options for creating graphs and statistical tables.

* The first submenu is the network menu, which has functions for visualizing networks and analyzing network communities.

'Analyze -> Network -> Show Network'
-------------------------------

* Use this function to visualize the method in an interactable matplotlib graph.
* Note that this window will be very slow for graphs with a huge number of nodes.
* Selecting this displays the following menu:

.. image:: _static/analyze_1.png
   :width: 300px
   :alt: Network Vis Menu

Parameter Explanations
~~~~~~~~~~~~~~~

1. Enable the 'geo_layout' button to have nodes be placed in representative regions of their 3D location.
    * Their XY position in the graph will correspond to the image, while the node size will represent their Z-position (with larger nodes representing smaller Z-vals, ie. closer to the viewer).
    * Note that this method may run rather slowly on graphs with a huge number of nodes.
    * Keeping this unselected will group nodes using the NetworkX spring layout.
2. Execution Mode (Menu Includes the following options):
    1. Default:
        * All nodes are represented by their numerical ID, and nothing else.
    2. Community Coded (Uses Current...)
        * Colors nodes by their community, assuming they have already been community partitioned. Will prompt the user to partition if not.
    3. Node-ID Coded
        * Display nodes color-coded by their node ID (if it exists).
3. Output directory
    * If a string path is included to a directory, the resulting graph will be autosaved there.
    * Note these graphs can be saved in the matplotlib window anyway.
* Press Show Network to open a new matplotlib window displaying the graph.

'Analyze -> Network -> Generic Network Report'
-------------------------------

* This option will have the program report some basic things about the current Network 3D Objects.
* This includes the number of nodes, the number of edges, the number of nodes per 'node identity' property category, and the number of nodes per 'community' (if assigned).
* The report will go in the upperright table.

'Analyze -> Network -> Community Partition + Generic Network Stats'
-------------------------------
* Use this function to partition the nodes into communities.
* Node communities will be saved and loaded with any 'Save/Load Network3D object' options.
* Selecting this displays the following menu:

.. image:: _static/analyze2.png
   :width: 300px
   :alt: Com Menu

Parameter Explanations
~~~~~~~~~~~~~~~

1. Use Weighted Network
    * Enabling this option has the community partition consider graph weights.
    * By default, generated networks aquire weights when two or more discrete node objects join objects together.
    * Objects joined by heavily weighted edges will be more likely to be grouped into the same community.
2. Execution Mode (Menu Includes the following options):
    1. Label Propogation
        * Partition the network using NetworkX's label propogation algorithm.
    2. Louvain
        * Partition the network using NetworkX's louvain algorithm.
    * Both of these options are quick, efficient ways to group networks. Label propogation is a bit faster but more variable.
    * Note that network community detection in general (and in these cases) has some degree of randomness in how it decides to group objects (based on what nodes it starts from).
3. Community Stats
    * Whether or not to calculate community-based stats for the graph
    * If yes, these are the stats that are returned, referring to the entire network:
        * Modularity Entire Network
        * Number of Communities
        * Community Sizes
        * Average Community Size 
        * Number of Iterations (Louvain only - the number of iterations the algorithm ran)
        * Global Clustering Coefficient (NetworkX)
        * Assortativity (NetworkX)
        * Inter-community Edges (How many edges exist between communities)
        * Mixing Parameter (ratio of external to total edges for nodes)
    * And for each discrete community, these stats are returned:
        * Density (NetworkX)
        * Conductance (NetworkX)
        * Average Clustering (NetworkX)
        * Degree Centrality (NetworkX)
        * Average Shortest Path Length
    * These stats come from the NetworkX. Please see the below documentation for more information:
    * networkx documentation: https://networkx.org/
4. Seed (int): Sets the random seed for the community partition to use (since the starting point effects the outcome). You should use the same seed each time for reproducibility, or vary the seed to see how it effects partitioning. Leaving the seed empty will just use the seed of the rand (and numpy random) modules, which is initialized at program start.
* Press partition to seperate the nodes into communities based on the selected parameters. In addition to setting the node_communities property, tables showing the community for each node and the stats will be generated in the tabulated data widget.

'Analyze -> Network -> Identity Makeup of Network Communities (And UMAP)'
-------------------------------
* This method is designed to be run on groups of nodes that have been community partitioned and have associated 'node_identities' property, to evaluate their general compositions.
* It can yield compositional proportions of node identities per community or a weighted average of the compositions of all communities.
    * For the latter option, the communities are weighted by size, so larger communities contribute to this value more.
* It can also generate a UMAP for the communities. Within the UMAP, communities that are in close proximity have more similar identity compositions.
* This method can be a good way to characterize what communities in the network consist of. For example, if I have grouped neighborhoods of different cell types and am wondering what a generic community looks like.



Parameter Explanations
~~~~~~~~~~~~~~~

#. Mode
    * The dropdown menu has two options:
        1. Average Identities per community - This option provides compositional info on all communities.
        2. Weighted Average Identity of All Communities - This option provides compositional info of all communities, weighted by community size. (Does not support UMAP)
#. Generate UMAP
    * Select this option to generate a UMAP comparing the community compositions.
#. Label UMAP Points How?
    * 'No Label' - UMAP points are not labeled
    * By Community - Assigns each point their numerical label.
    * By Neighborhood - Colors communities in the UMAP by what neighborhood they belong to, presuming they have been assigned a neighborhood via 'Analyze -> Network -> Convert Network Communities...'
#. Min Community Size to be grouped...
    * If empty, this param does nothing.
    * If an int is entered, any communities with nodes fewer than this val will not be included in the UMAP — since we might not care about small, insignificant communities.
#. Return Node Type Distribution Robust UMAP
    * Normally, communities are grouped in the UMAP by their proportional compositions of node types.
    * If this option is selected, they will instead be grouped based on how much they 'overrepresent' specific node types. Overrepresenting = the proportion of nodes of that type in the community (vs all nodes of that type) is greater than the proportion of all nodes within that community (vs all nodes in the image).

* Press 'Get Community ID Info' to populate the data to the upper right tabulated data widget, and to show the UMAP if selected.

Algorithm Explanation
~~~~~~~~~~~~~~~~~~~~

* If not using the weighted average of all communities:
1. Simply finds the proportion of each identity per community.

* If generating the UMAP (with the umap module):
1. Extract community data by getting community IDs and stacking their community composition arrays (from above) into a matrix
2. Initialize UMAP reducer and random seed (42) for reproducible dimensionality reduction
3. Transform compositions using UMAP to reduce high-dimensional cluster vectors to 2D coordinates
4. Create scatter plot with points colored by cluster ID.
5. Print composition analysis showing the raw data and identifying the two most dominant classes per community


* If using the weighted average for all communities (does not support UMAP):
1. Groups nodes by their community ID
2. For each community, counts the number of nodes with each identity type
3. Weights these counts by the size of the community
4. Sums these weighted counts across all communities
5. Normalizes the results twice: first by the total number of nodes, then to ensure all proportions sum to 1
6. Returns a dictionary mapping each identity type to its weighted proportion in the network

'Analyze -> Network -> Convert Network Communities Into Neighborhoods (Also Returns Compositional Heatmaps)'
-------------------------------

* This method finds the average of compositions of all communities (assuming 'node_identities' exist - then uses the above function), then groups similar communities into 'neighborhoods'.
* The purpose of this is to let the user crunch communities into a smaller set of neighborhoods for analysis of similar domains across the image.
* The number of neighborhoods assigned is up to the user.
* Running this method will also show a heatmap graph of what 'node_identity' is prominant in what neighborhood.
* Neighborhoods by default get assigned by size as well, with 1 being the largest group, and n (the highest neighborhood ID) being the smallest, allowing their relative sizes to be easily compared.
* Running this method will reassign the 'communities' property to these neighborhoods instead, so be sure to save the former first.
    * This is so the user can use all community-associated functions on the new neighborhoods instead.
    * However during an active session, this method will always run on the original communities and not the neighborhoods (it stores it in another temp property only for that purpose). This lets the user run this method with different params to evaluate different neighborhoods, but note that these temp communities are not used for anything else.

Parameter Explanations
~~~~~~~~~~~~~~~

#. Num Neighborhoods
    * The number of neighborhoods the user wants to group communities into. Presumably, you would want the number of communities to be larger by a logical amount than the number of new neighborhoods.
    * Arbitrary neighborhood numbers can only be used for K-means clustering. DBSCAN clustering will always decide on its own how many to use. K-means will also try to guess a good neighborhood count if nothing is entered.
#. Clustering Seed
    * The random seed (int) used for neighborhood assignment. By default this is 42.
#. Min Community Size to be grouped...
    * If empty, this param does nothing.
    * If an int is entered, any communities with nodes fewer than this val will be assigned to 'Neighborhood 0' — since we might not care about small, insignificant communities.
#. Return Node Type Distribution Robust Heatmaps
    * This method always returns a heatmap showing the proportional composition of each neighborhood for each ID type.
    * However, by pressing this option, two more heatmaps will be returned as well:
    * The second heatmap shows in each cell the proportion of that node type in that neighborhood as compared to the total available nodes of that type.
    * The third heatmap (which I like to use) takes the results from the second and divides them by the proportion of total nodes (of any type) that comprise that neighborhood. This gives us a result showing what node types are 'overrepresented' in this neighborhood (since we would expect each node type within a neighborhood to have the same proportional representation vs all nodes of that type as the total representation of all nodes of that neighborhood itself)
        * In short though, cells with values above 1 overrepresent that node type, while values below 1 underrepresent it. This is a pretty great way to eyeball compositional anomolies.
#. Mode - A dropdown menu to select the clustering algorithm.
    * KMeans - Uses K-means clustering. (Generally recommended)
    * DBSCAN - Uses DBSCAN clustering. (Somewhat experimental). Note that DBSCAN likes to assign low-appearing groups as 'outliers', which in the current implementation get assigned to 'Neighborhood 0' (Along with any communities the user decided to threshold out by size).

* Press 'Get Neighborhoods' to group the nodes into neighborhoods and to get the heatmaps. This also returns a number of tables to the tabulated data widget (top right), including:
    * Tabulated versions of all heatmaps.
    * Proportion of total nodes in network for each neighborhood.
    * Nodes to neighborhood ID table.

Algorithm Explanation
~~~~~~~~~~~~~~~~~~~~

* This method primarily uses the sklearn.cluster KMeans algorithm: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
1. Finds the composition of all communities using 'Analyze -> Network -> Identity Makeup of Communities' logic.
2. Converts compositions to numpy array to prepare data for scikit-learn clustering algorithm
3. Applies K-means clustering with specified number of neighborhoods and random seed

* Alternatively, the sklearn DBSCAN algorithm can be used: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
1. Calculate min_samples - minimum neighbors a point needs to be a 'core point': max(3, sqrt(n_samples) * 0.2)
2. Estimate eps (neighborhood radius): Use 80th percentile of 4th nearest neighbor distances. Non-core points within eps of core points become border points of that cluster.
3. Run DBSCAN with calculated parameters
4. Points that are neither core points nor within eps of core points become outliers

* If using KMeans, and no neighborhood count is provided:
1. Neighborhood counts from sizes 1 to 20 will be temporarily generated.
2. They will be graded on quality based on their calinksi harabasz score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
3. The neighborhood count with the highest score will be used.

'Analyze -> Network -> Create Communities Based on Cuboidal Proximity Cells?'
-------------------------------

* This method splits the image into cells (of user-defined size) and assigns nodes to be in communities based on whether they share a cell.
* It doesn't have anything to do with the network but is an alternate way to group the nodes into communities, with a greater spatial focus.

Parameter Explanations
~~~~~~~~~~~~~~~

#. Cell Size
    * The volume of a cell (Can be 2D or 3D). The cells will always be cubes (or squares).
#. xy scale
    * The 2D plane scaling of the image.
#. z scale
    * The 3D voxel depth scaling of the image.

* The latter two params will scale the cell to be cuboidal based on provided scaling (ie its side lengths will be the same in true units).
* Press 'Get Communities' to assign the communities based on cells.

* The second submenu is 'Stats', and is primarily used to create tables and graphs about the network or image morphology.

'Analyze -> Stats -> Calculate Generic Network Stats'
-----------------------------------------
* This function simply generates and displays (in the tabulated data widget) a number of generic stats about the network.
* The following stats will be generated:
    * num_nodes
    * num_edges
    * density
    * is_directed (Note that networks currently will always be undirected)
    * is_connected
    * num_connected_components
    * largest_component_size
    * avg_degree
    * max_degree
    * min_degree
    * avg_betweenness_centrality
    * avg_closeness_centrality
    * avg_eigenvector_centrality
    * avg_clustering_coefficient
    * transitivity
    * diameter
    * avg_shortest_path_length
    * is_tree
    * num_triangles
    * degree_assortativity
    * Unconnected nodes (left out from node image)
* These stats are all more or less generated by networkx.
* Please see networkx documentation for more information: https://networkx.org/

'Analyze -> Stats -> Network Statistics Histograms'
-----------------------------------------
* This function allows easy generation and displaying (as matplotlib histos and in the tabulated data widget) of a number of network histograms about distributions of node properties in the network.
* The histograms displayed are all generated using networkx functions to find information about the graph, such as shown in this networkx documentation: https://networkx.org/nx-guides/content/exploratory_notebooks/facebook_notebook.html (Please use this reference for information about the histograms)
* Assuming you have calculated a network, selecting this option will display the following menu:

.. image:: _static/network_histos.png
   :width: 500px
   :alt: Network_histos
*Once a network is generated, selecting any of the green button options will generate a graph of the corresponding distribution, and export the data to the tabulated data widget*

Parameter Explanations
~~~~~~~~~~~~~~~~~~~~
#. Degree Distribution...
    * Shows the count of connections per node across your network.
    * Use to identify network architecture: power-law distributions indicate hub-based networks, normal distributions show egalitarian connectivity.
#. Shortest Path Length Distribution...
    * Displays the minimum number of steps between all node pairs.
    * Use to assess network efficiency: narrow peaks at low values indicate efficient "small world" networks.
#. Degree Centrality...
    * Measures direct influence through immediate connections.
    * Use to identify nodes with the most direct reach in your network.
#. Betweenness Centrality...
    * Identifies critical bridge nodes that connect different network regions.
    * Use to find bottlenecks and assess network vulnerability to node removal.
#. Closeness Centrality...
    * Measures how quickly each node can reach all other nodes.
    * Use to identify nodes optimally positioned for information spreading.
#. Eigenvector Centrality...
    * Measures prestige by weighting connections to highly-connected nodes.
    * Use to identify nodes connected to important hubs (quality over quantity of connections).
#. Harmonic Centrality...
    * Robust version of closeness centrality that handles disconnected components.
    * Use when your network may have isolated clusters or components.
#. Load Centrality...
    * Shows traffic burden each node would carry in network flow.
    * Use to identify potential communication bottlenecks and workload distribution.
#. Current Flow Betweenness...
    * Models flow using electrical circuit principles (considers all paths, not just shortest).
    * Use for more realistic assessment of node importance in flow networks.
#. Communicability Betweenness...
    * Measures bridging importance based on walks of all lengths.
    * Use to identify nodes important for sustained, multi-step communication processes.
#. Clustering Coefficient...
    * Measures how interconnected each node's immediate neighbors are.
    * Use to identify tight-knit communities versus sparse, tree-like regions.
#. Triangle Count...
    * Counts triangular connections (three mutually connected nodes) per node.
    * Use to assess local group cohesion and community strength.
#. K-Core Decomposition...
    * Identifies nested dense subgroups where all nodes have minimum degree k.
    * Use to reveal hierarchical community structure and dense network cores.
#. Eccentricity...
    * Shows the maximum distance from each node to any other reachable node.
    * Use to identify peripheral versus central nodes and assess network compactness.
#. Node Connectivity...
    * Measures minimum nodes needed to disconnect each node's neighborhood.
    * Use to assess local network robustness around individual nodes.
#. Average Dispersion...
    * Measures how scattered each node's neighbors are from each other.
    * Use to distinguish bridge nodes (high dispersion) from community-centered nodes (low dispersion).
#. Network Bridges...
    * Identifies edges whose removal would disconnect network components.
    * Use to find critical connections essential for network cohesion.

'Analyze -> Stats -> Radial Distribution Analysis'
-----------------------------------------
* This method creates a graph showing the average number of neighboring nodes (of any given node) on the y axis and the distance from any given node in the x axis.
* Use this method to evaluate how far apart your connected nodes tend to be in 3D space, and how those relationships are distributed.
* For example, we would typically expect more efficient networks to mostly have an abundance of short connections and a minority of long connections.

Parameter Explanations
~~~~~~~~~~~~~~~

1. Bucket Distance...
    * This is the distance that will be used as a step size while searching outward from nodes in the graph to evaluate how close in 3D space their neighbors are.
2. Output Directory
    * If a string path is included to a directory, the resulting graph will be autosaved there.
    * Note these graphs can be saved in the matplotlib window anyway.

* Press 'Get Radial Distribution' to open a new matplotlib window showing the graph, and also place the obtained data in as a new table in the tabulated data widget.

'Analyze -> Stats -> Identity Distribution of Neighbors'
-----------------------------------------
* This method allows us to explore what kinds of nodes (as categorized by their node_identities) tend to be located nearby/connected to nodes of some desired ID.
* Use this method when you want to characterize what interacts with what, for example, if I have cellular neighborhoods and want to know what's near what.
* Selecting this displays the following menu:

.. image:: _static/analyze4.png
   :width: 200px
   :alt: NeighborID Menu

Parameter Explanations
~~~~~~~~~~~~~~~

1. Root Identity to Search...
    * This is the identity of the sorts of nodes we will search outward from. The neighborhoods of these nodes will be characterized.
2. Output Directory
    * If a string path is included to a directory, the resulting outputs will be autosaved there.
3. Mode (Menu Includes the following options):
    1. From Network - Based on Absolute Connectivity
        * Reveals information about neighors based on the connectivity of the network.
    2. Use Labeled Nodes - Based on Morphological Densities 
        * Reveals information about neighors based on what sorts of nodes are physically in the vicinity.
4. Search Radius (if using Mode 2)
    * The distance that nodes will search to characterize their neighborhoods. Option one currently will always just search for immediate network neighbors.
5. Fast dilation option (if using Mode 2)
    * If disabled, searching will be done with perfect distance transforms. If enabled, searching will be done with with psuedo-3D kernels which may be faster but less accurate. For more information on this algorithm, see :ref:`dilation`.

* Press 'Get Neighborhood Identity Distribution' to display a few matplotlib barcharts, with associated data tables being added to the tabulated data widget.
* The following tables (and corresponding graphs) will appear:
* If using mode 1:
    1. Neighborhood Distribution of Nodes in Network from Nodes: 'X'
        * Shows how many total neighbors of each ID that nodes of ID 'X' have (including other type 'X').
    2. Neighborhood Distribution of Nodes in Network from Nodes 'X' as a proportion of total nodes of that ID.
        * For each ID category, shows what proportion of that node type in the network are neighbors of nodes of ID 'X' (including other type 'X')
* If using mode 2:
    1. Volumetric Neighborhood Distribution of Nodes in image that are 'y' distance from nodes: 'X'
        * Shows the total volumes of nodes of each ID within distance 'y' from nodes of ID 'X' (does not include other type 'X')
    2. Density Distribution of Nodes in image that are 'y' from Nodes 'X' as a proportion of totaly node volume of that ID.
        * For each ID category, shows what proportion of the volume of that node type are within distance 'y' from nodes of ID 'X' (does not include other type 'X')
    3. Clustering Factor of Node Identities within 'y' from nodes 'X'
        * For each ID category, shows the volumetric density of nodes of that ID type within distance 'y' from nodes of ID 'X', divided by the densities of nodes of that ID type in the entire image. (does not include other type 'X')
        * This is also known as relative density. Essentially, a val greater than 1 means said node ID is unevenly distributed to be closer to nodes of ID 'X', while a val less than 1 means they are preferentially avoiding nodes of ID 'X'.

Algorithm Explanation
~~~~~~~~~~~~~~~~~~~~

1. Mode 1 just counts neighbors that are immediate neighbors in the network of the desired node ID.
2. Mode 2 searches using either a distance transform or psuedo-3D binary dilation. It searches outward from nodes of the desired ID type, and hence does not actually include them. This is why this option never evaluates its own clustering.


'Analyze -> Stats -> Ripley Clustering Analysis'
-----------------------------------------
* This method generates a Ripley's K curve, which is a function that compares relative object clustering to distance r from some random node.
* It is a good way to identify if objects are clustering or dispersed, and how that varies through an image.
* This method can evaluate if a node of some identity is clustered around a node of another identity type, or just if nodes of one type are clustered with themselves. 
* This method can be run with labeled nodes, or just node centroids themselves. It will prompt for node centroids if they do not exist. Since it uses centroids, it says nothing about the actual shapes of nodes.
* Selecting this option displays the following menu:

.. image:: _static/ripley_menu.png
   :width: 400px
   :alt: Ripley Menu

Parameter Explanations
~~~~~~~~~~~~~~~
#. Root Identity to Search for Neighbors
    * This is the node identity type whose neighborhood you want to evaluate for clustered objects.
#. Targ Identity to be Searched For
    * This is the node identity who will be evaluated for cluster behavior around param 1.
    * Note that param 1 and 2 only appear if identities are assigned. Otherwise, all nodes will just evaluate clustering against themselves.
#. Bucket Distance for Searching For Clusters...
    * This is the bucket distance for each iteration of r. It is auto-scaled for your image, so enter a true distance here if you have scaling properties set.
    * Note that smaller buckets will slow down processing time (in exchange for higher fidelity).
#. Proportion of image to search...
    * A 0-1 float representing the proportion of the image to search from each node.
    * A value of 1 will have each node try to evaluate the clustering of every other node in the image, while values closer to 0 will restrict the function calculation to just the immediate neighborhood.
    * Note that higher values will increase border artifacts (Since the method can't 'see' nodes beyond the image borders so it presumes those regions to be empty, decreasing clustering appraisal).
#. Exclude Root Nodes Near Borders?
    * For border safety, the user can enable this to optionally have the nodes near the image boundaries not search for neighbors. The degree to which the border nodes are excluded is set with the following param.
#. Proportion of most internal nodes to use...?
    * If the above param is enabled, this value will tell it what degree of internal nodes to use.
    * This should be a float between 0 and 1. It represents the proportion of most internal nodes, so higher vals exclude more of the border.
    * As an example, setting this to 0.9 would have it include only the 10% most internal space of the array to find nodes to search from. If no nodes can be found within these bounds, the analysis will not be performed.
#. Define boundaries how?
    * If restricting analysis to be within borders, this dropdown menu can be used to tell the program what to consider as 'boundaries'. By default, it will use the boundaries of the entire array. However, if your image is, for example, a piece of tissue with background space, and you want the search to be restricted to stay within the actual tissue volume, you will need to first create a binary mask for your foreground. Then, place that mask in any of the other channels (besides nodes). Lastly, use the dropdown menu to select the channel containing the mask. The easiest way to yield a foreground mask is to first threshold via intensity, then to run 'Process -> Image -> Fill Holes' if desired. If the tissue is 2D, the user may also trace the foreground with the pen tool, then run Fill Holes.
#. Keep search radii within border...?
    * If restricting analysis to be within borders, and you want the nodes to also be forced to keep their search radii within those borders, this setting can be enabled. It will automatically calculate the minimum distance from the most external nodes being considered to the borders, and use that distance for the search radii. Generally speaking, this setting should be enabled if masking within tissue boundaries, or rather if not masking but also not using param 9. Note that this will override param 4.
#. Use Border Correction...
    * This param can be enabled if the user is not masking the root nodes, but also wants to use search radii that extend beyond the image borders. To compensate for edge artifacts, it will have the program clone centroids by reflecting them over the borders of the array, essentially forcing the space beyond the array to mimic the space within the array (an extrapolation). As mentioned, this does not really work if the user is forcing the analysis to stay within a tissue mask, as the reflection is designed for the boundaries of the rectanguloid image, not a weird mask shape.

* Press "Get Ripley's H" to have the program calculate both the Ripley's K function and Ripley's H function for your dataset. Tables for each will populate the tabulated data widget, while some form of the following graph will appear:

.. image:: _static/ripley_graph.png
   :width: 400px
   :alt: Ripley Graph
*In this case, the x axis represents the distance from any random node, and the y, a factor representing the clustering intensity observed around nodes at that distance. The blue line is our observed line, while the red line represents expected behaviors from a Poisson distribution of nodes. Essentially, regions above the red dotted line are unexpectedly clustered, while those below are unexpectedly dispersed. The right graph is a normalized version of the left, to have a straight center line. Note that due the possibility of border artifacts in the datasets, it might be best to compare between multiple datasets or with a dataset of randomly-seeded nodes, rather than directly to the red line*

Algorithm Explanations
~~~~~~~~~~~~~~~
* This algorithm is an implementation of the Ripley's K function. See 10.1016/j.bpj.2009.05.039
#. We take two sets of points: root points and target points (these can be the same set)
#. We build a KDTree from the root points for efficient nearest-neighbor searches
#. We calculate the volume/area of the study region
#. We compute the intensity (λ) as number of reference points divided by volume
#. For each root point at each distance in our bucketed r_values, we find Neighbors using KDTree and record how many target points are within this radius
#. If we're comparing a set to itself, remove self-counts to avoid counting points as their own neighbors.
#. Sum all the weighted counts and normalize by:
    * Number of subset points (n_subset)
    * Point intensity (λ)
#. Return the array of K values for each radius value
#. K values can then be normalized to H values by 'h_values = np.sqrt(k_values / np.pi) - r_values' (in 2D), or 'h_values = np.cbrt(k_values / (4/3 * np.pi)) - r_values' (in 3D)
#. These are plotted versus the theoretical functions 'theo_k = np.pi * r_values**2' (2D) or 'theo_k = (4/3) * np.pi * r_values**3' (3D), while theoretical H values are just 0.

* For border correction:
#. Restricting to internal nodes for the whole image is done by just checking if the nodes are beyond the requested distance to the border.
#. If the user asks internal nodes to remain within a masked space, the distance transform to the background of the mask is obtained. This dt mask is thresholded to only contain the internal proportion the user desired. Finally, the root nodes within that volume are considered valid.
#. If the user is using the node reflection option:
    * For 2D: Creates 8 potential mirror regions (4 edges + 4 corners)
    * For 3D: Creates 26 potential mirror regions (all adjacent cubes minus center)
    * Each region defined by direction vectors (-1, 0, +1 for each dimension)
    * For each mirror region, finds points within max_r distance of relevant boundaries
    * A point needs mirroring if it's close enough to a boundary that analysis might miss neighbors
    * For qualifying points, creates copies using reflection formula: new_coord = 2 × boundary - old_coord
    * Applies this transformation only to dimensions where mirroring is needed
    * Preserves other coordinates unchanged
    * Returns original points plus all mirrored copies

'Analyze -> Stats -> Community Cluster Heatmap'
-----------------------------------------

* This method plots the nodes into a 2D or 3D graph, with a color corresponding to community density.
* Red nodes are higher density than expected in a community, blue ones are lower density than expected.

Parameter Explanations
~~~~~~~~~~~~~~~

#. (Optional) - Total Number of Nodes
    * The total number of nodes is used to decide how many nodes belong in a community on average.
    * If unassigned, the program will just get the number of nodes that exist in the current properties.
    * This is here in case the nodes in the active session are a subset (ie some number of nodes have been filtered out with the excel helper). In that case, the user can still enter the number of nodes that belong in the dataset if the filtering had not occurred.
#. Use 3D Plot...
    * By default, the program will graph the heatmap in 3D.
    * Disable this if your data is 2D. Do not disable if it is 3D as the program will get confused.
#. Overlay
    * If enabled, the heatmapped will be returned as an RGB image overlay that goes into Overlay2, rather than a matplotlib graph.

* Press 'Run' to show the heatmap graph, and yield a table showing community id vs density intensity.
* It will require you to get 'node_centroids' and 'communities' properties if unassigned.

Algorithm Explanations:
~~~~~~~~~~~~~~~

1. Determine total nodes by trying multiple fallback sources: network nodes, centroids, identities, or unique node array values.
2. Calculate baseline density as the expected nodes per community if randomly distributed (total nodes / num communities).
3. Compute heat values using natural log ratio of actual community size to expected random size.
4. Generate heatmap visualization with matplotlib.

'Analyze -> Stats -> Average Nearest Neighbors'
-----------------------------------------

* This method will provide information about the nearest neighbors of your nodes.
* If node identities are assigned, the nearest neighbor information can be specific about the relationship between two identity types. Otherwise, it will just look at all the nodes together.
* The output can be the distribution of nearest neighbor values (+ their average), or it can be the average of all identity combinations (for bulk processing).
* This method can also yield heatmaps for nearest neighbor relationships as either graphs or image overlays.

Parameter Explanations
~~~~~~~~~~~~~~~

#. Root Identity... (If node identities property exists) - Identities of this node type will be evaluated for nearest neighbors of some other node type.
#. Neighbor identities... (If node identities property exists) - Identities of this node type will be searched for. Can be the same as param 1, or can include all nodes except param 1.
#. Number of Nearest Neighbors... - Default set to 1. This is the number of nearest neighbors each node will find. If 1, it just looks for its closest neighbor distance. Increasing this value will have each node instead get the average distance to that many nearest neighbors. (This value will cause the program to return if it is greater than the number of possible neighbors).
#. Use Centroids? - Whether to use centroids to find the neighbors, or to search from entire objects. Note that centroids is faster but only works well for spheroids. Entire objects, however, do not support averaging out multiple nearest neighbor, instead always setting the above param to 1 if selected.
#. Heatmap - Enabling this will cause a heatmap to be generated. Red nodes will be closer on average to their nearest neighbor, while blue nodes will be further.
#. 3D - If generating a matplotlib heatmap, enabling this will make the graph 3D. Disabling it will make it 2D.
#. Overlay - If enabled, the heatmap will be created as an image overlay in Overlay2 channel instead of a graph.
#. For heatmap, measure theoretical point distribution how? - When generating the heatmap, nodes are colored based on an approximate estimate of the distance between points, if they were evenly (note - not randomly) distributed throughout the array. By default, this dropdown menu is set to 'Anywhere', which tells the program to consider how the distances would be if the points were uniformly distributed throughout the entire image. However, if your image contains background, and you want it to consider only distribution throughout the foreground (i.e. an actual tissue), you will need to first create a binary mask for your foreground. Then, place that mask in any of the other channels (besides nodes). Lastly, use the dropdown menu to select the channel containing the mask. The easiest way to yield a foreground mask is to first threshold via intensity, then to run 'Process -> Image -> Fill Holes' if desired. If the tissue is 2D, the user may also trace the foreground with the pen tool, then run Fill Holes.
#. Quantifiable Overlay - If enabled, will generate a grayscale image with each node being assigned a val equal to its calculated nearest neighbor distance, which will go in Overlay1.



* Pressing 'Get Average Nearest Neighbor...' will yield a table of every 'root' node paired to its average distance to the desired number of nearest neighbors. It will also create a heatmap or 'quantifiable overlay' if selected.
* (If node identities property exists) - Pressing 'Get All Averages' will yield a table of the average nearest neighbor distance (for the desired number of nearest neighbor) across all nodes for every identity vs identity combination available. This can be a fast way to query the dataset, but it does not yield distributions and heatmaps, which need to be individually obtained. However, note that this should not be used if the number of identities would make this cumbersome.
* Note that this method automatically applies the xy_scale and z_scale set in the current properties. To ensure property distances, please make sure those are correct in Image -> Properties. By default, they are 1. 

Algorithm Explanations:
~~~~~~~~~~~~~~~

1. Depending on the desired identities, the nodes are broken into a root set and a neighbor set.
2. The centroids (or if not using centroids, assesses entire object borders obtained from skimage find_boundaries method) of the neighbor set are used to build a KDTree, which is a points-based data structure good for querying distance relationships. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
3. For each point in the root set, the desired number of nearest neighbors are obtained by querying the KDTree. These values are averaged per point and returned. The total average for the set is also returned.
4. When generating the heatmap, color intensity is based on whether the object is closer than would be expected in a uniform distribution, with ln(approx expected dist in a uniform distribution / actual dist of point) being used to create the color scale.
5. To measure the theoretical distance in a uniform distrubition, the image is cloned as a numpy array, and all the 'target points' are uniformly distrubted throughout it (or throughout the masked area, if that is set up). The most central of these points is selected and used with the same KDTree method to search for neighbors with the same parameters as the actual analysis. Because one of the target points is used to search, it will be skipped (since distance to itself is 0), and therefore if the number of nearest neighbors searched for is equal to the number of available target points, the program will approximate the most furthest distance by just assuming it to be equal to the second-furthest-distance.

'Analyze -> Stats -> Calculate Volumes'
-----------------------------------------
* This method finds the volumes of all objects in the 'Active Image'.
* The volumes are scaled by the axis scalings and returned as a table in the tabulated data widget.
* Algorithm explanation: This method uses the np.bincount() method to count each label and then just multiplies the outputs by the scalings.

'Analyze -> Stats -> Calculate Radii'
-----------------------------------------
* This method finds the largest radii of all objects in the 'Active Image'.
* It may be good to use, for example, on labeled branches to evaluate how thick the branches are.

Parameter Explanation
~~~~~~~~~~~~~~~~~~~~
* This method has one parameter, 'GPU'
* If you enable it, the system will attempt to attempt to use the GPU to calculate. Note that this is only possible with a working CUDA toolkit.

Algorithm Explanation
~~~~~~~~~~~~~~~~~~~~

1. The scipy.ndimage.find_objects() method is used to get bounding boxes around all the labeled objects.
2. For each object, a subarray is cut out around it using its bounding box, with padding on all sides.
3. The object in question is boolen indexed within its subarray.
4. The scipy.ndimage.distance_transform_edt() method is used to get a distance transform for the object, with the maximum value (ie, furthest from the background) representing the largest radii.
5. This process is paralellized across all available CPU cores. It *will* hog your entire machine if given a big task.

'Analyze -> Stats -> Calculate Node < > Edge Interactions'
-----------------------------------------
* This method will provide information about the volume of positive 'edge' image surrounding each labeled object in your 'node' image.
* You would essentially use it for a basic measurement of how much the edge channel image is surrounding each node.
* This measurement is performed for every node in the image individually.
* When you select this option, you will see this menu:

.. image:: _static/analyze5.png
   :width: 800px
   :alt: edgenode Menu

'Analyze -> Stats -> Show Identities Violin/UMAP'
-----------------------------------------
* This method can be used to visualize normalized violin plots and UMAPs for nodes that were assigned identities via multiple channel markers (via 'File -> Images -> Node Identities -> Assign Node Identities from Overlap with Other Images')
* The aforementioned identity assignment funtion produces a table that shows the average intensity of each node for each marker. Please save this table from the upper-right data tables for use in this function. This data is the only one natively compatible with this function.
* Upon running, the user will be prompted to retrieve this data table as a .csv or .xlsx file. Note that this table would ideally be the one created during the aforementioned node assignment, with the node identities themselves also derived from that function.

Parameter Explanation
~~~~~~~~~~~~~~~~~~~~

1. 'Return Identity Violin Plots?' - 'None' by default, but the dropdown menu can be used to select one of the current node_identities in the session, which informs the program to yield a violin plot displaying the normalized intensity expression for each channel of all nodes belonging to the aforementioned identity. This is useful for seeing what other channels a particular identity is generally positive in.
2. 'Return Neighborhood/Community Violin Plots?' - 'None' by default, but the dropdown menu can be used to select one of the current communities (or rather neighborhoods, if communities have been grouped into neighborhoods) in the session, which informs the program to yield a violin plot displaying the normalized intensity expression for each channel of all nodes belonging to the aforementioned community/neighborhood. This is useful for seeing what channels constitute a particular community/neighborhood.
* To view any designated violin plots, press 'Show-Z-score-like Violin'. The plot will be shown, and the corresponding data will populate the upper right data tables.
* Alternatively, "Show Z-score UMAP" may be pressed to show a UMAP of the intensity Z-score for each node relative to the identity of each channel. (This is the same UMAP that can be displayed at the end of 'File -> Images -> Node Identities -> Assign Node Identities from Overlap with Other Images')

Algorithm Explanation
~~~~~~~~~~~~~~~~~~~~

1. For both violin-plot producing methods, the aforementioned data table is first normalized in a Z-score-like fashion. Essentially, for all nodes belonging to each unique identity, the minimum of those nodes is obtained. The data table values are then normalized using a Z-score, but centered around the minimum valid intensity for each identity (as it corresponds to each channel, i.e. a CD31 channel will be centered about the min of nodes with the CD31 identity).
    * This purpose of this normalization is so that the values in the data table reflect how far the nodes in that channel deviate from what the user designated as a true example of a node bearing that identity.
    * Pretty much, if evaluating channel identity overlap, any other channels that have violin values greater than 0 represent some amount of 'valid' overlap. If evaluating communities/neighborhoods, violins with values greater than 0 mean that neighborhood has nodes expressing the corresponding 'valid' amount of that marker.
    * If a column cannot be matched to an identity, the program will just take the median of the entire column to be the normalizing point, rather than the minimum value of the 'valid' points.
2. The normalized data table is then masked to contain only the nodes of the specified identity/neighborhood/community.
3. These resultant data are used to yield violin plots, with the channels corresponding to a violin, and the normalized node intensities within the masked data for that channel creating the violin.

* UMAP generation instead utilizes standard Z-scores, as the altered normalization is not relevant when comparing nodes in this manner. The UMAP itself is created with the Python umap module.

Parameter Explanations
~~~~~~~~~~~~~~~

1. node_search:
    * This value represents the distance one would like to search outwards from the nodes image to quantify edge interactions, and is scaled with the current image scalings.
2. Execution Mode:
    * This dropdown menu has two options:
        1. 'Include regions inside node' will include the node itself in the search region.
        2. 'Exclude regions inside node' will have the node only use the regions outside of it to search.
3. Use Fast Dilation...
    * If disabled, searching will be done with perfect distance transforms.
    * If enabled, searching will be done with psuedo-3D kernels, which may be faster but imperfect at measuring.
    * For more information on this algorithm, see :ref:`dilation`.

* Press Calculate to run the method with the desired parameters. The output data will be used to create a new table in the tabulated data widget.

Algorithm Explanation
~~~~~~~~~~~~~~~~~~~~

1. The scipy.ndimage.find_objects() method is used to get bounding boxes around all the labeled objects in the nodes channel.
2. For each object, a subarray is cut out around it using its bounding box, that includes the object plus any additional space that it will need to perform a search/dilation.
3. The same subarray is cut out of the edges channel (for neighborhood comparison).
4. The node object in question is boolen indexed within its subarray.
5. If not using the fast dilation option, then the scipy.ndimage.distance_transform_edt() method is used to get a distance transform for the object. This distance transform is thresholded based on the desired distance away from the node we want, then binarized.
6. If using fast dilation, the above is performed using psuedo-3D binary kernels without having to take a dt.
7. If internal regions are being excluded, an inverted boolean array of the original shape is used to 'cut out' the core from the dilated binary mask. The binary dilated mask is then multiplied against the edge subarray to isolate edges specific to the dilated region.
8. These edges are then counted, scaled volumetrically (by multiplying the three axis dimensions by the counted number), then added to a label:volume dictionary that will be eventually returned.
9. This process is paralellized across all available CPU cores. It *will* hog your entire machine if given a big task.

* The third submenu, 'Data/Overlays', has hybrid functions that both produce data while generating Overlays for the Image Viewer Window

'Analyze -> Data/Overlays -> Get Degree Information'
--------------------------------------
* This method can be used to extract information about the degrees of nodes in the image, while generating Overlays representing the same.

* When you select this option, you will see this menu:

.. image:: _static/analyze6.png
   :width: 800px
   :alt: edgenode Menu

Parameter Explanations
~~~~~~~~~~~~~~~

1. Execution Mode
    * This dropdown menu has three options:
        1. 'Just make table' - places a table with the ID of each node and its degree in the tabulated data widget, without generating any overlays.
        2. 'Draw Degree of Node as Overlay...' - This method creates an overlay where the degree value of each node is literally drawn onto its centroid as an overlay (ie, a node of degree 5 has a 5 drawn at its centroid). This can be used to quickly eyeball node connectivity.
        3. 'Label Nodes by Degree...' - This method takes each node label and reassigns its label to its degree. The idea would be to export the image and do downstream analysis elsewhere while thresholding for specific degree values.
            * Note this thresholding can be done in NetTracer3D by using the intensity thresholder.
        4. Create Heatmap of Degrees - Places in Overlay 2 an RGB heatmap of degrees. Degrees higher than average are more red while those lower than average are more blue.
2. Proportion of high degree nodes to keep...
    * By default this is set to 1 (meaning all nodes). Set this to a smaller float val between 0-1 to return that sub-proportion of nodes, prioritizing, high-degree ones. For example, a value of 0.1 would return the top 10% highest degree nodes in the output overlay only.
3. down_factor... 
    * Temporarily downsamples the image to speed up overlay creation. Downsampling is done in all three dimensions by the inputed factor.

* Press 'Get Degrees' to run the method with the desired parameters. The output data will be used to create a new table in the tabulated data widget. The overlay will go into the Overlay 2 channel.

'Analyze -> Data/Overlays -> Get Hub Information'
--------------------------------------
* This method can be used to extract information about hub nodes, which are the nodes that are the fewest degrees of seperation from any other node. 

Parameter Explanations
~~~~~~~~~~~~~~~
* This method only has two parameters.

1. Make Overlay.
    * If enabled, this method will create an overlay isolating the hub nodes.
2. 'Proportion of most connected hubs to keep...'
    * A 0-1 float val that tells the program how many 'nodes' you want back in the output. For example, 0.10 would return the top 10% nodes with the fewest degrees of separation. 1 would just return all the nodes.

* Press 'Get hubs' to run the method with the desired parameters. The output data will be used to create a new table in the tabulated data widget. The overlay will go into the Overlay 2 channel.
* Note that the hubs are considered independently for each seperate, distinct network component. Additionally, components that have too few nodes will not return any hubs if the upper proportion threshold is particularly small.

'Analyze -> Data/Overlays -> Get Mother Nodes'
--------------------------------------
* This method can be used to extract information about 'mother nodes', which are we define as those nodes that contain connections between one community and another.
* This method would be used to identify what nodes enable interaction between seperate communities.

Parameter Explanations
~~~~~~~~~~~~~~~
* This method only has one parameter.

1. Make Overlay.
    * If enabled, this method will create an overlay isolating the mother nodes.

* Press 'Get Mothers' to run the method with the desired parameters. The output data will be used to create a new table in the tabulated data widget. The overlay will go into the Overlay 1 channel.

'Analyze -> Data/Overlays -> Code Communities'
--------------------------------------
* This method can be used to generate an overlay that shows what nodes belong to which community.

Parameter Explanations
~~~~~~~~~~~~~~~
* This method only has two parameters.

1. down_factor
    * Temporarily downsamples the image to speed up overlay creation. Downsampling is done in all three dimensions by the inputed factor. This is particularly useful for the color overlay.
2. Execution Mode:
    * This dropdown menu has two options:
        1. 'Color Coded' - Create an RGB color overlay where each node is colored according to its community. This overlay is great for easily visualizing communities.
        2. 'Grayscale Coded' - Create a grayscale overlay where each node is labeled by the community number it was assigned in the node_communities parameter. The purpose of this overlay is to create an image where nodes can then be thresholded by their community, for more specific analysis.

* Press 'Community Code' to run the method with the desired parameters. The overlay will go into the Overlay 2 channel. Additionally, a legend displaying what label belongs to which community will be placed into the tabulated data widget.

'Analyze -> Data/Overlays -> Code Identities'
--------------------------------------
* This method can be used to generate an overlay that shows what nodes belong to which identity.

Parameter Explanations
~~~~~~~~~~~~~~~
* This method only has two parameters.

1. down_factor
    * Temporarily downsamples the image to speed up overlay creation. Downsampling is done in all three dimensions by the inputed factor. This is particularly useful for the color overlay.
2. Execution Mode:
    * This dropdown menu has two options:
        1. 'Color Coded' - Create an RGB color overlay where each node is colored according to its identity. This overlay is great for easily visualizing identities.
        2. 'Grayscale Coded' - Create a grayscale overlay where each node is labeled by numerical identities (with the number corresponding each to one of the identity subtypes). The purpose of this overlay is to create an image where nodes can then be thresholded by their identity, for more specific analysis.

* Press 'Identity Code' to run the method with the desired parameters. The overlay will go into the Overlay 2 channel. Additionally, a legend displaying what label belongs to which identity will be placed into the tabulated data widget.

* The last submenu is 'Randomize', and is used to generate random variants of data.

'Analyze -> Data/Overlays -> Centroid UMAP'
--------------------------------------
* This method will create a UMAP, clustering nodes based on the similarity of their centroids. In short, it lets you easily eyeball what sort of things are next to each other. Its uses are more for 3D data, as 2D data can give a similar impression just by looking at the image.
* If node_identities exist, the nodes will all be colored based on their identity. Unassigned nodes will get called 'Unknown' in such a case.
* This method does not have any parameters. Simply run it to show the UMAP.

'Analyze -> Randomize -> Generate Equivalent Random Network'
-----------------------------------------
* This method allows us to generate a random network with an equivalent number of edges and nodes as the current network.
* The purpose of this method is a quick way to compare our network to a similar random one, which can be used to demonstrate presence of non-randomness, for example.   
* The only parameter is 'weighted'. If selected, edges in the random network will be allowed to stack into weighted edges.
    * Note if my network is weighted, weights are included in total edge counts for the purpose of this method, so three nodes with one edge of weight one and one edge of weight two will allow three connections to be made in the corresponding random network.
    * The weighted param just tells the random network whether its allowed to use these total edges to make weighted edges (a weighted edge of 2 would *cost* the random network 2 of its available edges, so to speak).
    * The weighted param does not tell the random network to ignore weights in the original network. To do that, first de-weight the network with 'Process -> Modify Network'.
* Press 'Generate Random Network' to place the random network in the 'Selection' network table. From here, it can be right clicked to either save it or to swap it into the active network.
    * Note that swapping the random network to active runs the risk of overriding the old active network if a new selection is made, so be sure to save it first.

'Analyze -> Randomize -> Scramble Nodes (Centroids)'
-------------------------------------------------------------
* This method allows us to randomize our node locations, for the purposes of comparing our dataset to a random one.
* This method uses our node centroids and randomizes the centroids themselves - 3D node objects are not included for this purpose.
* Selecting this option will display a window with a single parameter, 'Mode'. Its dropdown menu includes the following option.
    1. Anywhere - The nodes can go anywhere in the image bounds.
    2. Within Dimensional Bounds of Nodes - The nodes can go anywhere within the min/max boundaries of the current nodes (In the bounding box).
    3. Within Masked Bounds of Edges - The nodes can go anywhere the edge channel is non-zero.
    4. Within Masked Bounds of Overlay1 - The nodes can go anywhere the Overlay1 channel is non-zero.
    5. Within Masked Bounds of Overlay2 - The nodes can go anywhere the Overlay2 channel is non-zero.

* If a nodes channel image exists, it will be overrided by a equivalently-sized image.
* If a nodes channel image does not exist, no new image will be loaded and only the centroids will be randomized.
    * These centroids will be randomized within the bounds of any other available image channel. If there are none, they will use the min/max bounds of the current centroids.
* The purpose of params 3-5 is to allow creation of arbitrary boundary regions, for example by dilating data of interest, to allow the nodes to populate.

Next Steps
---------
This concludes the explanations of the analyze functions. Next, proceed to :doc:`process_menu` for information on the process menu functions.