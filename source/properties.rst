.. _properties:

============
Properties of a Network3D Object
============

Main Properties
------------------------------
The Network3D Object is how NetTracer3D groups together the data in an ongoing session. These properties will be saved/loaded when using 'File -> Save (As) Network 3D Object', or the equivalent load.
Many of NetTracer3D's methods may need to reference one or more of these properties to function correctly.

* These properties are as follows:
    1. Nodes - the image in the nodes channel that represents objects to be grouped in a network.
    2. Edges - the image in the edges channel that represents objects to use to connect nodes together.
    3. Overlay 1 - the image in the overlay1 channel
    4. Overlay 2 - the image in the overlay2 channel
    5. Network - The network itself.
    6. Node Centroids - The [Z,Y,X] centroid of each node.
    7. Edge Centroids - The [Z, Y, X] centroid of each edge.
    8. Node Communities - The communities that nodes in the network belong to.
    9. Node identities - The assigned identities of nodes in the network.
    10. xy_scale - The real-dimension per pixel of the 2D x/y plane (ie, 5 microns per pixel). This value will always be the same for both x and y. NetTracer3D does not currently support differentially scaled x and y dimensions.
    11. z_scale - The real dimension per voxel-depth of the 3D z plane.
* Whenever 'Process -> Calculate Connectivity Network' is run, NetTracer3D will aquire an additional hidden property.
    12. Search Region - An image of the nodes after they have been expanded by the desired parameter to search for edge connections.
    * Note this property does occupy RAM. It will be saved alongside the 'Network 3D Object' if it exists, but it will not be loaded in when loading 'Network 3D Object'.
    * The main reason this exists is to allow it to be saved, after which the user can simply load the Search Region itself into the nodes channel, so that they may compute Connectivity Networks under new parameters, while being able to skip the node_search step completely (which is the slow step).
* Note that the majority of the properties can be purged from RAM by using 'Image -> Properties'. **The xy_scale and z_scale values should also be assigned here.**

Necessary Structures to Properties stored in CSVs
---------------------------------------
* When NetTracer3D saves its properties, it organizes several into csv spreadsheets, containing specific data organization that it expects to find when loading the same properties back in.
* Some users may wish to load these properties in from elsewhere (.csv or .xlsx can be used to load). For example, if they want to manually assign nodes to certain centroids or identities by editing microsoft excel, or batch organizing datasets for import this way using something like the pandas module in python.
* These structures are as follows (make sure to use correct headers):

1. network property:

* Network tables are organized with this structure. Adjacent nodes in the same row are connected. The edge to the right of a node-pair in the same row is the edge that was found to connect them. If labeled edges were not used, this value is simply 0.
* For example, this table specifically is saying node 18 is paired to node 20, connected via edge 175, etc.

+------------+------------+-----------+
| Node 1A    | Node 1B    | Edge 1C   |
+============+============+===========+
| 18         | 20         | 175       |
+------------+------------+-----------+
| 16         | 20         | 176       |
+------------+------------+-----------+

* It may be worth mentioning that for very large networks that exceed the .csv column length limit, the rows begin to populate besides each other like this:

+------------+------------+-----------+------------+------------+-----------+
| Node 1A    | Node 1B    | Edge 1C   | Node 2A    | Node 2B    | Edge 2C   |
+============+============+===========+============+============+===========+
| 18         | 20         | 175       | 21         | 22         | 177       |
+------------+------------+-----------+------------+------------+-----------+
| 16         | 20         | 176       | 19         | 23         | 178       |
+------------+------------+-----------+------------+------------+-----------+

* These new columns should be thought of their own rows, as in node 21 is paired to node 22, connected by edge 177. Note that that once these additional columns hit the column length limit, another set of rows will populate to the right, but this time with the number '3' instead of the number '2'. This can continue as long as it needs to.

2. node_identities:

* Node identities are organized with a simple Node:Identity structure. Note that as of yet, these tables do not handle overflow in the .csv column length.

+--------+----------+
| NodeID | Identity |
+========+==========+
| 1      | [Value]  |
+--------+----------+
| 2      | [Value]  |
+--------+----------+
| 3      | [Value]  |
+--------+----------+

3. node_communities:

* This property has the same structure as node_identities, and similarly does not handle column overflow.

+--------+-----------+
| NodeID | Community |
+========+===========+
| 1      | [Value]   |
+--------+-----------+
| 2      | [Value]   |
+--------+-----------+
| 3      | [Value]   |
+--------+-----------+

4. node_centroids:

* Node centroids organize using Node:Zval:Yval:Xval. Again, they do not handle column overflow.
* Make sure to use Z, Y, X. This is because of the way numpy organizes dimensions.
* Coordinates should generally be ints greater than 0.

+---------+-------+-------+-------+
| Node ID | Z     | Y     | X     |
+=========+=======+=======+=======+
| 1       | [Val] | [Val] | [Val] |
+---------+-------+-------+-------+
| 2       | [Val] | [Val] | [Val] |
+---------+-------+-------+-------+
| 3       | [Val] | [Val] | [Val] |
+---------+-------+-------+-------+

4. edge_centroids:

* Edge centroids are ostensibly the same as node centroids, albeit with a different header.

+---------+-------+-------+-------+
| Edge ID | Z     | Y     | X     |
+=========+=======+=======+=======+
| 1       | [Val] | [Val] | [Val] |
+---------+-------+-------+-------+
| 2       | [Val] | [Val] | [Val] |
+---------+-------+-------+-------+
| 3       | [Val] | [Val] | [Val] |
+---------+-------+-------+-------+

Temp Properties
------------------

* The following properties will be maintained for the duration of an active session but are neither saved nor loaded with 'Network 3D Objects':
    1. Object volumes.
    2. Object Radii.
    
    * While both of these properties exist in the active session, they will be displayed when their corresponding objects are clicked in the 'Info on Object' table.



Next Steps
---------
This concludes the tutorial section about using NetTracer3D. Although I covered many network-generating options in some detail, there are a plethora of other features and functions to learn about. The rest of this guide will go over all the algorithms and associated parameters within NetTracer3D in detail, in a more informative and less tutorial-oriented style. For questions about any particular function, please locate the section in the corresponding section guide to read more about it.