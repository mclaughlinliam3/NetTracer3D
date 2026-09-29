.. _properties:

================================
Properties of a Network3D Object
================================

The Network3D object is how NetTracer3D groups together the data of an ongoing
session. Its properties are saved and loaded with **File → Save (As) Network 3D
Object** and the equivalent load option, and many of NetTracer3D's methods
reference one or more of them to function correctly.

.. contents:: On this page
   :local:
   :depth: 2


Main Properties
---------------

1. **Nodes** — the image in the nodes channel, representing objects to be grouped
   in a network.
2. **Edges** — the image in the edges channel, representing objects used to
   connect nodes together.
3. **Overlay 1** — the image in the overlay1 channel.
4. **Overlay 2** — the image in the overlay2 channel.
5. **Network** — the network itself.
6. **Node Centroids** — the [Z, Y, X] centroid of each node.
7. **Edge Centroids** — the [Z, Y, X] centroid of each edge.
8. **Node Communities** — the communities that nodes in the network belong to.
9. **Node Identities** — the assigned identities of nodes in the network.
10. **xy_scale** — the real dimension per pixel of the 2D x/y plane (for example,
    5 microns per pixel). This value is always the same for both x and y;
    differentially scaled x and y dimensions are not currently supported.
11. **z_scale** — the real dimension per voxel-depth of the 3D z plane.

Running **Process → Calculate Connectivity Network** acquires an additional
hidden property:

12. **Search Region** — an image of the nodes after expansion by the desired
    parameter to search for edge connections.

This property occupies RAM. It is saved alongside the Network3D object if it
exists, but is not loaded back in with one. Its purpose is to allow the search
region to be saved and then loaded directly into the nodes channel, so that
connectivity networks can be computed under new parameters while skipping the
node search step entirely — the slow step of the process.

Most properties can be purged from RAM using **Image → Properties**. **The
xy_scale and z_scale values should also be assigned there.**


Structures Required for Properties Stored in CSVs
-------------------------------------------------

NetTracer3D organises several properties into CSV spreadsheets with a specific
data layout, which it expects to find when loading those properties back in.
Users may wish to supply these properties from elsewhere — both ``.csv`` and
``.xlsx`` can be loaded — for example to assign nodes to particular centroids or
identities by hand in Microsoft Excel, or to batch-organise datasets for import
using a library such as pandas.

.. note::

   Node identities are also saved to a ``.json``, which takes priority over the
   ``.csv`` when loading. To load the ``.csv`` instead, delete the ``.json``.

.. note::

   ``.xlsx`` files have a row limit, so saving as ``.csv`` is generally
   recommended, especially for large property lists.

The required structures are given below. Be sure to use the correct headers.

1. network
~~~~~~~~~~

Adjacent nodes in the same row are connected, and the edge to the right of a
node-pair in that row is the edge found to connect them. If labeled edges were
not used, this value is simply 0. The table below states that node 18 is paired
with node 20 via edge 175, and node 16 with node 20 via edge 176.

+------------+------------+-----------+
| Node A     | Node B     | Edge C    |
+============+============+===========+
| 18         | 20         | 175       |
+------------+------------+-----------+
| 16         | 20         | 176       |
+------------+------------+-----------+

Additional column sets should be read as their own rows — node 21 paired to node
22, connected by edge 177.

2. node_identities
~~~~~~~~~~~~~~~~~~

Node identities use a simple Node:Identity structure.

+--------+----------+
| NodeID | Identity |
+========+==========+
| 1      | [Value]  |
+--------+----------+
| 2      | [Value]  |
+--------+----------+
| 3      | [Value]  |
+--------+----------+

Node identities typically pair a node to a single identity value, but nodes may
also be assigned multiple identities. In that case the identity value is a list
of all corresponding identities, stored as a string in memory for compatibility
with other methods and converted back to a list in relevant functions using
``ast.literal_eval()``.

3. node_communities
~~~~~~~~~~~~~~~~~~~

This property shares the structure of node_identities.

+--------+-----------+
| NodeID | Community |
+========+===========+
| 1      | [Value]   |
+--------+-----------+
| 2      | [Value]   |
+--------+-----------+
| 3      | [Value]   |
+--------+-----------+

4. node_centroids
~~~~~~~~~~~~~~~~~

Node centroids are organised as Node:Zval:Yval:Xval. The Z, Y, X order reflects
the way numpy organises dimensions. Coordinates should generally be integers
greater than 0.

+---------+-------+-------+-------+
| Node ID | Z     | Y     | X     |
+=========+=======+=======+=======+
| 1       | [Val] | [Val] | [Val] |
+---------+-------+-------+-------+
| 2       | [Val] | [Val] | [Val] |
+---------+-------+-------+-------+
| 3       | [Val] | [Val] | [Val] |
+---------+-------+-------+-------+

5. edge_centroids
~~~~~~~~~~~~~~~~~

Edge centroids are identical to node centroids apart from the header.

+---------+-------+-------+-------+
| Edge ID | Z     | Y     | X     |
+=========+=======+=======+=======+
| 1       | [Val] | [Val] | [Val] |
+---------+-------+-------+-------+
| 2       | [Val] | [Val] | [Val] |
+---------+-------+-------+-------+
| 3       | [Val] | [Val] | [Val] |
+---------+-------+-------+-------+


Temporary Properties
--------------------

The following properties are maintained for the duration of an active session but
are neither saved nor loaded with Network3D objects:

1. Object volumes
2. Object radii

While these properties exist in the active session, they are displayed in the
**Info on Object** table when their corresponding objects are clicked.


Next Steps
----------

Next, read :doc:`excel_helper` for guidance on the excel loader GUI, a tool for
loading data from excel files.
