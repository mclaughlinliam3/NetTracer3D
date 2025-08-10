.. _proximity:

==========
Proximity Networks - Ideal for Cellular Data
==========



Generating a Network Based on Proximity
--------------
We will go over another simple example of creating networks. This one is even easier, as it only requires nodes to function, and simply groups nodes as connected pairs based on their distance to each other.
Open a new instance of NetTracer3D and once more load in the binary segmentation of the nodes that was created above.
First, use 'Process -> Image -> Label Objects' to assign each binary object a unique label.
Next, select 'Process -> Calculate -> Calculate Proximity Network'.

.. image:: _static/proximity_menu.png
   :width: 800px
   :alt: Proximity Network Menu
*Here we can see the menu to generate proximity networks. The search distance here is set to 300, which means nodes will look 300 pixels out for connections (although this will correspond to your scalings). Note there are two options available for searching, shown in the carrot dropdown next to 'Execution Mode'. The first option searches from centroids and works quite well with big data as the data structure is far simpler. The second option searches from object borders and may be slower on large images by comparison. In this case, I use the second option since these objects are heterogenously sized. For more information on using this algorithm, see* :ref:`proximity_network` 

And after algorithm execution:

.. image:: _static/proximity.png
   :width: 800px
   :alt: Proximity Network

.. image:: _static/proximity2.png
   :width: 800px
   :alt: Proximity Network 2


Proximity networks are a generic way to group together objects in 3D space and are ideal, for example, for grouping together cellular neighborhoods.
One use for such cellular neighborhoods is grouping them into communities and analyzing their composition!

Using Proximity Networks for Advanced Cellular Neighborhood Analysis - Example Analysis of CODEX sample
-----------------------------------------------------------------------

Proximity networks generally make the best use of the 'node identities' property, which lets us eventually lasso together groups of nodes into neighborhoods that share local expression of similar cell types.
Node identities can represent anything about your node (ie what it is, its qualities, etc). They can be assigned by loading the node identities directly from a spreadsheet (see :doc:`excel_helper`), however we can also pull them directly from images.
One useful setting for this is multiplexing (where a tissue is imaged over and over across a plethora of markers). I will use the following CODEX dataset as an example, obtained from this publication: Canela, V.H., Bowen, W.S., Ferreira, R.M. et al. A spatially anchored transcriptomic atlas of the human kidney papilla identifies significant immune injury in patients with stone disease. Nat Commun 14, 4140 (2023). https://doi.org/10.1038/s41467-023-38975-8
This is an image of a renal papilla. Here it is stained with DAPI (Nuclei).

.. image:: _static/renal_papilla.png
   :width: 800px
   :alt: papilla


* There are a few options for getting multiple channels into the same set of nodes. The first is to merge the nodes directly with 'File -> Images -> Node Identities -> Merge Labeled Images into Nodes'. If you have segmentations of seperate objects already (ie different cells or different FTUs), this option can be used to pull them into NetTracer3D as different identities. However, this doesn't work when the nodes in each channel are on top of each other (ie the same cell with a different marker), or if the task of segmenting each is too burdensome.
* Because of this, we will use the second option (outside importing the nodes with identities from a spreadsheet), which 'File -> Images -> Node Identities -> Assign Node Identities From Overlap With Other Images', which is designed for CODEX-type images specifically.
* This option uses a primary segmentation of all nodes (for example, from a DAPI stain), and assigns each node an identity based on what labeled markers from the other channels that the node overlaps with.
* To make use of this, we first need to obtain an initial set of segmented nodes to use as 'anchors'. This will typically be done by segmenting DAPI channels.
* NetTracer3D offers a binary ML-segmenter. If we used it to segment our cells (like we did in the previous example), we could label them afterwards with direct labeling if they don't overlap. Should they overlap, we can use binary watershedding. A third option is to just segment out the background of the image with the intensity thresholder and then use gray-watershedding to split the foreground into labeled nodes. (See :doc:`process_menu` for explanations on each of these if interested).
* While those options are good for certain types of data, or for quick results, there are already great tools for segmenting cells with high specificity. One option is Cellpose3: Stringer, C., Pachitariu, M. Cellpose3: one-click image restoration for improved cellular segmentation. Nat Methods 22, 592–599 (2025). https://doi.org/10.1038/s41592-025-02595-5
* Cellpose3 can be bundled in the NetTracer3D package if the user wants, although it is somewhat mandatory to use a GPU with. For the most accurate results though, I recommend segmenting clustered nuclei with a user-trained model in Cellpose3, like shown here:

.. image:: _static/cellpose.png
   :width: 800px
   :alt: cellpose
*Please see https://cellpose.readthedocs.io/en/latest/ for specific instructions on how to install/use cellpose*

.. image:: _static/start.png
   :width: 800px
   :alt: start
*Here are the DAPI segmented nuclei (green borders) overtop the original DAPI channel in NetTracer3D. I am selecting one in yellow*


* Regardless of how we do it, once we obtain our segmented DAPI nuclei we want to load them into the nodes channel.
* Here, we have a few options on how to get the rest of the desired channels into the node identities.
* One option that's both fast and robust to error is 'File -> Images -> Node Identities -> Assign Node Identities From Overlap With Other Images', then selecting 'Manual' for the binarization strategy.
* To use this, you will first need to place all the channels of interest into a folder, arranged as separate .tif/.tiff images (ImageJ can be used to easily split up your channels if they are saved in a stack). For this mode, the channels should be raw instead of segmented.
* Next, use the following options and select your folder like so:

.. image:: _static/chooseone.png
   :width: 800px
   :alt: chooseone
*The step-out distance parameter can be used to optionally make each cell consider a wider area, such as to account for staining beyond the nuclei*

* NetTracer3D will then do some calculation - it will evaluate each cell for the average intensity it expresses in each channel.
* Once its done, you will get prompted to do manual thresholding on each image, telling NetTracer3D which subset of cells you would like to assign that identity based on their relative intensities in that channel. For example, here I am doing CD68 (stains macrophages):

.. image:: _static/monocytes.png
   :width: 800px
   :alt: monocytes
*During the thresholding steps, I am able to interact with NetTracer3D's image canvas. My nodes/cells remain in the Nodes channel. Each channel being thresholded will be temporarily rendered in the 'Overlay1' channel. I use the brightness/contrast controls to make them more visible, then zoom up close. I use the thresholder on the right to select the cell nuclei that seem like they truly overlap with the channel data, highlighted in yellow. This repeats for all channels in the folder.*

* After all channels have been thresholded, the nodes will be assigned identities, as shown in the upper right table. If this manual assignment was used, if the option to generate the UMAP was selected at the start, I will also get this UMAP comparing the Z-score for each marker of the cells I pulled out to assign identities.

.. image:: _static/UMAP.png
   :width: 800px
   :alt: UMAP
*Note that the UMAP adds an extra processing step but it can be a decent option for validation. If your cells are overlapping where they shouldn't, either the thresholding step or the image itself was poor.*

* This method of identity assignment can be quite useful to shield against phenotyping errors, since the mean intensity of each node is considered when assigning cells, with the user-in-the-loop, rather than just auto-assigning them based on overlap with the channel's segmented foreground.
* However the auto-assignment option also exists, and I am going to be using it from here on out anyway, since I happened to already have segmented several channels of interest.
* To use the auto-assignment, choose the following option:

.. image:: _static/choosetwo.png
   :width: 800px
   :alt: choosetwo
*Select the folder containing your binary segmented or to-be-auto-segmented channels*

* The auto-assignment compares the nodes to a foreground-binarized version of each channel, and simply assigns them to that identity based on rote overlap.
* Each channel can be already binarized (ie I segmented it myself first), but if it's not binary, NetTracer3D will detect this and auto-binarize it with Otsu's algorithm, which basically tries to find the larger intensity peak in the intensity histogram. Note that this auto-binarization only really works if the SNR is pretty good. If you think there's a chance it won't work, I highly recommend either manually segmenting all channels, or at least checking what Otsu's binarization does to that channel (Can be done in NetTracer3D with 'Process -> Image -> Binarize').

.. image:: _static/segment_cd31.png
   :width: 800px
   :alt: segment_cd31
*Here, I am using NetTracer3D's ML segmenter to segment CD31, a stain for blood vessels*

* We may be interested in reassigning nodes with multiple identities to their own class (this is called 'phenotyping'). Since its common for some cell types to co-express markers, it can be a good way to group them into more specific categories (ie T-cell, instead of lymphocyte).
* Or, we might also have already assigned nodes some group of identities in a program like QuPath. Whatever the case, we will want to get the information with the node identities in a spreadsheet (ie save the node identities spreadsheet we just got by right clicking the upper right table). We can then use 'File -> Load -> Load From Excel Helper' to reassign the identities and pull them back into NetTracer3D:

.. image:: _static/excel_helper.png
   :width: 800px
   :alt: excel_helper

.. image:: _static/spreadsheet_loader.png
   :width: 800px
   :alt: spreadsheet_loader
*Here we are telling NetTracer3D to reassign anything containing the terms 'Ki-67+' and 'Vimentin+' to the identity 'Proliferating Cells', just as an example.*

.. image:: _static/spreadsheet_loader_2.png
   :width: 800px
   :alt: spreadsheet_loader2
*We press 'Preview Classification', which assigns all the combinations of identities containing the string classifications we setup to their new ID. If we then press the green 'Export' button, this data will be sent into NetTracer3D's main window.* See :doc:`excel_helper` for more information on using this tool.

* Note if we really don't want to have multiple node identities for a node and also don't want to bother with the excel loader to reassign identities, we can just use 'Process -> Modify Network/Properties', then select 'Force Any Multiple IDs to Pick a Single Random ID' to force any multiple-ID node to randomly choose to be one of their sub-identities.
* Here is what my group of cells look like with all cells forced to take on a single identity (note you don't have to do this for most of the analytical functions, but the visualization ones work better with fewer groups of identities).

.. image:: _static/cells_mapped.png
   :width: 800px
   :alt: cells_mapped
*You can make this overlay with 'Analyze -> Data/Overlays -> Code Identities'

* Now once more, we calculate a proximity network. I don't use this, but a cool feature to note is "Create Networks only from a specific node identity" can be used to make connections only start from a certain identity, for highly specified analysis.

.. image:: _static/prox.png
   :width: 800px
   :alt: prox
*I am going to use centroids to make this proximity network, which is acceptable when my objects are circuloid/cuboid. Centroids can be obtained with 'Process -> Calculate Network -> Calculate Centroids', although if they don't exist and you run this, NetTracer3D will prompt you to calculate them. In this instance, I tell the Calculate Centroids Window to 'Skip ID-less centroids', which is a parameter that makes it exclude any centroids for Nodes that were not assigned an identity, and likewise those nodes will be excluded from the network. If I am interested, though, in having these ID-less nodes in the network, ie to represent the tissue architecture, this option does not need to be used. Finally, 'Process -> Modify Network/Properties' can be used to kick out centroids that don't have identities if I change my mind.*

* Here is an example of what the network itself looks like. Note that since I opted to not use centroids that don't have an identity, any nodes that did not overlap with a channel marker are not participating in this network at the moment.

.. image:: _static/network.png
   :width: 800px
   :alt: network

* To cluster the nodes based on their network participation, I use 'Analyze -> Network -> Community Partition...' to place the nodes into communities.

.. image:: _static/communities.png
   :width: 800px
   :alt: communities
*My nodes, having been assigned a community position, based on their network involvement, via the 'Louvain algorithm'. For the proximity network, this offers a way to group nodes based on their spatial relationship. This colored overlay can then be obtained with 'Analyze -> Data/Overlays -> Code Communities'.

* I usually assign communities based on network. However, an alternative option to assign communities just based on splitting the image into cuboidal proximity cells of arbitrary size is available through 'Analyze -> Network -> Create Communities based on Cuboidal Proximity Cells'. These communities would have nothing to do with network structure, but may be more useful in partitioning the image if, for example, the cell layout is very dense and chaotic.
* However, I just use the standard network partition. Now I can do analysis to see what the community compositions tend to look like. But I may have hundreds of communities in a big network. So in that case, an additional option is to use the communities to further group the nodes into neighborhoods. Neighborhoods are formed by evaluating the proportion of each node identity in a community, then grouping together communities that show similar compositions, which may represent areas of similar tissue or disease process, for example.
* If you make neighborhoods, be sure to backup your communities first by saving them, if you want (although they are usually quick to regenerate). This is because neighborhoods actually take the place of the community property in the active session. The only exception is the method that creates neighborhoods in the first place (Analyze -> Network -> Convert Network Communities into Neighborhoods), as well as 'Analyze -> Network -> Identity Makeup of Communities', which both always reference the old communities even if neighborhoods have been created (unless, of course, the neighborhoods themselves are saved and reloaded directly into the communities property).

.. image:: _static/hoods.png
   :width: 800px
   :alt: hoods
*Here I am making neighborhoods. I can tell the window directly how many neighborhoods I want to be formed, although if this is empty, it will just try to calclulate a good number. I also assign a min-community size, so that any community with fewer than 5 cells will not be included*

* Once we run the above method, it will generate our neighborhoods. One important thing to note is that any communities I deemed to small will get assigned to Neighborhood 0 (the outlier neighborhood). All other Neighborhoods are organized by their size, so Neighborhood 1 is the largest, Neighborhood 5 is mid-sized, while Neighborhood 10 is the smallest and may represent some anomaly.
* Running this method also provide us some heatmaps telling us how much each neighborhood is expressing each label. The one I find the most useful is this one:

.. image:: _static/hood_guide.png
   :width: 800px
   :alt: hood_guide
*This graph tells us the log-normalized overexpression/underexpression of each label in a neighborhood. In short, any value greater than one (red) is over-represented in that community, while any value elss than one (blue) is under-represented in that community. Also rcall that Neighborhood 0 always contains discarded nodes only, while Neighborhoods are largest at 1 and get smaller from there. If I wanted, for whatever reason, to see this for the communities themselves, just tell the previous window to assign a neighborhood number equal to the number of available communities.*

* I can run the 'Code Communities' method again to visualize the neighborhoods directly on the image:

.. image:: _static/hood_colors.png
   :width: 800px
   :alt: hood_colors

* Something else I can do is 'Analyze -> Network -> Identity Makeup of Communities' to see the relative compositions of each community and/or neighborhood. Note that this runs on communities in the current session regardless of whether neighborhoods have been assigned. But it can also generate us a UMAP showing how similar the communities are. If I have already made neighborhoods via above, I can also see exactly how the communities got grouped if I tell the UMAP to label itself by neighborhoods, like below:

.. image:: _static/umap_params.png
   :width: 400px
   :alt: umap_params

.. image:: _static/UMAP2.png
   :width: 800px
   :alt: UMAP2

* This has been a fairly long analysis explanation but don't forget we can also analyze network composition with 'Analyze -> Stats -> Network Statistics Histograms', and that these histograms can be used to threshold the nodes. 
* An important mention is that these histograms may be slow to generate for very complex networks. If over-connectivity is an issue for proximity networks, there is a parameter during their generation that can be used to actually limit the number of connections each node is allowed to make to their nearest 'n' neighbors! This can be important to use for very knotted networks (but it only works for the proximity networks made from centroids)! (Also worth mentioning that this means we can generate networks based on nearest neighbors as opposed to distance if we just tell NetTracer3D to limit itself to n nearest neighbors, but with a very large search distance).
* Finally, the histogram algorithms may refuse to run on disconnected networks, which many of these cell graphs can frequently yield. One option then is to first get the subgraph. You can click on any node in the image, then right click and choose 'Show Connected Component'. The corresponding subgraph will populate the 'Selection' table in the lower left. Right click the selection table to save it, then load it back into NetTracer3D with load network (or right click it and choose swap with network table, but beware that this risks losing the wider network information if it's not backed up). Once we've put the subgraph in the main network, the network analysis algorithms will run on the subgraph instead!

Next Steps
---------
Once you have a hang on generating the default network types, proceed to the :doc:`branches` to learn about using NetTracer3D to label branches of objects and create branch networks.
