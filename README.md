NetTracer3D is a python package developed for both 2D and 3D analysis of microscopic images in the .tif file format. It supports generation of 3D networks showing the relationships between objects (or nodes) in three dimensional space, either based on their own proximity or connectivity via connecting objects such as nerves or blood vessels. In addition to these functionalities are several advanced 3D data processing algorithms, such as labeling of branched structures or abstraction of branched structures into networks. Note that nettracer3d uses segmented data, which can be segmented from other softwares such as ImageJ and imported into NetTracer3D, although it does offer its own segmentation via intensity and volumetric thresholding, or random forest machine learning segmentation. NetTracer3D currently has a fully functional GUI. To use the GUI, after installing the nettracer3d package via pip, enter the command 'nettracer3d' in your command prompt:

--- Documentation ---

Please see: https://nettracer3d.readthedocs.io/en/latest/

--- Installation ---

To install nettracer3d, simply install Python and use this command in your command terminal:

pip install nettracer3d

I recommend installing the program as an Anaconda package to ensure its modules are work together on your specific system:
(Install anaconda at the link below, set up a new python env for nettracer3d, then use the same pip command).

https://www.anaconda.com/download?utm_source=anacondadocs&utm_medium=documentation&utm_campaign=download&utm_content=installwindows

nettracer3d mostly utilizes the CPU for processing and visualization, although it does have a few GPU-aided options. If you would like to use the GPU for these, you will need an NVIDIA GPU and a corresponding CUDA toolkit which can be installed here:
https://developer.nvidia.com/cuda-toolkit

To install nettracer3d with associated GPU-supporting packages, please use:

If your CUDA toolkit is version 11: pip install nettracer3d[CUDA11]
If your CUDA toolkit is version 12: pip install nettracer3d[CUDA12]
If you just want the entire cupy library: pip install nettracer3d[cupy]


This gui is built from the PyQt6 package and therefore may not function on dockers or virtual envs that are unable to support PyQt6 displays.


For a (slightly outdated) video tutorial on using the GUI: https://www.youtube.com/watch?v=cRatn5VTWDY

NetTracer3D is free to use/fork for academic/nonprofit use so long as citation is provided, and is available for commercial use at a fee (see license file for information).

NetTracer3D was developed by Liam McLaughlin while working under Dr. Sanjay Jain at Washington University School of Medicine.

-- Version 0.6.9 updates --

1. Adjusted all distance transform-based dilation/radius calculating methods to simply use the already supported scipy.ndimage.distance_transform_edt() sampling parameter to account for differentially scaled axis (previously the image was being resampled but now it no longer will need to do that).

2. Added new right click option to extract highlighted regions and implant their data onto a separate image or into a new empty image.

3. General bug fixes and improvements.

4. Now specifies python 3.11. 
