NetTracer3D is a python package developed for both 2D and 3D analysis of microscopic images in the .tif file format. It supports generation of 3D networks showing the relationships between objects (or nodes) in three dimensional space, either based on their own proximity or connectivity via connecting objects such as nerves or blood vessels. In addition to these functionalities are several advanced 3D data processing algorithms, such as labeling of branched structures or abstraction of branched structures into networks. Note that nettracer3d uses segmented data, which can be segmented from other softwares such as ImageJ and imported into NetTracer3D, although it does offer its own segmentation via intensity and volumetric thresholding, or random forest machine learning segmentation. NetTracer3D currently has a fully functional GUI. To use the GUI, after installing the nettracer3d package via pip, enter the command 'nettracer3d' in your command prompt:


--- Documentation ---

Please see: https://nettracer3d.readthedocs.io/en/latest/


--- Video Tutorial ---

Please see: https://www.youtube.com/watch?v=_4uDy0mzG94&list=PLsrhxiimzKJMZ3_gTWkfrcAdJQQobUhj7


--- Using the windows installer ---

If you are on windows, an installer will be available here: https://doi.org/10.5281/zenodo.17873800

Simply download and run the installer, then open the program in windows. You can also download the executable and its dependencies directly. Note that this version will be missing a few features compared to the Python package, namely GPU segmentation support. It will also not be updated as often.


--- Installing as a Python package ---

1. **Get Python and Pip on your path**: To install nettracer3d, first install Python version 3.12. (You can try other python versions also but I can't say which one's will cause dependency problems). Make sure the Python installation installs pip, and that both Python and pip are available on your PATH. I recommend installing Python using the installer which is available here. Make sure to check the option to 'add Python to PATH' when it appears: https://www.python.org/downloads/


2. **Base Package**: Next, use this command in your command terminal

	* pip install nettracer3d


3. **For 3D Displays**: Or if you also want Napari for 3D displays:

	* pip install nettracer3d[viz]


4. **Optional Performance Boost**: If you are trying to process large images, you may also want to include the 'edt' module in your package. This will allow parallelized CPU calculations for several of the search functions which can increase their speed by an order of magnitude or more depending on how many cores your CPU has. This can be a major benefit if you have a strong CPU and sufficient RAM. It requires an extra pre-installation step, thus is not included by default. You will also have to install the C++ build tools from windows. Please head to this link, then download and run the installer: https://visualstudio.microsoft.com/visual-cpp-build-tools/. In the menu of the installer, select the 'Desktop Development with C++' option, then proceed to download/install it using the installation menu. You will likely want to be using the Python distributed from the actual Python website and not the windows store (or elsewhere) or the edt module may not work properly. To bundle with edt use:

	* pip install nettracer3d[edt]


5. **Recommended full package**: Or if you want to just get both edt and napari at once. This will also give you the module to run leiden clustering:
	
	* pip install nettracer3d[rec]


6. Likewise, if you already installed the default version, you can add napari and/or edt with just:

	* pip install edt
	* pip install napari

   
--- Installing as a Python package in Anaconda---

I recommend installing the program as an Anaconda package to ensure its modules are work together on your specific system:
(Install anaconda at the link below, set up a new python env for nettracer3d, then use the same pip command).

https://www.anaconda.com/download?utm_source=anacondadocs&utm_medium=documentation&utm_campaign=download&utm_content=installwindows



--- Optional Packages ---

I recommend including Napari (Chi-Li Chiu, Nathan Clack, the napari community, napari: a Python Multi-Dimensional Image Viewer Platform for the Research Community, Microscopy and Microanalysis, Volume 28, Issue S1, 1 August 2022, Pages 1576–1577, https://doi.org/10.1017/S1431927622006328) in the download as well, which allows NetTracer3D to use 3D displays. The standard package only comes with its native 2D slice display window. 
If Napari is present, all 3D images and overlays from NetTracer3D can be easily displayed in 3D with a click of a button. To package with Napari, use this install command instead: 

    pip install nettracer3d[viz]

Additionally, for easy access to high-quality cell segmentation, as of version 0.8.2, NetTracer3D can be optionally packaged with Cellpose3. (Stringer, C., Pachitariu, M. Cellpose3: one-click image restoration for improved cellular segmentation. Nat Methods 22, 592–599 (2025). https://doi.org/10.1038/s41592-025-02595-5)
Cellpose3 is not involved with the rest of the program in any way, although its GUI can be opened from NetTracer3D's GUI, provided both are installed in the same environment. It is a top-tier cell segmenter which can assist in the production of cell networks.
To include Cellpose3 in the install, use this command:


    pip install nettracer3d[cellpose]

Alternatively, Napari, Cellpose, and edt can be included in the package with this command: (Or they can be independently installed with pip from the base package env)


    pip install nettracer3d[all]


--- GPU ---
NetTracer3D is mostly CPU-bound, but a few functions can optionally use the GPU. To install optional GPU functionalities, first set up a CUDA toolkit that runs with the GPU on your machine. This requires an NVIDIA GPU. Then, find your GPUs compatible CUDA toolkit and install it with the auto-installer from the NVIDIA website: https://developer.nvidia.com/cuda-toolkit

With a CUDA toolkit installed, use:

    pip install nettracer3d[CUDA11] #If your CUDA toolkit is version 11
    pip install nettracer3d[CUDA12] #If your CUDA toolkit is version 12
    pip install nettracer3d[cupy] #For the generic cupy library (The above two are usually the ones you want)

Or if you've already installed the NetTracer3D base package and want to get just the GPU associated packages:

    pip install cupy-cuda11x #If your CUDA toolkit is version 11
    pip install cupy-cuda12x #If your CUDA toolkit is version 12
    pip install cupy #For the generic cupy library (The above two are usually the ones you want)

While not related to NetTracer3D, if you want to use Cellpose3 (for which GPU-usage is somewhat obligatory) to help segment cells for any networks, you will also want to install pytorch here: https://pytorch.org/. Use the pytorch build menu on this webpage to find a pip install command that is compatible with Python and your CUDA version.


This gui is built from the PyQt6 package and therefore may not function on dockers or virtual envs that are unable to support PyQt6 displays.


NetTracer3D is freely available for academic and nonprofit use and can obtained from pip (pip install nettracer3d), provided that citation is included in any abstract, paper, or presentation utilizing NetTracer3D.

(The official paper to cite is coming soon)

NetTracer3D was developed by Liam McLaughlin while working under Dr. Sanjay Jain at Washington University School of Medicine.


