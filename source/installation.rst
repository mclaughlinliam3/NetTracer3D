.. _installation:

============
Installation
============

System Requirements
------------------

Before installing NetTracer3D, ensure your system meets the following requirements:

* Operating System: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)
* CPU: Multi-core processor (4+ cores recommended)
* RAM: Minimum 8GB (16GB+ recommended for larger images. For very large images such as lightsheet data, you would even want a workstation with 128+, for example)
* GPU: NVIDIA dedicated GPU (Optional)
* Storage: 1GB for installation, additional space for captured data
* Python: 3.11

Installing NetTracer3D
----------------------

Using pip
~~~~~~~~~

The easiest way to install NetTracer3D is using pip:

.. code-block:: bash

    pip install nettracer3d

This will install NetTracer3D and all its core dependencies.

Optional Packages
~~~~~~~~~~~~~~~~~~
I recommend including Napari (Chi-Li Chiu, Nathan Clack, the napari community, napari: a Python Multi-Dimensional Image Viewer Platform for the Research Community, Microscopy and Microanalysis, Volume 28, Issue S1, 1 August 2022, Pages 1576–1577, https://doi.org/10.1017/S1431927622006328) in the download as well, which allows NetTracer3D to use 3D displays. The standard package only comes with its native 2D slice display window. 
If Napari is present, all 3D images and overlays from NetTracer3D can be easily displayed in 3D with a click of a button. To package with Napari, use this install command instead: 

.. code-block:: bash

    pip install nettracer3d[viz]

Additionally, for easy access to high-quality cell segmentation, as of version 0.8.2, NetTracer3D can be optionally packaged with Cellpose3. (Stringer, C., Pachitariu, M. Cellpose3: one-click image restoration for improved cellular segmentation. Nat Methods 22, 592–599 (2025). https://doi.org/10.1038/s41592-025-02595-5)
Cellpose3 is not involved with the rest of the program in any way, although its GUI can be opened from NetTracer3D's GUI, provided both are installed in the same environment. It is a top-tier cell segmenter which can assist in the production of cell networks.
To include Cellpose3 in the install, use this command:

.. code-block:: bash

    pip install nettracer3d[cellpose]

Alternatively, both Napari and Cellpose can be included in the package with this command: (Or they can be independently installed with pip from the base package env)

.. code-block:: bash

    pip install nettracer3d[all]

GPU
~~~~~~~~~~~~~~~~~~
NetTracer3D is mostly CPU-bound, but a few functions can optionally use the GPU. To install optional GPU functionalities, first set up a CUDA toolkit that runs with the GPU on your machine. This requires an NVIDIA GPU. Then, find your GPUs compatible CUDA toolkit and install it with the auto-installer from the NVIDIA website: https://developer.nvidia.com/cuda-toolkit

With a CUDA toolkit installed, use:

.. code-block:: bash

    pip install nettracer3d[CUDA11] #If your CUDA toolkit is version 11
    pip install nettracer3d[CUDA12] #If your CUDA toolkit is version 12
    pip install nettracer3d[cupy] #For the generic cupy library (The above two are usually the ones you want)

Or if you've already installed the NetTracer3D base package and want to get just the GPU associated packages:

.. code-block:: bash

    pip install cupy-cuda11x #If your CUDA toolkit is version 11
    pip install cupy-cuda12x #If your CUDA toolkit is version 12
    pip install cupy #For the generic cupy library (The above two are usually the ones you want)

While not related to NetTracer3D, if you want to use Cellpose3 (for which GPU-usage is somewhat obligatory) to help segment cells for any networks, you will also want to install pytorch here: https://pytorch.org/. Use the pytorch build menu on this webpage to find a pip install command that is compatible with Python and your CUDA version.



Verifying Installation
---------------------

To verify that NetTracer3D has been installed correctly, run:

.. code-block:: bash

    nettracer3d --version

You should see the current version number displayed.

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Conflicting Dependencies**

If you encounter errors about conflicting dependencies, download anaconda and try installing them in a new anaconda env:

.. code-block:: bash

    conda create --name nettracer3d python=3.11

    conda activate nettracer3d

    pip install nettracer3d

The highest probability is that a version of a package you are running is not compatible with numpy version 2 and above. Please try upgrading/downgrading the incompatible packages as needed.
This is generally done along the lines of:

.. code-block:: bash
    
    pip install 'numpy<=2.2' # If a version of numpy beyond 2.2 was causing a conflict, for example.

Getting Help
~~~~~~~~~~~

If you continue to experience installation issues:

* Check the :doc:`troubleshooting` guide
* Email me at liamm@wustl.edu

Next Steps
---------

After installation, proceed to the :doc:`quickstart` guide to begin using NetTracer3D.