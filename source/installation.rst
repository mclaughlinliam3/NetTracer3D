.. _installation:

============
Installation
============

System Requirements
------------------

Before installing NetTracer3D, ensure your system meets the following requirements:

* Operating System: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)
* CPU: Multi-core processor (4+ cores recommended)
* RAM: Minimum 8GB (16GB+ recommended for larger images. For very large images such as lightsheet data, you would even want a workstation with 128+, for example, or downsample the data accordingly)
* GPU: NVIDIA dedicated GPU (Optional)
* Storage: ~7GB for installation (at least, that's how big the compiled version's dependencies get), additional space for captured data
* Python: 3.12

Installing NetTracer3D
----------------------

Using pip
~~~~~~~~~

The easiest way to install NetTracer3D is using pip. First, you will want to go to the Python website (https://www.python.org/). Download the Python installer for the latest version of Python. When you run it, make sure to include the option to also install Pip, and to add both Pip and Python to your path.

After installing Python and pip, you can verify (on windows at least) the installation by pressing the windows key + R. In the window that pops up, type 'cmd' to open the command terminal. In the command terminal, type 'where pip' or 'where python'.
If your installation of them worked properly, the window will tell you what folder your pip/python .exe resides in. If it cannot find them, it means they either did not get installed, or that they were not added to your PATH. 
You can try reinstalling them, if perhaps the wrong setting was selected, or you can manually add them to the environment variables (the PATH) from the windows control panel. (I am not a mac user so I cannot provide advice there).

Once you have Python and pip set up, open the command terminal once more (Windows + R, followed by typing 'cmd' in the window that appears). Type the following command.

.. code-block:: bash

    pip install nettracer3d


Or if you want the 3D visualization options to be included:

.. code-block:: bash

    pip install nettracer3d[viz]


This will install NetTracer3D and all its core dependencies. Then, if you want to run nettracer3d, open the command terminal like before and enter the following command:

.. code-block:: bash

    nettracer3d


Using Anaconda
~~~~~~~~~~~~~~~~~~~~~~

Anaconda is a useful program that manages Python packages for you and can help work out problems regarding conflicting package versions during installation. Installing NetTracer3D with Anaconda is similar to the above method, except it will be housed in a dedicated NetTracer3D anaconda environment rather than in your computer's Python packages.
You would first want to download the version of anaconda that is compatible with your operating system here: https://www.anaconda.com/download.

Run the installer they provide. Once it's finished, search 'anaconda prompt' in your taskbar and open the corresponding program. You will be taken to a command line window.
Next, you can run these commands:

.. code-block:: bash

    conda create --name nettracer3d python=3.12

    conda activate nettracer3d

    pip install nettracer3d[viz]

This should install the program in the 'nettracer3d' environment in conda. Then, whenever you wanted to run nettracer3d, you would first open the 'anaconda prompt', followed by entering these commands:

.. code-block:: bash

    conda activate nettracer3d

    nettracer3d

Using the compiled version
~~~~~~~~~~~~~~~~~~~~~~~~
If you want to avoid going the Python/pip route, you can download a compiled .exe of version 1.2.4 here: https://doi.org/10.5281/zenodo.17873800 

Unzip the folder, then double click the NetTracer3D executable to run the program. Note that this version will be missing a few features compared to the Python package, namely the 3D display, any GPU support, and the ability to print updates to the command window. It will also not be updated as often.


Optional Packages
~~~~~~~~~~~~~~~~~~
This was touched on above, but I recommend including Napari (Chi-Li Chiu, Nathan Clack, the napari community, napari: a Python Multi-Dimensional Image Viewer Platform for the Research Community, Microscopy and Microanalysis, Volume 28, Issue S1, 1 August 2022, Pages 1576–1577, https://doi.org/10.1017/S1431927622006328) in the download as well, which allows NetTracer3D to use 3D displays. The standard package only comes with its native 2D slice display window. 
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

    conda create --name nettracer3d python=3.12

    conda activate nettracer3d

    pip install nettracer3d

The highest probability is that a version of a package you are running is not compatible with numpy version 2 and above. Please try upgrading/downgrading the incompatible packages as needed. I would generally not downgrade numpy below version 2.0, as it is the main workhorse when it comes to image processing in Python, and the 2.0 release made a lot of nice optimizations.
This is generally done along the lines of:

.. code-block:: bash
    
    pip install 'numpy<=2.2' # If a version of numpy beyond 2.2 was causing a conflict, for example.


**Python Version Problems**

Another problem might be due to some of the packages that NetTracer3D depends on not having proper build protocols set up for newer Python releases. If you are attempting to install the package and find that it starts the installation but crashes part-way through, it might be a good idea to try downgrading your version of Python.
Generally speaking, the installation should work fine with Python version 3.12. However, issues may arise with later versions as of writing this (12/13/2025) specifically with the sklearn package.
To downgrade Python you can either uninstall it from your computer (on windows, from the add/remove programs menu), followed by reinstalling version 3.12. 
On anaconda, you can just make a new environment and specify what Python version you want.



Getting Help
~~~~~~~~~~~

If you continue to experience installation issues:

* Check the :doc:`troubleshooting` guide
* Email me at liamm@wustl.edu

Next Steps
---------

After installation, proceed to the :doc:`quickstart` guide to begin using NetTracer3D.