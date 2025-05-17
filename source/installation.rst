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

This will install NetTracer3D and all its dependencies.

To install optional GPU functionalities, first set up a CUDA toolkit that runs with the GPU on your machine and use:

.. code-block:: bash
    pip install nettracer3d[CUDA11] #For CUDA toolkit 11 compatibility
    pip install nettracer3d[CUDA12] #For CUDA toolkit 12 compatibility
    pip install nettracer3d[cupy] #For the generic cupy library.


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

The highest probability is that a version of a package you are running is not compatible with numpy version 2 and above. Please try updating the incompatible package in that case, or downgrading numpy to a version 1 distribution.

Getting Help
~~~~~~~~~~~

If you continue to experience installation issues:

* Check the :doc:`troubleshooting` guide
* Email me at liamm@wustl.edu

Next Steps
---------

After installation, proceed to the :doc:`quickstart` guide to begin using NetTracer3D.