.. _installation:

============
Installation
============

NetTracer3D can be installed either as a Python package via pip (recommended)
or, on Windows, as a standalone installer.

.. contents:: On this page
   :local:
   :depth: 2


System Requirements
-------------------

* **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)
* **CPU**: Multi-core processor (4+ cores recommended)
* **RAM**: 8 GB minimum; 16 GB or more recommended for larger images. Very large
  datasets such as lightsheet data may require a workstation with 128 GB or more,
  or downsampling of the data beforehand.
* **GPU**: NVIDIA dedicated GPU (optional)
* **Python**: 3.12


Windows Installer
-----------------

On Windows, the simplest option is to download and run the installer from the
GitHub releases page:
https://github.com/mclaughlinliam3/NetTracer3D/releases

.. note::

   The installer version lacks a few features compared to the Python package —
   most notably GPU support for segmentation — and is updated less frequently.


Installing with pip
-------------------

1. **Install Python and pip**

   Download the latest Python installer from https://www.python.org/. During
   installation, enable the options to install pip and to add both Python and
   pip to your ``PATH``.

   To verify the installation on Windows, press :kbd:`Win` + :kbd:`R`, type
   ``cmd`` to open a command terminal, and run ``where pip`` or ``where python``.
   If the installation succeeded, the terminal reports the folder containing the
   corresponding executable. If neither is found, they were either not installed
   or not added to your ``PATH``. Reinstall with the correct options selected, or
   add them manually to the environment variables through the Windows Control
   Panel.

2. **Install the base package**

   Open a command terminal and run:

   .. code-block:: bash

       pip install nettracer3d

3. **Optional — performance boost for large data**

   The ``edt`` module enables parallelised CPU calculations for several of the
   search functions, which can improve their speed by an order of magnitude or
   more depending on your core count. This is a substantial benefit on systems
   with a strong CPU and sufficient RAM.

   Because ``edt`` requires an extra pre-installation step, it is not included by
   default. First install the Microsoft C++ build tools from
   https://visualstudio.microsoft.com/visual-cpp-build-tools/, selecting the
   **Desktop Development with C++** option in the installer. Use the Python
   distribution from python.org rather than the Windows Store version, as ``edt``
   may not work correctly otherwise. Then run:

   .. code-block:: bash

       pip install nettracer3d[edt]

4. **Optional — recommended full package**

   To install ``edt`` together with the Leiden partition dependencies:

   .. code-block:: bash

       pip install nettracer3d[rec]

5. **Adding optional components later**

   If the default version is already installed, the Leiden and ``edt``
   components can be added individually:

   .. code-block:: bash

       pip install edt
       pip install leidenalg
       pip install igraph

Once NetTracer3D and its core dependencies are installed, launch the program
from a command terminal with:

.. code-block:: bash

    nettracer3d


Installing with Anaconda
------------------------

Anaconda manages Python packages for you and can resolve conflicting package
versions during installation. The process mirrors the pip installation above,
except that NetTracer3D is housed in a dedicated Anaconda environment rather
than among your system Python packages.

Download the version of Anaconda matching your operating system from
https://www.anaconda.com/download and run the installer. When it finishes,
search for **Anaconda Prompt** in your taskbar, open it, and run:

.. code-block:: bash

    conda create --name nettracer3d python=3.12

    conda activate nettracer3d

    pip install nettracer3d[rec]

This installs the program into the ``nettracer3d`` environment. To run
NetTracer3D thereafter, open the Anaconda Prompt and enter:

.. code-block:: bash

    conda activate nettracer3d

    nettracer3d


GPU Support
-----------

NetTracer3D is mostly CPU-bound, but a few functions can optionally use the GPU.
GPU support requires an NVIDIA GPU with a CUDA toolkit installed. Find the CUDA
toolkit compatible with your GPU and install it using the auto-installer from
https://developer.nvidia.com/cuda-toolkit.

With a CUDA toolkit in place, use:

.. code-block:: bash

    pip install nettracer3d[CUDA11] #If your CUDA toolkit is version 11
    pip install nettracer3d[CUDA12] #If your CUDA toolkit is version 12
    pip install nettracer3d[cupy] #For the generic cupy library (The above two are usually the ones you want)

If the base package is already installed and you want only the GPU-associated
packages:

.. code-block:: bash

    pip install cupy-cuda11x #If your CUDA toolkit is version 11
    pip install cupy-cuda12x #If your CUDA toolkit is version 12
    pip install cupy #For the generic cupy library (The above two are usually the ones you want)

The Cellpose plugin, for which GPU usage is effectively obligatory, additionally
requires PyTorch. Use the build menu at https://pytorch.org/ to obtain a pip
command compatible with your Python and CUDA versions.


Verifying the Installation
--------------------------

To confirm that NetTracer3D installed correctly, run:

.. code-block:: bash

    pip show nettracer3d

The current version number should be displayed.


Troubleshooting
---------------

Conflicting dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter dependency conflicts, install NetTracer3D into a clean Anaconda
environment:

.. code-block:: bash

    conda create --name nettracer3d python=3.12

    conda activate nettracer3d

    pip install nettracer3d

The most likely cause is a package that is incompatible with NumPy 2.0 or above.
Upgrade or downgrade the offending package as needed. Avoid downgrading NumPy
below 2.0 — it is the main workhorse for image processing in Python, and the 2.0
release introduced significant optimisations. A version pin generally takes the
form:

.. code-block:: bash

    pip install 'numpy<=2.2' # If a version of numpy beyond 2.2 was causing a conflict, for example.

Python version problems
~~~~~~~~~~~~~~~~~~~~~~~

Some packages that NetTracer3D depends on lack proper build protocols for newer
Python releases. If installation begins but crashes part-way through, try
downgrading Python. Installation should work reliably on Python 3.12; as of
writing (12/13/2025), later versions may cause issues, specifically with the
``sklearn`` package.

To downgrade on Windows, uninstall Python from the Add/Remove Programs menu and
reinstall version 3.12. Under Anaconda, simply create a new environment
specifying the desired Python version.

Installed but not starting
~~~~~~~~~~~~~~~~~~~~~~~~~~

If Windows does not recognise ``nettracer3d`` as a command, it is likely using
the Windows Store version of Python. Uninstall it and install Python from
python.org instead, which allows packages to be run directly from the command
line.

Getting help
~~~~~~~~~~~~

If installation issues persist, email liamm@wustl.edu.


Next Steps
----------

After installation, proceed to the :doc:`quickstart` guide to begin using
NetTracer3D.
