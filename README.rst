SmartSim OLCF/NERSC Workshop 2024
=================================

This repository contains materials to run a hybrid scientific simulation/AI
workflow prepared for a joint workshop between NERSC and OLCF. This workflow has
four components: a mock simulation, a sampling service that reduces the amount
of data from the simulation, a training service that consumes the data to train
a simple neural network, and an in-memory datastore used to transfer data
between the various components.

The data used for this example is a small subset of work done under an
ASCR Leadership Computing Challenge project using the MFIX-Exa multiphase
flow solver:
- William Fullmer, Jordan Musser, Aytekin Gel, Sarah Beetham, Machine
Learning-Enhanced Multiphase CFD for Carbon Capture Modeling, 11/29/2023,
https://edx.netl.doe.gov/dataset/machine-learning-enhanced-multiphase-cfd-for-carbon-capture-modeling,
DOI: 10.18141/2344941

Installation (Perlmutter)
-------------------------

**Step 1:** Follow the guide here to install SmartSim and SmartRedis
`Perlmutter Instructions <https://www.craylabs.org/develop/installation_instructions/platform.html#nersc-perlmutter>`_

**Step 2:** Setup the compilation environment

.. code:: bash

    module load conda cudatoolkit/12.2 cudnn/8.9.3_cuda12 PrgEnv-gnu
    conda activate smartsim
    export SMARTREDIS_HOME=CHANGE_ME/SmartRedis

**Step 3:** Clone down this repository

.. code:: bash

    cd $PSCRATCH
    git clone --branch nersc_olcf_2024 https://github.com/CrayLabs/smartsim_workshops.git

**Step 4:** Compile the mock simulation

.. code:: bash

    cd smartsim_workshops
    mkdir -p build
    cd build
    cmake -Dsmartredis_DIR=$SMARTREDIS_HOME/install/share/cmake/smartredis \
          -Dsmartredis-fortran_DIR=$SMARTREDIS_HOME/install/share/cmake/smartredis-fortran \
          -DCMAKE_INSTALL_PREFIX=../ ..
    make
    cd ..

**Step 5:** Install the python packages needed for the example

.. code:: bash

    pip install -r requirements.txt

Installation (Frontier)
-----------------------

**Step 1:** Follow the guide here to install SmartSim and SmartRedis
`Frontier Instructions <https://www.craylabs.org/develop/installation_instructions/platform.html#olcf-frontier>`_

**Step 2:** Setup the compilation environment

.. code:: bash

    export PROJECT_NAME=CHANGE_ME
    export SCRATCH=/lustre/orion/$PROJECT_NAME/scratch/$USER/
    module load PrgEnv-gnu miniforge3 rocm/6.1.3
    source activate smartsim

**Step 3:** Clone down this repository

.. code:: bash

    cd $SCRATCH
    git clone --branch nersc_olcf_2024 https://github.com/CrayLabs/smartsim_workshops.git

**Step 4:** Compile the mock simulation

.. code:: bash

    cd smartsim_workshops
    mkdir -p build
    cd build
    cmake -Dsmartredis_DIR=$SCRATCH/SmartRedis/install/share/cmake/smartredis \
          -Dsmartredis-fortran_DIR=$SCRATCH/SmartRedis/install/share/cmake/smartredis-fortran \
          -DCMAKE_INSTALL_PREFIX=../ ..
    make
    cd ..

**Step 5:** Install the python packages needed for the example

.. code:: bash

    pip install -r requirements.txt

Running Examples (Perlmutter)
-----------------------------

.. code:: bash

    cd $PSCRATCH/smartsim_workshops
    salloc -N 4 -A ntrain1 --reservation=smartsim_workshop --constraint gpu
    python driver.py

Running Examples (Frontier)
-----------------------------

.. code:: bash

    cd $SCRATCH/smartsim_workshops
    salloc -N 4 -A CHANGE_ME --reservation=smartsim_2024
    python driver.py
