# Sensory Island Task (SIT)

The project provides several Jupyter notebooks for the behavioral testing of animals in open-field-based sensory perception  experiments, the Sensory Island Task, as described in Ferreiro et al. 2020 (doi:XYZ **TODO**) and the pre-processing of the output data for subsequent analysis in MATLAB.

### Installation

This project was developed and tested on different Windows (Win7 and Win10) machines. We recommend using Anaconda to run the project notebooks.

To install Anaconda, download the Anaconda installer with Python 3 (tested with Anaconda3-2019.10, bundled with Python 3.7) and follow the instructions on the download page.

After installation, we recommend creating an Anaconda environment from which Jupyter Notebook will be started to run the provided notebooks. You can create new environments using Anaconda's navigator or from an Anaconda prompt using:

    conda create --name myenv

Once created, activate the environment and make sure it has both, Python 3 and R installed. Usually, you will also have to install an R-Kernel for Jupyter Notebook. To do so, open an Anaconda promt and go to the environment you created:

    activate myenv

Once the environment has been activated, install R, its essentials, and the kernel via:

    conda install r-base
    conda install r-essentials 
    conda install -c r r-irkernel

Finally, for the notebooks to run, you will have to install additional python modules and R libraries and their dependencies to your environment. Notebooks may not run smoothly, if the wrong versions of the modules are installed. Especially having the wrong versions of openCV can cause trouble, which is why we provide the Jupyter notebooks that run on Python 3 for two different versions of openCV (3.4.2 & 4.0.1). The code below will install the latest version of the modules/libraries to your environment. Version numbers in brackets are the latest module/library versions the notebooks were tested with (5th of Febuary, 2020).

**Python modules:** numpy (1.18.1), openCV (3.4.2 or 4.0.1), tkinter (8.6.8), scipy (1.3.2), sounddevice (0.3.14), pyfirmata (1.1.0)

    conda install numpy
    conda install opencv
    conda install tk
    conda install scipy
    conda install -c conda-forge python-sounddevice
    conda install -c conda-forge pyfirmata

**R libraries:** R.matlab (3.6.2), xlsx (0.6.1)

    conda install -c r r-R.matlab
    conda install -c r r-xlsx

Once you have installed the modules and libraries, run Juypter Notebook from your environment

    jupyter notebook

To work with the different notebooks, select the folder to which you downloaded the files of this repository in the Jupyter browser and follow the instruction provided in the respective notebook.

## Authors

* **Daniel Schmidtke**
* **Diana Amaro**
* **Andrey Sobolev**

## License

This project is licensed under the MIT License (see the license file for details).

## Acknowledgments

* The implemented tracking algorithm was inspired by **colinlaney**'s animal-tracking at: https://github.com/colinlaney/animal-tracking
