# Installing _Cloudy_

_sunbather_ has been developed and tested with _Cloudy v17.02_ and _v23.01_. Newer versions of _Cloudy_ are likely also compatible with _sunbather_, but this has not been thoroughly tested. Therefore, we currently recommend using _v23.01_. Complete _Cloudy_ download and installation instructions can be found [here](https://gitlab.nublado.org/cloudy/cloudy/-/wikis/home). In short, for most Unix systems, the steps are as follows:

1. Go to the [v23 download page](https://data.nublado.org/cloudy_releases/c23/) and download the "c23.01.tar.gz" file (or go to the [v17 download page](https://data.nublado.org/cloudy_releases/c17/old/) and  download the "c17.02.tar.gz" file).
2. Extract it in a location where you want to install _Cloudy_.
3. `cd` into the _/c23.01/source/_ or _/c17.02/source/_ folder and compile the code by running `make`.
4. Quickly test the _Cloudy_ installation: in the source folder, run `./cloudy.exe`, type "test" and hit return twice. It should print "Cloudy exited OK" at the end.

If you have trouble installing _Cloudy_, we refer to the download instructions linked above, as well as the _Cloudy_ [help forum](https://cloudyastrophysics.groups.io/g/Main/topics).

# Installing _sunbather_

1. Clone _sunbather_ from Github. The code runs entirely in Python. It was developed using Python 3.9.0 and the following packages are prerequisites: `numpy (v1.24.3), pandas (v1.1.4), matplotlib (v3.7.1), scipy (v1.8.0), astropy (v5.3), p-winds (v1.3.4)`. _sunbather_ also succesfully ran with the newest versions (as of Sep. 18, 2023) of these packages. We have however not yet thoroughly tested all of its functionality with these newer versions, so we currently cannot guarantee that it works, but feel free to try! In any case, we recommend making a Python [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) to run _sunbather_ in.
2. Create a directory anywhere on your machine where the code will save all models/simulations/etc. This will be the "project" folder, and you can give it any name you like. This is to keep the output of _sunbather_ separate from the _sunbather_ source code.
3. Set an environmental variable `$CLOUDY_PATH` to your _Cloudy_ installation base directory, and set `$SUNBATHER_PROJECT_PATH` to the "project" folder. We recommend setting these in your _~/.bashrc_ or _~/.zshrc_ file: 
	```
	export CLOUDY_PATH="/full/path/to/c23.01/"
	export SUNBATHER_PROJECT_PATH="/full/path/to/project/folder/"
	```
4. Copy the */sunbather/planets.txt* file to your project folder.
5. Copy the stellar spectra from _/sunbather/stellar_SEDs/_ to _$CLOUDY_PATH/data/SED/_ . These include the [MUSCLES](https://archive.stsci.edu/prepds/muscles/) spectra.
6. Test your _sunbather_ installation: run _/sunbather/tests/test.py_, which should print "Success". If the test fails, feel free to open an issue or contact d.c.linssen@uva.nl with your error.

# Getting started

1. To get familiar with _sunbather_, we recommend you go through the Jupyter notebooks in the _/sunbather/examples/_ folder, where example use cases (such as creating atmospheric profiles, calculating transmission spectra and fitting observational data) are worked out and explained. 
2. For more details on how to use the code, check out the Glossary and FAQ pages on this wiki. We specifically recommend you read the glossary sections "The _planets.txt_ file" and "Stellar SED handling". 