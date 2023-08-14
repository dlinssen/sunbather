# Installing _Cloudy_

_cloupy_ was developed using _Cloudy v17.02_. In fall 2022, _Cloudy v22_ was released. As we have not yet tested _cloupy_ with this new version, we currently require you to install v17.02. Complete download and installation instructions can be found [here](https://trac.nublado.org/wiki/StepByStep). In short, for most Unix systems, the steps are as follows:

1. Go to the [download page](https://data.nublado.org/cloudy_releases/c17/old/) and  download the "c17.02.tar.gz" file.
2. Extract it in a location where you want to install _Cloudy_.
3. `cd` into the _/c17.02/source/_ folder and compile the code by running `make`.
4. Quickly test the installation: in the source folder, run `./cloudy.exe`, type "test" and hit return twice. It should print "Cloudy exited OK" at the end.

If you have trouble installing _Cloudy v17.02_, we refer to the download instructions linked above, as well as the _Cloudy_ [help forum](https://cloudyastrophysics.groups.io/g/Main/topics).

# Installing _cloupy_

1. Clone _cloupy_ from Github. The code runs entirely in Python. It was developed using Python 3.9.0 and the following packages are prerequisites: `numpy (v1.24.3), pandas (v1.1.4), matplotlib (v3.7.1), scipy (v1.8.0), astropy (v5.3), p-winds (v1.3.4)`. _cloupy_ has not yet been tested on different versions of Python and prerequisite packages, so we currently cannot guarantee that it works, but feel free to try! In any case, we recommend making a Python [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) to run _cloupy_ in.
2. Create a directory anywhere on your machine where the code will save all models/simulations/etc. This will be the "project" folder, and you can give it any name you like. This is to keep the output of _cloupy_ separate from the _cloupy_ source code.
3. Open the _/src/config.ini_ file and add your path to the _Cloudy v17.02_ installation, as well as the path to the "project" folder.
4. Test your installation. Copy _/examples/materials/eps_Eri_binned.spec_ to _/c17.02/data/SED/_ and then run _/tests/test.py_, which should print "Success". If the test fails, feel free to contact d.c.linssen@uva.nl with your error.

# Getting started

1. To get familiar with _cloupy_, we reccommend you go through the Jupyter notebooks in the _/cloupy/examples/_ folder, where example use cases (such as creating atmospheric profiles, calculating transmission spectra and fitting observational data) are worked out and explained. 
2. For more details on how to use the code, check out the Glossary page on this wiki. We specifically reccommend you read the sections "The _src/planets.txt_ file" and "Stellar SED handling". 