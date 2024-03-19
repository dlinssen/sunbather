This wiki page is a glossary that provides additional information on various modules/classes/functionalities included in _sunbather_. We also refer to "Hazy", which is the official documentation of _Cloudy_ and can be found in your _$CLOUDY_PATH/docs/_ folder. 

<br>

- #### The `tools.py` module
This module contains many basic functions and classes that are used by the other _sunbather_ modules, and can also be used when postprocessing/analyzing _sunbather_ output. 

This module is not intended to be run from the command line, but rather imported into other scripts in order to use its functions.

<br>

- #### The `RT.py` module
This module contains functions to perform radiative transfer calculations of the planet transmission spectrum. 

This module is not intended to be run from the command line, but rather imported into other scripts in order to use its functions.

<br>

- #### The `construct_parker.py` module
This module is used to create Parker wind profiles. The module can make pure H/He profiles, in which case it is basically a wrapper around the [`p-winds` code](https://github.com/ladsantos/p-winds) (dos Santos et al. 2022). The code can however also make Parker wind profiles for an arbitrary composition (e.g. at a given scaled solar metallicity), which is much more computationally expensive, because it then iteratively runs `p-winds` and _Cloudy_. In this mode, _Cloudy_ is used to obtain the mean molecular weight structure of the atmosphere for the given composition, which `p-winds` uses to calculate the density and velocity structure. 

This module is intended to be run from the command line while supplying arguments. Running `python construct_parker.py --help` will give an explanation of each argument.

Example use: `python construct_parker.py -plname WASP52b -pdir z_10 -T 8000 -Mdot 11.0 -z 10`. This creates a Parker wind profile for the planet WASP52b (must be defined in *planets.txt*) for a temperature of 8000 K, mass-loss rate of 10^11 g s-1 and a 10x solar metallicity composition, and saves the atmospheric structure as a .txt file in *$SUNBATHER_PROJECT_PATH/parker_profiles/WASP52b/z_10/*.

<br>

- #### The `convergeT_parker.py` module
This module is used to run Parker wind profiles through _Cloudy_ to (iteratively) solve for a non-isothermal temperature structure. Additionally, the "converged" simulation can then be postprocessed with functionality of the `RT.py` module in order to make transmission spectra. This module is basically a convenience wrapper which sets up the necessary folder structure and input arguments for the `solveT.py` module that actually performs the iterative scheme described in Linssen et al. (2022).

This module is intended to be run from the command line while supplying arguments. Running `python convergeT_parker.py --help` will give an explanation of each argument.

Example use: `python convergeT_parker.py -plname HATP11b -pdir fH_0.99 -dir fiducial -T 5000 10000 200 -Mdot 9.0 11.0 0.1 -zelem He=0.1 -cores 4 -save_sp H He Ca+`. This simulates Parker wind models with Cloudy for the planet HATP11b (must be defined in *planets.txt*) for a grid of temperatures between 5000 K and 10000 K in steps of 200 K, mass-loss rates between 10^9 g s-1 and 10^11 g s-1 in steps of 0.1 dex. It looks for the density and velocity structure of these models in the folder *$SUNBATHER_PROJECT_PATH/parker_profiles/HATP11b/fH_0.99/* (so these models have to be created first in that folder using `construct_parker.py`) and saves the _Cloudy_ simulations in the folder *$SUNBATHER_PROJECT_PATH/sims/1D/HATP11b/fiducial/*. It scales the abundance of helium (which is solar by default in _Cloudy_, i.e. ~10% by number) by a factor 0.1 so that it becomes 1% by number. 4 different calculations of the $T$-$\dot{M}$-grid are done in parallel, and the atomic hydrogen, helium and singly ionized calcium output are saved by _Cloudy_, so that afterwards we can use `RT.FinFout_1D()` to make Halpha, metastable helium and Ca II infrared triplet spectra.

<br>

- #### The `solveT.py` module
This module contains the iterative scheme described in Linssen et al. (2022) to solve for a non-isothermal temperature structure of a given atmospheric profile. It is called by `convergeT_parker.py`. As long as you're simulating Parker wind profiles (and not some other custom profile), you should be fine using `convergeT_parker.py` instead of this module.

<br>

- #### The _$SUNBATHER_PROJECT_PATH / tools.projectpath_ directory
This is the directory on your machine where all Parker wind profiles and _Cloudy_ simulations are saved. You can choose any location and name you like, as long as it doesn't contain any spaces. The full path to this directory must be set as your `$SUNBATHER_PROJECT_PATH` environmental variable (see installation instructions). The reason _sunbather_ uses a project path is to keep all output from simulations (i.e. user-specific files) separate from the source code.

<br>

- #### The _planets.txt_ file
This file stores the bulk parameters of the planets that are simulated. A template of this file is provided in the _sunbather_ base directory, but you must copy it to your _$SUNBATHER_PROJECT_PATH_ in order for it to work. Every time you want to simulate a new planet/star system, you must add a line to this file with its parameters. You can add comments at the end of the line with a # (for example referencing where the values are from). The first column specifies the "name", which is a tag for this system that cannot contain spaces and is used for the `-plname` argument of `construct_parker.py` and `convergeT_parker.py`, as well as for the `tools.Planet` class to access the system parameters in Python. The second column specifies the "full name", which can be any string you like and can be used e.g. when plotting results. The third column is the radius of the planet in Jupiter radii (7.1492e9 cm). The fourth column is the radius of the star in solar radii (6.9634e10 cm). The fifth column is the semi-major axis of the system in AU (1.49597871e13 cm). The sixth column is the mass of the planet in Jupiter masses (1.898e30 g). The seventh column is the mass of the star in solar masses (1.9891e33 g). The eighth column is the transit impact parameter (dimensionless, 0 is across the center of the stellar disk, 1 is grazing the stellar limb). The ninth column is the name of the stellar SED - see "Stellar SED handling" below in this glossary.

<br>

- #### Stellar SED handling
When running _sunbather_, the spectral energy distribution (SED) of the host star has to be available to _Cloudy_, which looks for it in its _$CLOUDY_PATH/data/SED/_ folder. Therefore, every SED you want to use has be **copied to that folder, and requires a specific format**: the first column must be wavelengths in units of Å and the second column must be the $\lambda F_{\lambda} = \nu F_{\nu}$ flux **at a distance of 1 AU** in units of erg s-1 cm-2. Additionally, on the first line, after the first flux value, the following keywords must appear: "units angstrom nuFnu". In the */sunbather/stellar_SEDs/* folder, we have provided a few example SEDs in the correct format. Even though _Cloudy_ in principle supports other units, _sunbather_ doesn't, so please stick to the units as described. Normalization of the flux to the planet orbital distance is done automatically by *sunbather* based on the semi-major axis value given in the *planets.txt* file.
