# sunbather

![sunbather logo](logo/Logo_1000dpi.png)

This is a Python package to simulate the upper atmospheres of exoplanets and their observational signatures.

The main use of the code is to construct 1D Parker wind profiles (Parker 1958) using the Python _[p-winds](https://github.com/ladsantos/p-winds)_ package (dos Santos et al. 2022), to simulate these with photoionization code _[Cloudy](https://gitlab.nublado.org/cloudy/cloudy)_ (Ferland et al. 1998; 2017), and to postprocess the output with a custom radiative transfer module to predict the transmission spectra of exoplanets.

If you make use of _sunbather_, please cite Linssen et al. (2024).

## Installation
Installation instructions can be found in the wiki.