import os
import sys

# other imports
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import shutil

# sunbather imports
from sunbather import tools, RT


# the absolute path where this code lives
this_path = os.path.dirname(os.path.abspath(__file__))
src_path = this_path.split('tests')[-2] + 'src/'

print(
    "\nWill perform installation check by running the three main sunbather modules and checking if the output is as expected. "
    + "Expected total run-time: 10 to 60 minutes. Should print 'success' at the end.\n"
)

# SETUP CHECKS

# make sure projectpath exists
projectpath = tools.get_sunbather_project_path()

assert os.path.isdir(
    projectpath
), "Please create the projectpath folder on your machine"
# make sure the planets.txt file exists
assert os.path.isfile(
    projectpath + "/planets.txt"
), "Please make sure the 'planets.txt' file is present in $SUNBATHER_PROJECT_PATH"
# make sure the SED we need for this test has been copied to Cloudy
assert os.path.isfile(
    tools.get_cloudy_path() + "/data/SED/eps_Eri_binned.spec"
), "Please copy /sunbather/stellar_SEDs/eps_Eri_binned.spec into $CLOUDY_PATH/data/SED/"


# ## CHECK IF test.py HAS BEEN RAN BEFORE ###

parker_profile_file = (
    projectpath
    + "/parker_profiles/WASP52b/test/pprof_WASP52b_T=9000_M=11.000.txt"
)
simulation_folder = projectpath + "/sims/1D/WASP52b/test/parker_9000_11.000/"

if os.path.exists(parker_profile_file) or os.path.exists(simulation_folder):
    confirmation = input(
        f"It looks like test.py has been ran before, as {parker_profile_file} and/or {simulation_folder} already exist. Do you want to delete the previous output before continuing (recommended)? (y/n): "
    )
    if confirmation.lower() == "y":
        if os.path.exists(parker_profile_file):
            os.remove(parker_profile_file)
        if os.path.exists(simulation_folder):
            shutil.rmtree(simulation_folder)
        print("\nFile(s) deleted successfully.")
    else:
        print("\nDeletion cancelled.")


print(
    "\nChecking construct_parker.py. A runtime for this module will follow when done...\n"
)

### CREATING PARKER PROFILE ###

# create a parker profile - we use the p-winds/Cloudy hybrid scheme
os.system(
    f"cd {tools.sunbatherpath} && python construct_parker.py -plname WASP52b -pdir test -Mdot 11.0 -T 9000 -z 10 -zelem Ca=0 -overwrite"
)
# load the created profile
pprof_created = pd.read_table(
    projectpath
    + "/parker_profiles/WASP52b/test/pprof_WASP52b_T=9000_M=11.000.txt",
    names=["alt", "rho", "v", "mu"],
    dtype=np.float64,
    comment="#",
)
# load the expected output
pprof_expected = pd.read_table(
    this_path + "/materials/pprof_WASP52b_T=9000_M=11.000.txt",
    names=["alt", "rho", "v", "mu"],
    dtype=np.float64,
    comment="#",
)
# check if they are equal to within 1% in altitude and mu and 10% in rho and v.
assert (
    np.isclose(pprof_created[["alt", "mu"]], pprof_expected[["alt", "mu"]], rtol=0.01)
    .all()
    .all()
), "The profile created with the construct_parker.py module is not as expected"
assert (
    np.isclose(pprof_created[["rho", "v"]], pprof_expected[["rho", "v"]], rtol=0.1)
    .all()
    .all()
), "The profile created with the construct_parker.py module is not as expected"


print(
    "\nChecking convergeT_parker.py. A runtime for this module will follow when done...\n"
)

# ## CONVERGING TEMPERATURE STRUCTURE WITH CLOUDY ###

# run the created profile through Cloudy
os.system(
    f"cd {tools.sunbatherpath} "
    f"&& python convergeT_parker.py "
    f"-plname WASP52b -pdir test -dir test "
    f"-Mdot 11.0 -T 9000 -z 10 -zelem Ca=0 -overwrite"
)
# load the created simulation
sim_created = tools.Sim(
    projectpath + "/sims/1D/WASP52b/test/parker_9000_11.000/converged"
)
# load the expected simulation
sim_expected = tools.Sim(this_path + "/materials/converged")
# interpolate them to a common altitude grid as Cloudy's internal depth-grid may vary between simulations
alt_grid = np.logspace(
    np.log10(max(sim_created.ovr.alt.iloc[-1], sim_expected.ovr.alt.iloc[-1]) + 1e4),
    np.log10(min(sim_created.ovr.alt.iloc[0], sim_expected.ovr.alt.iloc[0]) - 1e4),
    num=100,
)
T_created = interp1d(sim_created.ovr.alt, sim_created.ovr.Te)(alt_grid)
T_expected = interp1d(sim_expected.ovr.alt, sim_expected.ovr.Te)(alt_grid)
# check if they are equal to within 10%
assert np.isclose(
    T_created, T_expected, rtol=0.1
).all(), "The converged temperature profile of Cloudy is not as expected"


print("\nChecking RT.py...\n")

### MAKING TRANSIT SPECTRA ###

# make a helium spectrum
wavs = np.linspace(10830, 10836, num=300)
FinFout_created, found_lines, notfound_lines = RT.FinFout(sim_created, wavs, "He")
# load the expected helium spectrum
FinFout_expected = np.genfromtxt(this_path + "/materials/FinFout_helium.txt")[:, 1]
assert np.isclose(
    FinFout_created, FinFout_expected, rtol=0.05
).all(), "The created helium spectrum is not as expected"
# make a magnesium+ spectrum
wavs = np.linspace(2795.5, 2797, num=300)
FinFout_created, found_lines, notfound_lines = RT.FinFout(sim_created, wavs, "Mg+")
# load the expected magnesium+ spectrum
FinFout_expected = np.genfromtxt(this_path + "/materials/FinFout_magnesium+.txt")[:, 1]
assert np.isclose(
    FinFout_created, FinFout_expected, rtol=0.05
).all(), "The created magnesium+ spectrum is not as expected"


# if we made it past all the asserts, the code is correctly installed
print("\nSuccess.")
