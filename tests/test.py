import os
import sys
this_path = os.path.dirname(os.path.abspath(__file__)) #the absolute path where this code lives
src_path = this_path.split('tests')[-2] + 'src/'
sys.path.append(src_path)

#sunbather imports
import tools
import RT

#other imports
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d



print("\nWill perform installation check by running the three main sunbather modules and checking if the output is as expected. " \
      +"Expected total run-time: 10 to 60 minutes. Should print 'success' at the end.\n")

### SETUP CHECKS ###

#check if paths are set by user
assert tools.cloudypath != '/full/path/to/c17.02/', "Please set the path to your Cloudy installation in config.ini"
assert tools.projectpath != '/full/path/to/project/', "Please set your project path in config.ini"
#make sure projectpath exists
assert os.path.isdir(tools.projectpath), "Please create the projectpath folder on your machine"
#make sure the SED we need for this test has been copied to Cloudy
assert os.path.isfile(tools.cloudypath+'/data/SED/eps_Eri_binned.spec'), "Please copy /sunbather/stellar_SEDs/eps_Eri_binned.spec into /c17.02/data/SED/"



print("\nChecking construct_parker.py. A runtime for this module will follow when done...\n")

### CREATING PARKER PROFILE ###

#create a parker profile - we use the p-winds/Cloudy hybrid scheme
os.system("cd .. && cd src && python construct_parker.py -plname WASP52b -pdir test -Mdot 11.0 -T 9000 -z 10 -zelem Ca=0 -overwrite")
#load the created profile
pprof_created = pd.read_table(tools.projectpath+'/parker_profiles/WASP52b/test/pprof_WASP52b_T=9000_M=11.000.txt',
                                names=['alt', 'rho', 'v', 'mu'], dtype=np.float64, comment='#')
#load the expected output
pprof_expected = pd.read_table('materials/pprof_WASP52b_T=9000_M=11.000.txt',
                                names=['alt', 'rho', 'v', 'mu'], dtype=np.float64, comment='#')
#check if they are equal to within 1% in altitude and mu and 10% in rho and v.
assert np.isclose(pprof_created[['alt', 'mu']], pprof_expected[['alt', 'mu']], rtol=0.01).all().all(), "The profile created with the construct_parker.py module is not as expected"
assert np.isclose(pprof_created[['rho', 'v']], pprof_expected[['rho', 'v']], rtol=0.1).all().all(), "The profile created with the construct_parker.py module is not as expected"



print("\nChecking convergeT_parker.py. A runtime for this module will follow when done...\n")

### CONVERGING TEMPERATURE STRUCTURE WITH CLOUDY ###

#run the created profile through Cloudy
os.system("cd .. && cd src && python convergeT_parker.py -plname WASP52b -pdir test -dir test -Mdot 11.0 -T 9000 -z 10 -zelem Ca=0 -save_sp He Mg+ -overwrite")
#load the created simulation
sim_created = tools.Sim(tools.projectpath+'/sims/1D/WASP52b/test/parker_9000_11.000/converged')
#load the expected simulation
sim_expected = tools.Sim('materials/converged')
#interpolate them to a common altitude grid as Cloudy's internal depth-grid may vary between simulations
alt_grid = np.logspace(np.log10(max(sim_created.ovr.alt.iloc[-1], sim_expected.ovr.alt.iloc[-1])+1e4), 
                       np.log10(min(sim_created.ovr.alt.iloc[0], sim_expected.ovr.alt.iloc[0])-1e4), num=100)
T_created = interp1d(sim_created.ovr.alt, sim_created.ovr.Te)(alt_grid)
T_expected = interp1d(sim_expected.ovr.alt, sim_expected.ovr.Te)(alt_grid)
#check if they are equal to within 10%
assert np.isclose(T_created, T_expected, rtol=0.1).all(), "The converged temperature profile of Cloudy is not as expected"



print("\nChecking RT.py...\n")

### MAKING TRANSIT SPECTRA ###

#make a helium spectrum
wavs = np.linspace(10830, 10836, num=300)
FinFout_created, found_lines, notfound_lines = RT.FinFout_1D(sim_created, wavs, 'He')
#load the expected helium spectrum
FinFout_expected = np.genfromtxt('materials/FinFout_helium.txt')[:,1]
assert np.isclose(FinFout_created, FinFout_expected, rtol=0.05).all(), "The created helium spectrum is not as expected"
#make a magnesium+ spectrum
wavs = np.linspace(2795.5, 2797, num=300)
FinFout_created, found_lines, notfound_lines = RT.FinFout_1D(sim_created, wavs, 'Mg+')
#load the expected magnesium+ spectrum
FinFout_expected = np.genfromtxt('materials/FinFout_magnesium+.txt')[:,1]
assert np.isclose(FinFout_created, FinFout_expected, rtol=0.05).all(), "The created magnesium+ spectrum is not as expected"



#if we made it past all the asserts, the code is correctly installed
print("\nSuccess.")
