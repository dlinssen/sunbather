import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
from shutil import copyfile
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import scipy.stats as sps
from scipy.ndimage import gaussian_filter1d
from fractions import Fraction
import configparser


config = configparser.ConfigParser()
sunbather_path = os.path.dirname(os.path.abspath(__file__)) #the absolute path where this code lives
config.read(os.path.join(sunbather_path, 'config.ini'))
cloudypath = config.get('General', 'cloudypath') #the path where the Cloudy installation is
projectpath = config.get('General', 'projectpath') #the path where you save your simulations and do analysis
cloudyruncommand = cloudypath+'/source/cloudy.exe -p' #the -p flag is important!

#read planet parameters globally instead of in the Planets class (so we do it only once)
planets_file = pd.read_csv(sunbather_path+'/planets.txt', dtype={'name':str, 'full name':str, 'R [RJ]':np.float64,
                            'Rstar [Rsun]':np.float64, 'a [AU]':np.float64, 'M [MJ]':np.float64, 'Mstar [Msun]':np.float64,
                            'transit impact parameter':np.float64, 'SEDname':str}, comment='#')

#define constants:
c = 2.99792458e10 #cm/s
h = 4.135667696e-15 #eV s, used to plot wavelengths in keV units
mH = 1.674e-24 #g
k = 1.381e-16 #erg/K
AU = 1.49597871e13 #cm
pc = 3.08567758e18 #cm
RJ = 7.1492e9 #cm
RE = 6.371e8 #cm
Rsun = 69634000000 #cm
Msun = 1.9891e33 #g
MJ = 1.898e30 #g
ME = 5.9722e27 #g
G = 6.6743e-8 #cm3/g/s2
Ldict = {'S':0, 'P':1, 'D':2, 'F':3, 'G':4, 'H':5, 'I':6, 'K':7, 'L':8,
        'M':9, 'N':10, 'O':11, 'Q':12, 'R':13, 'T':14} #atom number of states per L orbital

element_names = {'H':'hydrogen', 'He':'helium', 'Li':'lithium', 'Be':'beryllium', 'B':'boron', 'C':'carbon',
                'N':'nitrogen', 'O':'oxygen', 'F':'fluorine', 'Ne':'neon', 'Na':'sodium',
                'Mg':'magnesium', 'Al':'aluminium', 'Si':'silicon', 'P':'phosphorus',
                'S':'sulphur', 'Cl':'chlorine', 'Ar':'argon', 'K':'potassium', 'Ca':'calcium',
                'Sc':'scandium', 'Ti':'titanium', 'V':'vanadium', 'Cr':'chromium', 'Mn':'manganese',
                'Fe':'iron', 'Co':'cobalt', 'Ni':'nickel', 'Cu':'copper', 'Zn':'zinc'}
element_symbols = dict((reversed(item) for item in element_names.items())) #reverse dictionary mapping e.g. 'hydrogen'->'H'

#The index no. until which the Cloudy .en and NIST energies agree. After that they will start to diverge (manually confirmed).
#Some species are missing as they have no lines (with necessary coefficients) in the NIST database and so there's no use saving their densities
species_enlim = {'H':21,
                'He':43, 'He+':55,
                'Li':15, 'Li+':9, 'Li+2':15,
                'Be':15, 'Be+':15, 'Be+2':9, 'Be+3':15,
                'B':15, 'B+':15, 'B+2':15, 'B+3':13, 'B+4':15,
                'C':15, 'C+':15, 'C+2':15, 'C+3':15, 'C+4':31, 'C+5':15,
                'N':50, 'N+':15, 'N+2':15, 'N+3':15, 'N+4':15, 'N+5':8, 'N+6':15,
                'O':29, 'O+':15, 'O+2':15, 'O+3':15, 'O+4':15, 'O+5':15, 'O+6':8, 'O+7':15,
                'F+':7, 'F+2':15, 'F+3':15, 'F+4':5, 'F+5':15, 'F+6':15, 'F+7':8, 'F+8':15,
                'Ne':15, 'Ne+':2, 'Ne+2':14, 'Ne+3':15, 'Ne+4':15, 'Ne+5':15, 'Ne+6':15, 'Ne+7':15, 'Ne+8':8, 'Ne+9':15,
                'Na':15, 'Na+':15, 'Na+2':3, 'Na+3':9, 'Na+4':13, 'Na+5':9, 'Na+6':15, 'Na+7':10, 'Na+8':15, 'Na+9':8, 'Na+10':15,
                'Mg':15, 'Mg+':15, 'Mg+2':15, 'Mg+3':3, 'Mg+4':15, 'Mg+5':15, 'Mg+6':15, 'Mg+7':15, 'Mg+8':13, 'Mg+9':9, 'Mg+10':8, 'Mg+11':15,
                'Al':15, 'Al+':15, 'Al+2':15, 'Al+3':15, 'Al+4':3, 'Al+5':15, 'Al+6':15, 'Al+7':9, 'Al+8':5, 'Al+9':10, 'Al+10':15, 'Al+11':8, 'Al+12':15,
                'Si':15, 'Si+':15, 'Si+2':15, 'Si+3':15, 'Si+4':15, 'Si+5':8, 'Si+6':15, 'Si+7':15, 'Si+8':15, 'Si+9':15, 'Si+10':12, 'Si+11':15, 'Si+12':8,
                'P':15, 'P+':15, 'P+2':8, 'P+3':15, 'P+4':15, 'P+5':15, 'P+6':3, 'P+7':10, 'P+8':15, 'P+9':15, 'P+10':15, 'P+11':10, 'P+12':15,
                'S':15, 'S+':15, 'S+2':15, 'S+3':15, 'S+4':15, 'S+5':15, 'S+6':15, 'S+7':3, 'S+8':10, 'S+9':15, 'S+10':15, 'S+11':15, 'S+12':13,
                'Cl':15, 'Cl+':5, 'Cl+2':5, 'Cl+3':5, 'Cl+4':15, 'Cl+5':15, 'Cl+6':5, 'Cl+7':1, 'Cl+8':15, 'Cl+9':8,
                'Ar':15, 'Ar+':15, 'Ar+2':15, 'Ar+3':15, 'Ar+4':15, 'Ar+5':15, 'Ar+6':14, 'Ar+7':15, 'Ar+8':15,
                'K':15, 'K+':15, 'K+2':15, 'K+3':15, 'K+4':5, 'K+5':5, 'K+6':2, 'K+7':15, 'K+8':15, 'K+9':15, 'K+10':3, 'K+11':10, 'K+12':15,
                'Ca':15, 'Ca+':15, 'Ca+2':15, 'Ca+3':15, 'Ca+4':5, 'Ca+5':15, 'Ca+6':15, 'Ca+7':12, 'Ca+8':15, 'Ca+9':15, 'Ca+10':2,
                'Sc':15, 'Sc+':15, 'Sc+2':15, 'Sc+3':15, 'Sc+4':15, 'Sc+6':15, 'Sc+7':15, 'Sc+8':2, 'Sc+9':15, 'Sc+10':15, 'Sc+11':12, 'Sc+12':12,
                'Ti':1, 'Ti+':1, 'Ti+2':15, 'Ti+3':15, 'Ti+5':2, 'Ti+7':15, 'Ti+8':15, 'Ti+9':2, 'Ti+10':14, 'Ti+11':15, 'Ti+12':15,
                'V':1, 'V+':1, 'V+2':1, 'V+7':15, 'V+8':15, 'V+9':15, 'V+10':2, 'V+11':15, 'V+12':15,
                'Cr':1, 'Cr+':15, 'Cr+9':15, 'Cr+10':15, 'Cr+11':15, 'Cr+12':14,
                'Mn':15, 'Mn+':1, 'Mn+10':15, 'Mn+11':15, 'Mn+12':15,
                'Fe':15, 'Fe+':80, 'Fe+2':25, 'Fe+4':25, 'Fe+6':9, 'Fe+10':9, 'Fe+11':12, 'Fe+12':9,
                'Co':1, 'Co+':15, 'Co+2':15, 'Co+12':15,
                'Ni':15, 'Ni+':15, 'Ni+2':15, 'Ni+4':15, 'Ni+12':15,
                'Cu':15, 'Cu+':1,
                'Zn':1}


def get_specieslist(max_ion=6, exclude_elements=[]):
    '''
    Returns a list of atomic and ionic species names. Default returns all species up to 6+
    for which sunbather can do useful things (=NIST has lines). Higher than 6+ ionization is rarely
    attained in an exoplanet atmosphere, but it can definitely occur when using a high XUV
    flux (e.g. in a young system). 
    
    max_ion:            maximum ionization degree of included species
    exclude_elements:   list of elements to exclude (both in atomic and ionic form)
    '''

    if max_ion > 12:
        print("tools.get_specieslist(): You have set max_ion > 12, but " \
              "sunbather is currently only able to process species up to 12+ ionzed. " \
              "This should however be enough even when using a strong XUV flux.")

    if isinstance(exclude_elements, str): #turn into list with one element
        exclude_elements = [exclude_elements]

    specieslist = list(species_enlim.keys()) #all species up to 12+

    for element in exclude_elements:
        specieslist = [sp for sp in specieslist if sp.split('+')[0] != element]

    for sp in specieslist[:]:
        sp_split = sp.split('+')

        if len(sp_split) == 1:
            deg_ion = 0
        elif sp_split[1] == '':
            deg_ion = 1
        else:
            deg_ion = int(sp_split[1])

        if deg_ion > max_ion:
            specieslist.remove(sp)

    return specieslist


def get_mass(species):
    '''
    Returns the mass of an atomic or positive ion in g. For positive ions,
    it just returns the mass of the atom, since the electron mass is negligible.

    species:     name of atom or ion for which to return the mass in g
    '''

    atom = species.split('+')[0]

    mass_dict = {'H':1.6735575e-24, 'He':6.646477e-24, 'Li':1.15e-23, 'Be':1.4965082e-23,
            'B':1.795e-23, 'C':1.9945e-23, 'N':2.3259e-23, 'O':2.6567e-23,
            'F':3.1547e-23, 'Ne':3.35092e-23, 'Na':3.817541e-23, 'Mg':4.0359e-23,
            'Al':4.48038988e-23, 'Si':4.6636e-23, 'P':5.14331418e-23, 'S':5.324e-23,
            'Cl':5.887e-23, 'Ar':6.6335e-23, 'K':6.49243e-23, 'Ca':6.6551e-23,
            'Sc':7.4651042e-23, 'Ti':7.9485e-23, 'V':8.45904e-23, 'Cr':8.63416e-23,
            'Mn':9.1226768e-23, 'Fe':9.2733e-23, 'Co':9.786087e-23, 'Ni':9.74627e-23,
            'Cu':1.0552e-22, 'Zn':1.086e-22} #g

    return mass_dict[atom]





'''
Functions that deal with processing Cloudy's output files
'''

def process_continuum(filename, nonzero=False):
    '''
    This function reads a .con file from the 'save continuum' command.
    It renames the columns and adds a wav column. The flux units of the continuum
    can be tricky to understand, but they are found as follows:
    Take the SED in spectral flux density, so F(nu) instead of nu*F(nu), and
    find the total area by integration. Then multiply with the frequency,
    to get nu*F(nu), and normalize that by the total area found, and multiply
    with the total luminosity. Those are the units of Cloudy.

    filename:       filename (including full path to it and the .con extension)
    nonzero:        if True, removes the rows from the dataframe where the incident
                    spectrum is 0 (i.e., not defined)
    '''

    con_df = pd.read_table(filename)
    con_df.rename(columns={'#Cont  nu':'nu', 'net trans':'nettrans'}, inplace=True)
    wav = c * 1e8 / con_df.nu #wav in AA
    con_df.insert(1, "wav", wav)
    if nonzero:
        con_df = con_df[con_df.incident != 0]
    return con_df


def process_heating(filename, Rp=None, altmax=None):
    '''
    This function reads a .heat file from the 'save heating' command.
    If Rp and altmax are given, it adds an altitude/radius scale.
    For each unique heating agent, it adds a column with its rate at each radial bin.

    filename:       filename (including full path to it and the .heat extension)
    Rp:             planet radius in cm
    altmax:         maximum radius of the simulation in units of Rp
    '''

    #determine max number of columns (otherwise pd.read_table assumes it is the number of the first row)
    max_columns = 0
    with open(filename, 'r') as file:
        for line in file:
            num_columns = len(line.split('\t'))
            max_columns = max(max_columns, num_columns)
    #set up the column names
    fixed_column_names = ['depth', 'temp', 'htot', 'ctot']
    num_additional_columns = (max_columns - 4) // 2
    additional_column_names = [f'htype{i}' for i in range(1, num_additional_columns + 1) for _ in range(2)]
    additional_column_names[1::2] = [f'hfrac{i}' for i in range(1, num_additional_columns + 1)]
    all_column_names = fixed_column_names + additional_column_names
    heat = pd.read_table(filename, delimiter='\t', skiprows=1, header=None, names=all_column_names)

    if heat['depth'].eq("#>>>>  Ionization not converged.").any():
        print(f"WARNING: the simulation you are reading in exited OK but does contain ionization convergence failures: {filename[:-5]}")
        heat = heat[heat['depth'] != "#>>>>  Ionization not converged."] #remove those extra lines from the heat DataFrame

    #remove the "second rows", which sometimes are in the .heat file and do not give the heating at a given depth
    if type(heat.depth.iloc[0]) == str: #in some cases there are no second rows
        heat = heat[heat.depth.map(len)<12] #delete second rows
    
    heat.depth = pd.to_numeric(heat.depth) #str to float
    heat.reset_index(drop=True, inplace=True) #reindex so that it has same index as e.g. .ovr

    if Rp != None and altmax != None: #add altitude scale
        heat['alt'] = altmax * Rp - heat.depth

    agents = []
    for column in heat.columns:
        if column.startswith('htype'):
            agents.extend(heat[column].unique())
    agents = list(set(agents)) #all unique heating agents that appear somewhere in the .heat file

    for agent in agents:
        heat[agent] = np.nan #add 'empty' column for each agent

    #now do a (probably sub-optimal) for-loop over the whole df to put all hfracs in the corresponding column
    htypes = [f'htype{i+1}' for i in range(num_additional_columns)]
    hfracs = [f'hfrac{i+1}' for i in range(num_additional_columns)]
    for htype, hfrac in zip(htypes, hfracs):
        for index, agent in heat[htype].items():
            rate = heat.loc[index, hfrac]
            heat.loc[index, agent] = rate

    if np.nan in heat.columns: #sometimes columns are partially missing, resulting in columns called nan
        heat.drop(columns=[np.nan], inplace=True)

    heat['sumfrac'] = heat.loc[:,[col for col in heat.columns if 'hfrac' in col]].sum(axis=1)

    return heat


def process_cooling(filename, Rp=None, altmax=None, cloudy_version="17"):
    '''
    This function reads a .cool file from the 'save cooling' command.
    If Rp and altmax are given, it adds an altitude/radius scale.
    For each unique cooling agent, it adds a column with its rate at each radial bin.

    filename:       filename (including full path to it and the .cool extension)
    Rp:             planet radius in cm
    altmax:         maximum radius of the simulation in units of Rp
    '''

    #determine max number of columns (otherwise pd.read_table assumes it is the number of the first row)
    max_columns = 0
    with open(filename, 'r') as file:
        for line in file:
            num_columns = len(line.split('\t'))
            max_columns = max(max_columns, num_columns)
    #set up the column names
    if cloudy_version == "17":
        fixed_column_names = ['depth', 'temp', 'htot', 'ctot']
    elif cloudy_version == "23":
        fixed_column_names = ['depth', 'temp', 'htot', 'ctot', 'adv']
    else:
        raise Exception("Only C17.02 and C23.01 are currently supported.")
    num_additional_columns = (max_columns - 4) // 2
    additional_column_names = [f'ctype{i}' for i in range(1, num_additional_columns + 1) for _ in range(2)]
    additional_column_names[1::2] = [f'cfrac{i}' for i in range(1, num_additional_columns + 1)]
    all_column_names = fixed_column_names + additional_column_names
    cool = pd.read_table(filename, delimiter='\t', skiprows=1, header=None, names=all_column_names)
    
    if cool['depth'].eq("#>>>>  Ionization not converged.").any():
        print(f"WARNING: the simulation you are reading in exited OK but does contain ionization convergence failures: {filename[:-5]}")
        #remove those extra lines from the cool DataFrame
        cool = cool[cool['depth'] != "#>>>>  Ionization not converged."]
        cool['depth'] = cool['depth'].astype(float)
        cool = cool.reset_index(drop=True) #so it matches other dfs like .ovr
    

    if Rp != None and altmax != None: #add altitude scale
        cool['alt'] = altmax * Rp - cool.depth

    agents = []
    for column in cool.columns:
        if column.startswith('ctype'):
            agents.extend(cool[column].unique())
    agents = list(set(agents)) #all unique cooling agents that appear somewhere in the .cool file

    for agent in agents:
        cool[agent] = np.nan #add 'empty' column for each agent

    #now do a (probably sub-optimal) for-loop over the whole df to put all cfracs in the corresponding column
    ctypes = [f'ctype{i+1}' for i in range(num_additional_columns)]
    cfracs = [f'cfrac{i+1}' for i in range(num_additional_columns)]
    for ctype, cfrac in zip(ctypes, cfracs):
        for index, agent in cool[ctype].items():
            rate = cool.loc[index, cfrac]
            cool.loc[index, agent] = rate

    if np.nan in cool.columns: #sometimes columns are partially missing, resulting in columns called nan
        cool.drop(columns=[np.nan], inplace=True)

    cool['sumfrac'] = cool.loc[:,[col for col in cool.columns if 'cfrac' in col]].sum(axis=1)

    return cool


def process_coolingH2(filename, Rp=None, altmax=None):
    '''
    This function reads a .coolH2 file from the 'save H2 cooling' command,
    which keeps track of cooling and heating processes unique to the
    H2 molecule, when using the 'database H2' command.

    From the source code "mole_h2_io.cpp" the columns are:

    depth, Temp, ctot/htot, H2 destruction rate Solomon TH85,
    H2 destruction rate Solomon big H2, photodis heating,
    heating dissoc. electronic exited states,
    cooling collisions in X (neg = heating),
    "HeatDexc"=net heat, "-HeatDexc/abundance"=net cool per particle

    filename:       filename (including full path to it and the .coolH2 extension)
    Rp:             planet radius in cm
    altmax:         maximum radius of the simulation in units of Rp
    '''

    coolH2 = pd.read_table(filename, names=['depth', 'Te', 'ctot', 'desTH85',
                            'desbigH2', 'phdisheat', 'eedisheat', 'collcool',
                            'netheat', 'netcoolpp'], header=1)
    if Rp != None and altmax != None:
        coolH2['alt'] = altmax*Rp - coolH2['depth']

    return coolH2


def process_overview(filename, Rp=None, altmax=None, abundances=None):
    '''
    This function reads in a '.ovr' file from the 'save overview' command.
    If Rp and altmax are given, it adds an altitude/radius scale.
    It also adds the mass density (in addition to the hydrogen number density).
    If the simulation has non-solar/default abundances, they must be specified
    as an abundances dictionary made with the get_abundances() function, in
    order for the hden -> rho conversion to be correct.

    filename:       filename (including full path to it and the .ovr extension)
    Rp:             planet radius in cm
    altmax:         maximum radius of the simulation in units of Rp
    abundances:     dictionary of the abundances of each element.
                    Can be made with the get_abundances() function.
    '''

    ovr = pd.read_table(filename)
    ovr.rename(columns={'#depth':'depth'}, inplace=True)
    ovr['rho'] = hden_to_rho(ovr.hden, abundances=abundances) #Hdens to total dens
    if Rp != None and altmax != None:
        ovr['alt'] = altmax * Rp - ovr['depth']
    ovr['mu'] = calc_mu(ovr.rho, ovr.eden, abundances=abundances)

    return ovr


def process_densities(filename, Rp=None, altmax=None):
    '''
    This function reads a .den file from the 'save species densities' command.
    If Rp and altmax are given, it adds an altitude/radius scale.

    filename:       filename (including full path to it and the .den extension)
    Rp:             planet radius in cm
    altmax:         maximum radius of the simulation in units of Rp    
    '''

    den = pd.read_table(filename)
    den.rename(columns={'#depth densities':'depth'}, inplace=True)

    if Rp != None and altmax != None:
        den['alt'] = altmax*Rp - den['depth']

    return den


def process_energies(filename, rewrite=True):
    '''
    This function reads a '.en' file from the 'save species energies' command.
    ALWAYS use that command alongside the 'save species densities' .den files,
    since they give the associated energy of each level printed in the
    densities file. Otherwise, it's not clear which level exactly is He[52]
    for example. This function returns a dictionary with the column names of
    the .den file and the corresponding energies. This can then be used for
    radiative transfer. The lines data from the NIST database gives
    the energy of the lower level for each line. The 'find_Ei_in_en_dict()'
    function will then search the energy dictionary generated by
    this function for the corresponding column name, such that the number
    densities of the right level can be extracted from the .den file.

    filename:       filename (including full path to it and the .en extension)
    rewrite:        the .en file usually has a large file size. However,
                    it contains many identical rows since Cloudy gives the energies
                    at each depth bin into the simulation, while they should
                    be the same at each depth. So if rewrite is True, 
                    the function veries that the energies are indeed the same in each row,
                    and then rewrites the .en file with only the first row.
    '''

    en = pd.read_table(filename, float_precision='round_trip') #use round_trip to prevent exp numerical errors

    if en.columns.values[0][0] == '#': #condition checks whether it has already been rewritten, if not, we do all following stuff:

        for col in range(len(en.columns)): #check if all rows are the same
            if len(en.iloc[:,col].unique()) != 1:
                raise Exception("In reading .en file, found a column with not identical values!"
                        +" filename:", filename, "col:", col, "colname:", en.columns[col], "unique values:",
                        en.iloc[:,col].unique())

        en.rename(columns={en.columns.values[0] : en.columns.values[0][10:]}, inplace=True) #rename the column

        if rewrite: #save with only first row to save file size
            en.iloc[[0],:].to_csv(filename, sep='\t', index=False, float_format='%.5e')

    en_df = pd.DataFrame(index = en.columns.values)
    en_df['species'] = [k.split('[')[0] for k in en_df.index.values] #we want to match 'He12' to species='He', for example
    en_df['energy'] = en.iloc[0,:].values
    en_df['configuration'] = ""
    en_df['term'] = ""
    en_df['J'] = ""


    #the & set action takes the intersection of all unique species of the .en file, and those known with NIST levels
    unique_species = list(set(en_df.species.values) & set(species_enlim.keys()))

    for species in unique_species:
        species_levels = pd.read_table(sunbather_path+'/RT_tables/'+species+'_levels_processed.txt') #get the NIST levels
        species_energies = en_df[en_df.species == species].energy #get Cloudy's energies

        atol = 0.001
        #tolerance of difference between Cloudy's and NISTs energy levels. They usually differ at the decimal level so we need some tolerance.
        #For some species, the differences are a bit bigger. Probably a NIST update or something? I have manually verfied that these energy levels
        #still probably represent the same atomic configuration. So we relax the threshold here:
        if species in ['B+4', 'N+6', 'C+4', 'B+3', 'Cl+6', 'Be+3', 'Si+2', 'C+5', 'Li+2', 'Ti+8', 
                      'Si+11', 'O+7', 'Fe+11', 'Ca+7', 'Ni+12', 'K+11', 'Ca+10', 'Ti+10', 'Cr+12']:
            atol = 0.01
        if species in ['Ne+6', 'Ar+7', 'Ne+9', 'F+8', 'S+12', 'Si+10', 'Si+7', 'Al+12', 'Na+10', 'Mg+11']:
            atol = 0.05
        if species in ['Ar+8']:
            atol = 0.2

        n_matching = species_enlim[species] #start by assuming we can match this many energy levels - which we in principle should for C17.02

        for n in range(n_matching):
            if not np.abs(species_energies.iloc[n] - species_levels.energy.iloc[n]) < atol:
                n_matching = n

                print(f"WARNING: In {filename} while getting atomic states for species {species}, I expected to be able to match the first {species_enlim[species]} " + \
                    f"energy levels between Cloudy and NIST to a precision of {atol} (for C17.02) but I have an energy mismatch at energy level {n_matching+1}. " + \
                    f"This should not introduce bugs, as I will now only parse the first {n_matching} levels.")
                
                #for debugging, you can print the energy levels of Cloudy and NIST:
                #print("\nCloudy, NIST, Match?")
                #for i in range(species_enlim[species]):
                #    print(species_energies.iloc[i], species_levels.energy.iloc[i], np.isclose(species_energies.iloc[:species_enlim[species]], species_levels.energy.iloc[:species_enlim[species]], rtol=0.0, atol=atol)[i])

                break

        #Now assign the first n_matching columns to their expected values as given by the NIST species_levels DataFrame
        first_iloc = np.where(en_df.species == species)[0][0] #iloc at which the species (e.g. He or Ca+3) starts.
        en_df.iloc[first_iloc:first_iloc+n_matching, en_df.columns.get_loc('configuration')] = species_levels.configuration.iloc[:n_matching].values
        en_df.iloc[first_iloc:first_iloc+n_matching, en_df.columns.get_loc('term')] = species_levels.term.iloc[:n_matching].values
        en_df.iloc[first_iloc:first_iloc+n_matching, en_df.columns.get_loc('J')] = species_levels.J.iloc[:n_matching].values
    
    return en_df


def find_line_lowerstate_in_en_df(species, lineinfo, en_df, printmessage=True):
    '''
    Also see process_energies() docstring.

    This function finds the column name of the .den file that corresponds to
    the ground state of the given line. So for example if species='He',
    and we are looking for the metastable helium line,
    it will return 'He2', meaning the 'He2' column of the .den file contains
    the number densities of the metastable helium atom.

    species:        atomic or ionic species name
    lineinfo:       one row of a dataframe of NIST spectral line coefficients
    en_df:          dataframe mapping the .den / .en column names to the energy of that level
    printmessage:   whether to print problems when trying to match the line
    '''

    en_df = en_df[en_df.species == species] #keep only the part for this species to not mix up the energy levels of different ones
    match, lineweight = None, None #start with the assumption that we cannot match it

    #check if the line originates from a J sublevel, a term, or only principal quantum number
    if str(lineinfo['term_i']) != 'nan' and str(lineinfo['J_i']) != 'nan':
        linetype = 'J' #then now match with configuration and term:
        matchedrow = en_df[(en_df.configuration == lineinfo.conf_i) & (en_df.term == lineinfo.term_i) & (en_df.J == lineinfo.J_i)]
        assert len(matchedrow) <= 1

        if len(matchedrow) == 1:
            match = matchedrow.index.item()
            lineweight = 1. #since the Cloudy column is for this J specifically, we don't need to downweigh the density

        elif len(matchedrow) == 0:
            #the exact J was not found in Cloudy's levels, but maybe the term is there in Cloudy, just not resolved.
            matchedtermrow = en_df[(en_df.configuration == lineinfo.conf_i) & (en_df.term == lineinfo.term_i)]

            if len(matchedtermrow) == 1:
                if str(matchedtermrow.J.values[0]) == 'nan': #this can only happen if the Cloudy level is a term with no J resolved.
                    #then we use statistical weights to guess how many of the atoms in this term state would be in the J state of the level and use this as lineweight
                    L = Ldict[''.join(x for x in matchedtermrow.loc[:,'term'].item() if x.isalpha())[-1]] #last letter in term string
                    S = (float(re.search(r'\d+', matchedtermrow.loc[:,'term'].item()).group())-1.)/2. #first number in term string
                    J_states = np.arange(np.abs(L-S), np.abs(L+S)+1, 1.0)
                    J_statweights = 2*J_states + 1
                    J_probweights = J_statweights / np.sum(J_statweights)

                    lineweight = J_probweights[J_states == Fraction(lineinfo.loc['J_i'])][0]

                    match = matchedtermrow.index.item()
                else:
                    if printmessage:
                        print("One J level of the term is resolved, but not the one of this line.")

            else:
                if printmessage:
                    print("Multiple J levels of the term are resolved, but not the one of this line.")

    elif str(lineinfo['term_i']) != 'nan':
        linetype = "LS"

        if printmessage:
            print("Currently not able to do lines originating from LS state without J number.")
            print("Lower state configuration:", species, lineinfo.conf_i)
    else:
        linetype = "n"

        if printmessage:
            print("Currently not able to do lines originating from n state without term. This is not a problem "+
                    'if this line is also in the NIST database with its different term components, such as for e.g. '+
                    "H n=2, but only if they aren't such as for H n>6, or if they go to an upper level n>6 from any given level.")
            print("Lower state configuration:", species, lineinfo.conf_i)

        '''
        NOTE TO SELF:
        If I do decide to make this functionality, for example by summing the densities of all sublevels of a
        particular n, I also need to tweak the cleaning of hydrogen lines algorithm. Right now, I remove
        double lines only for the upper state, so e.g. for Ly alpha, I remove the separate 2p 3/2 and 2p 1/2 etc. component
        and leave only the one line with upper state n=2.
        I don't do this for lower states though, which is not a problem yet because the lower n state lines are ignored as
        stated above. However if I make the functionality, I should also remove double lines in the lower level.
        '''

    return match, lineweight




'''
Miscellaneous functions
'''

def get_SED_norm_1AU(SEDname):
    '''
    Reads in a SED name in the data/SED/ folder of the Cloudy installation,
    and returns the normalization in nuF(nu) and Ryd units.
    It must then be scaled to the planet distance and the
    log10 taken, after which it can be used with the nuF(nu)= ... at ... Ryd command.
    Assumes (and checks) that the SED is in wav (Å) and nuFnu/lambFlamb (erg/s/cm-2) units.

    SEDname:    name of the SED including extension but excluding the path.
                The SED is expected to be present in the /c17.02/data/SED/ folder.
    '''

    with open(cloudypath+'/data/SED/'+SEDname, 'r') as f:
        for line in f:
            if not line.startswith('#'): #skip through the comments at the top
                assert ('angstrom' in line) or ('Angstrom' in line) #verify the units
                assert 'nuFnu' in line #verify the units
                break
        data = np.genfromtxt(f, skip_header=1) #skip first line, which has extra words specifying the units

    ang, nuFnu = data[-2,0], data[-2,1] #read out intensity somewhere
    Ryd = 911.560270107676 / ang #convert wavelength in Å to energy in Ryd

    return nuFnu, Ryd


def speciesstring(specieslist, selected_levels=False):
    '''
    Takes a list of species names and returns a long string with those species
    between quotes and [:] added (or [maxlevel] if selected_levels=True),
    and \n between them, so that this string can be used in a Cloudy input
    script for .den and .en files. The maxlevel is the number of energy levels
    that can be matched between Cloudy and NIST. Saving the higher levels is not
    really of use since they can't be postprocessed by the radiative transfer module.

    specieslist:        list of atomic and ionic species
    selected_levels:    if False, will use all energy levels, if True will
                        use only up to the energy level that can be matched to NIST
    '''

    if not selected_levels: #so just all levels available in cloudy
        speciesstr = '"'+specieslist[0]+'[:]"'
        if len(specieslist) > 1:
            for species in specieslist[1:]:
                speciesstr += '\n"'+species+'[:]"'

    elif selected_levels: #then we read out the max level from the species_enlim dictionary
        speciesstr = '"'+specieslist[0]+'[:'+str(species_enlim[specieslist[0]])+']"'
        if len(specieslist) > 1:
            for species in specieslist[1:]:
                speciesstr += '\n"'+species+'[:'+str(species_enlim[species])+']"'

    return speciesstr


def read_parker(plname, T, Mdot, pdir, filename=None):
    '''
    Reads a parker wind profile and returns it as a pandas Dataframe.
    Arguments:
        plname: [str]       planet name
        T:                  temperature
        Mdot: [float/str]   log10 of the mass-loss rate
        pdir: [str]         folder to use for the parker profile. This can be
                            any folder name as long as it exists. So e.g. you
                            can have a folder with pure H/He profiles named
                            fH=0.9 or fH=0.99, or have a folder with Cloudy-
                            produced parker profiles named z=10.
        filename: [str]     filename to read. If this argument is given then
                            plname, T, Mdot and dir are disregarded and we
                            directly read the specified profile.
    '''

    if filename == None:
        Mdot = "%.3f" % float(Mdot)
        T = str(int(T))
        filename = projectpath+'/parker_profiles/'+plname+'/'+pdir+'/pprof_'+plname+'_T='+T+'_M='+Mdot+'.txt'

    pprof = pd.read_table(filename, names=['alt', 'rho', 'v', 'mu'], dtype=np.float64, comment='#')
    pprof['drhodr'] = np.gradient(pprof['rho'], pprof['alt'])
    return pprof


def calc_mu(rho, ne, abundances=None, mass=False):
    '''
    Calculates the mean molecular weight, taking into account all elements and ions,
    but NEGLECTING MOLECULES and the mass contributed by electrons.

    Based on formula: mu = sum(ni*mi) / (sum(ni) + ne)   where ni and mi are the number density
                                                        and mass of all elements
                        then use ni = ntot * fi   and   ntot = rho / sum(fi*mi)
                        to get:
                        mu = sum(fi*mi) / (1 + (ne * sum(fi*mi))/rho)

    rho:        total mass density in g cm-3
    ne:         electron number density in cm-3
    abundances: dictionary with the abundance of each element made with get_abundances()
    mass:       if False, returns mu in units of amu, if True returns mu in units of g
    '''

    if abundances == None:
        abundances = get_abundances()

    sum_all = 0.
    for element in abundances.keys():
        sum_all += abundances[element] * get_mass(element)

    mu = sum_all / (1 + ne*sum_all / rho) #mu in g
    if not mass:
        mu = mu / mH #mu in amu

    return mu


def get_zdict(z=1., zelem={}):
    '''
    Function that returns a dictionary of the scale factors of each element.
    arguments:
        z:          metallicity relative to solar (in linear units, i.e. z=1 is solar)
                    scales all elements except hydrogen and helium with this factor
        zelem:      dictionary of abundance scale factor for specific elements
                    (e.g. {'C':2} to get double C abundance)
    '''

    assert 'H' not in zelem.keys(), "You cannot scale hydrogen, scale everything else instead."

    zdict = {'He':1., 'Li':z, 'Be':z, 'B':z, 'C':z, 'N':z, 'O':z, 'F':z, 'Ne':z,
    'Na':z, 'Mg':z, 'Al':z, 'Si':z, 'P':z, 'S':z, 'Cl':z, 'Ar':z, 'K':z, 'Ca':z,
    'Sc':z, 'Ti':z, 'V':z, 'Cr':z, 'Mn':z, 'Fe':z, 'Co':z, 'Ni':z, 'Cu':z, 'Zn':z}

    for element in zelem.keys():
        zdict[element] *= zelem[element]

    return zdict


def get_abundances(zdict=None):
    '''
    Function that returns a dictionary of the fractional abundances of each element (sums to 1).
    arguments:
        zdict:      dictionary of all fractional abundance scale factors made with get_zdict()
    '''

    #solar abundance relative to hydrogen (Hazy table 7.1):
    rel_abundances = {'H':1., 'He':0.1, 'Li':2.04e-9, 'Be':2.63e-11, 'B':6.17e-10,
    'C':2.45e-4, 'N':8.51e-5, 'O':4.9e-4, 'F':3.02e-8, 'Ne':1e-4,
    'Na':2.14e-6, 'Mg':3.47e-5, 'Al':2.95e-6, 'Si':3.47e-5, 'P':3.2e-7,
    'S':1.84e-5, 'Cl':1.91e-7, 'Ar':2.51e-6, 'K':1.32e-7, 'Ca':2.29e-6,
    'Sc':1.48e-9, 'Ti':1.05e-7, 'V':1e-8, 'Cr':4.68e-7, 'Mn':2.88e-7,
    'Fe':2.82e-5, 'Co':8.32e-8, 'Ni':1.78e-6, 'Cu':1.62e-8, 'Zn':3.98e-8}

    if zdict != None:
        assert 'H' not in zdict.keys(), "You cannot scale hydrogen, scale everything else instead."
        for element in zdict.keys():
            rel_abundances[element] *= zdict[element]

    total = sum(list(rel_abundances.values()))
    abundances = {k: v / total for k, v in rel_abundances.items()}

    return abundances


def rho_to_hden(rho, abundances=None):
    '''
    Converts a total density rho in g/cm3 to a hydrogen number density in cm-3

    Based on formula: rho = nH*mH + ntot*sum(fj*mj) where fj is the abundance(fraction) of element j (excluding H)
                                                    and ntot=rho/sum(fi*mi) where i is every element (including H)
            resulting in:
                    nH = rho/mH * (1 - sum(fj*mj)/sum(fi*mi))

    rho:        total mass density in g cm-3
    abundances: dictionary with the abundance of each element made with get_abundances()
    '''

    if abundances == None:
        abundances = get_abundances() #get a solar composition

    sum_all = 0.
    for element in abundances.keys():
        sum_all += abundances[element] * get_mass(element)

    sum_noH = sum_all - abundances['H'] * get_mass('H') #subtract hydrogen to get the sum without H

    hden = rho/mH * (1 - sum_noH / sum_all)

    return hden


def hden_to_rho(hden, abundances=None):
    '''
    Converts a hydrogen number density in cm-3 to a total density rho in g/cm3
    See rho_to_hden() for formula, just reversed here.

    hden:       hydrogen number density in cm-3
    abundances: dictionary with the abundances of each element made with get_abundances()
    '''

    if abundances == None:
        abundances = get_abundances() #get a solar composition

    sum_all = 0.
    for element in abundances.keys():
        sum_all += abundances[element] * get_mass(element)

    sum_noH = sum_all - abundances['H'] * get_mass('H') #subtract hydrogen to get the sum without H

    rho = hden*mH / (1 - sum_noH / sum_all)

    return rho


def roche_radius(a, Mp, Mstar):
    '''
    Returns the Roche / Hill radius.
    This is a small planet-to-star mass approximation.

    a:   semi-major axis of the planet in cm
    Mp:            planet mass in g
    Mstar:          star mass in g
    '''

    return a * pow(Mp/(3.0*(Mstar+Mp)), 1.0/3.0)


def set_alt_ax(ax, altmax=8, labels=True):
    '''
    This code takes in an axis object and sets the xscale to a log scale
    with readable ticks in non-scientific notation, in units of planet radii.

    ax:         matplotlib.Axes object
    altmax:     maximum radius in units of Rp
    labels:     whether to set an xlabel and xticklabels
    '''

    ax.set_xscale('log')
    ax.set_xlim(1, altmax)
    ticks = np.concatenate((np.arange(1, 2, 0.1), np.arange(2, altmax+1, 1)))
    if altmax <= 3:
        ticklabels = ['1', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9']
        ticklabels2 = ["%i" %t for t in np.arange(2, altmax+1, 1).astype(int)]
    elif altmax <= 10:
        ticklabels = ['1', '', '', '', '', '1.5', '', '', '', '']
        ticklabels2 = ["%i" %t for t in np.arange(2, altmax+1, 1).astype(int)]
    elif altmax <= 14:
        ticklabels = ['1', '', '', '', '', '', '', '', '', '', '2', '3', '4', '5', '', '7', '', '', '10']
        ticklabels2 = ['']*(altmax-10)
    else:
        ticklabels = ['1', '', '', '', '', '', '', '', '', '', '2', '3', '4', '5', '', '7', '', '', '10']
        ticklabels2 = ['']*(altmax-10)
        ticklabels2b = np.arange(15, altmax+0.1, 5).astype(int)
        index = 4
        for t2b in ticklabels2b:
            ticklabels2[index] = str(t2b)
            index += 5

    ticklabels = ticklabels + ticklabels2

    ax.set_xticks(ticks)
    if labels:
        ax.set_xticklabels(ticklabels)
        ax.set_xlabel(r'Radius [$R_p$]')
    else:
        ax.set_xticklabels([])


def cl_table(r, rho, v, altmax, Rp, nmax, negtozero=False, zdict=None):
    '''
    This function converts a density structure to the format read by Cloudy.
    It also calculates expansion/(T/mu) cooling rates for this density and
    velocity structure and returns them in Cloudy's input format, as well
    as advection/(d(T/mu)/dr) rates in Cloudy's format.
    These rates can then be multiplied with T/mu and d(T/mu)/dr, 
    as provided by a Cloudy simulation, respectively,
    to obtain the full expansion and advection rate.

    r:          radius from the planet center in cm
    rho:        total mass density in g cm-3
    v:          outflow velocity in cm s-1
    altmax:     maximum radius of the grid in units of Rp
    Rp:         planet radius in cm
    nmax:       number of points to use for the grid
    negtozero:  whether to set negative PdV rates (which should not happen)
                manually to zero. Negative rates are sometimes present
                when postprocessing other HD codes output with numerical errors
                and negative velocity values.
    zdict:      dictionary of all fractional abundance scale factors made with get_zdict().
                Needed to convert rho to a hydrogen number density.
    '''

    assert isinstance(r, np.ndarray), "Give r as numpy array (= pd.Series.values)"
    assert isinstance(rho, np.ndarray), "Give rho as numpy array (= pd.Series.values)"
    assert isinstance(v, np.ndarray), "Give v as numpy array (= pd.Series.values)"
    assert len(r) == len(rho) and len(r) == len(v), "Arrays don't have the same length."

    if altmax > r[-1]/Rp:
        altmax = r[-1]
        print("Altmax is higher than the max of the model.\n"
                +"Continuing with max of model:", altmax)

    PdVT = calc_expansionTmu(r, rho, v)

    if len(PdVT[PdVT < 0] > 0): #if there are negative values (heating)
        if negtozero: #if we're aware of those, pass True and we continue
            print("I found negative cooling rates, which probably indicates "
                    +"negative velocities. Are you sure you're not making "
                    +"mistakes?\nI'll set those cooling rates to zero.")
            PdVT[PdVT < 0] = 0
        else: #otherwise, we're probably not aware and will stop here.
            raise ValueError("I found negative cooling rates, which probably indicates"
                    +"negative velocities. Are you sure you're not making"
                    +"mistakes?\nI'll abort now.")

    advec = v*rho*(3/2)*k/mH #erg / s / cm2 / T, multiply with d(T/mu)/dR  in Cloudy

    #interpolation functions.
    ifunc_rho = interp1d(r, rho, kind='linear', fill_value='extrapolate')
    ifunc_PdVT = interp1d(r, PdVT, kind='linear', bounds_error=False, fill_value=0.)
    ifunc_advec = interp1d(r, advec, kind='linear', bounds_error=False, fill_value=0.)

    #altitude log-spaced grid
    ialt = np.logspace(np.log10(Rp), np.log10(altmax*Rp), num=int(0.7*nmax))
    #sample the illuminated side better to prevent Cloudy log-interpolation errors
    r_ill1 = (ialt[-1] - ialt)[::-1]
    r_ill2 = np.logspace(-2, np.log10(r_ill1[9]), num=(nmax-len(ialt)))
    r_ill = np.concatenate((r_ill2, r_ill1[10:]))
    ialt = ialt[-1] - r_ill[::-1]

    ialt[0] = Rp #prevent float precision errors

    #get quantities on the new log-spaced grid
    ihden = rho_to_hden(ifunc_rho(ialt), abundances=get_abundances(zdict)) #from rho to H number density
    iPdVT = ifunc_PdVT(ialt)
    iadvec = ifunc_advec(ialt)

    #flip the r-scales such that we go from the illuminated face inwards
    hden_ill = ihden[::-1]
    PdVT_ill = iPdVT[::-1]
    advec_ill = iadvec[::-1]

    r_ill[0] = 1e-35 #first point cannot be zero if we take log
    PdVT_ill[PdVT_ill == 0] = 1e-35 #set zero cooling to very low value because of log
    advec_ill[advec_ill <= 0] = 1e-35 #set negative to small value since we can only have outflow.

    #stack in 2D table that can be used with write_Cloudy_in()
    hdenprof = np.log10(np.column_stack((r_ill, hden_ill)))
    PdVprof = np.log10(np.column_stack((r_ill, PdVT_ill)))
    advecprof = np.log10(np.column_stack((r_ill, advec_ill)))

    return hdenprof, PdVprof, advecprof


def calc_expansionTmu(r, rho, v):
    '''
    Calcules the expansion cooling term / (T/mu), so afterwards you can
    multiply with the T/mu as given by Cloudy to obtain back the full
    expansion cooling rate.
    Requires that r is in the direction of v (i.e. usually in altitude scale).

    r:      radius in the atmosphere in cm
    rho:    density in g cm-3
    v:      velocity in cm s-1
    '''

    expTmu = -1 * np.gradient(rho, r) * v*k/mH #erg / s / cm3
    return expTmu #expTmu as positive values


def alt_array_to_Cloudy(alt, quantity, altmax, Rp, nmax, log=True):
    '''
    Takes as input an atmospheric quantity as a function of altitude/radius,
    and returns it as a 2D array of that same quantity as a function of
    distance from the top of the atmosphere. The latter is the format
    in which Cloudy expects quantites to be given, since it works from the 
    illuminated face of the 'cloud' towards the planet core.

    alt:        altitude/radius grid in units of cm (in ascending order)
    quantity:   array with same length of alt of some atmospheric quantity,
                for example density/temperature
    altmax:     maximum radius of the grid in units of Rp
    Rp:         planet radius in units of cm
    nmax:       number of points to use for the grid
    log:        whether to return the depth and quantity values in log10 units,
                which is what Cloudy expects.
    '''

    if isinstance(alt, pd.Series):
        alt = alt.values
    if isinstance(quantity, pd.Series):
        quantity = quantity.values

    assert alt[1] > alt[0] #should be in ascending alt order
    assert alt[-1] - altmax*Rp > -1. #For extrapolation: the alt scale should extend at least to within 1 cm of altmax*Rp

    if not np.isclose(alt[0], Rp, rtol=1e-2, atol=0.0):
        print("\n(tools.alt_array_to_cloudy): Are you sure the altitude array starts at Rp? alt[0]/Rp =",
                    alt[0]/Rp, "\n")

    depth = altmax*Rp - alt
    ifunc = interp1d(depth, quantity, fill_value='extrapolate')


    Clgridr1 = np.logspace(np.log10(alt[0]), np.log10(altmax*Rp), num=int(0.8*nmax))
    Clgridr1[0], Clgridr1[-1] = alt[0], altmax*Rp #reset these for potential log-numerical errors
    Clgridr1 = (Clgridr1 - Clgridr1[0])
    #sample the first 10 points better since Cloudy messes up with log-space interpolation there
    Clgridr2 = np.logspace(-2, np.log10(Clgridr1[9]), num=(nmax-len(Clgridr1)))
    Clgridr = np.concatenate((Clgridr2, Clgridr1[10:]))
    Clgridr[0] = 1e-35

    Clgridq = ifunc(Clgridr)
    law = np.column_stack((Clgridr, Clgridq))
    if log:
        law[law[:,1]==0., 1] = 1e-100
        law = np.log10(law)

    return law


def project_1D_to_2D(r1, q1, Rp, numb=101, directional=False, cut_at=None, **kwargs):
    '''
    Projects a 1D sub-stellar solution to a 2D symmetric structure,
    e.g. so that we can do RT with it. This function preserves the maximum altitude
    of the 1D ray, so that the 2D output looks like a half circle. However,
    the 2D quantity it outputs is a numpy arrays which must be 'rectangular'
    so it sets values in the rectangle outside of the circle to 0. That will
    ensure 0 density and no optical depth.

    r1:             altitude values from planet core in cm (ascending!)
    q1:             1D quantity to project.
    Rp:             planet core radius in cm. needed because we start there, and not
                    necessarily at the lowest r-value (which may be slightly r[0] != Rp)
    numb:           the number of bins in the y-directtion (impact parameters)
                    twice this number is used in the x-direction (l.o.s.)
    directional:    True or False. Whether the quantity q is directional, so
                    that we need to multiply with a projection factor, and
                    add a minus sign in some bins.
                    + = in direction away from observer (negative x), so e.g. for x-velocity,
                    + values will be redshifted.
    cut_at:          radius at which we 'cut' the 2D structure and set values to 0.
                    e.g. to set density 0 outside roche radius.
    '''

    assert r1[1] > r1[0], "arrays must be in order of ascending altitude"

    b = np.logspace(np.log10(0.1*Rp), np.log10(r1[-1] - 0.9*Rp), num=numb) + 0.9*Rp #impact parameters for 2D rays
    #x = np.linspace(-r1[-1], r1[-1], num=2*numb) #x values for the 2D grid #decrepated linear grid, not 100% if other functions relied on x being spaced equally.
    xos = np.logspace(np.log10(0.101*Rp), np.log10(r1[-1]+0.1*Rp), num=numb) - 0.1*Rp
    x = np.concatenate((-xos[::-1], xos)) #log-spaced grid, innermost point is at 1.001 Rp
    xx, bb = np.meshgrid(x, b)
    rr = np.sqrt(bb**2 + xx**2) #radii from planet core in 2D

    q2 = interp1d(r1, q1, fill_value=0., bounds_error=False)(rr)
    if directional:
        q2 = -q2 * xx / np.sqrt(xx**2 + bb**2) #need to add the minus because q is defined positive at negative x.

    if cut_at != None:
        q2[rr > cut_at] = 0.
    if 'skip_alt_range' in kwargs:
        assert kwargs['skip_alt_range'][0] < kwargs['skip_alt_range'][1]
        q2[(rr > kwargs['skip_alt_range'][0]) & (rr < kwargs['skip_alt_range'][1])] = 0.
    if 'skip_alt_range_dayside' in kwargs:
        assert kwargs['skip_alt_range_dayside'][0] < kwargs['skip_alt_range_dayside'][1]
        q2[(rr > kwargs['skip_alt_range_dayside'][0]) & (rr < kwargs['skip_alt_range_dayside'][1]) & (xx < 0.)] = 0.
    if 'skip_alt_range_nightside' in kwargs:
        assert kwargs['skip_alt_range_nightside'][0] < kwargs['skip_alt_range_nightside'][1]
        q2[(rr > kwargs['skip_alt_range_nightside'][0]) & (rr < kwargs['skip_alt_range_nightside'][1]) & (xx > 0.)] = 0.

    return b, x, q2


def smooth_gaus_savgol(y, size=None, fraction=None):
    '''
    Smooth an array using a gaussian filter, but smooth the edges with a
    savgol filter since otherwise those are not handled well.

    y:          array to smooth
    size:       gaussian filter size as an absolute number
    fraction:   gaussian filter size as a fraction of the array length
    '''

    if size != None and fraction == None:
        size = max(3, size)
    elif fraction != None and size == None:
        assert 0. < fraction < 1., "fraction must be greater than 0 and smaller than 1"
        size = int(np.ceil(len(y)*fraction) // 2 * 2 + 1) #make it odd
        size = max(3, size)
    else:
        raise ValueError("Please provide either 'size' or 'fraction'.")

    ygaus = gaussian_filter1d(y, size)
    ysavgol = savgol_filter(y, 2*int(size/2)+1, polyorder=2)

    savgolweight = np.zeros(len(y))
    savgolweight += sps.norm.pdf(range(len(y)), 0, size)
    savgolweight += sps.norm.pdf(range(len(y)), len(y), size)
    savgolweight /= np.max(savgolweight) #normalize
    gausweight = 1 - savgolweight

    ysmooth = ygaus * gausweight + ysavgol * savgolweight

    return ysmooth


'''
Cloudy I/O
'''

def remove_duplicates(law, fmt):
    '''
    Takes a Cloudy law (e.g. dlaw or tlaw) and a formatter, and removes
    duplicate rows from the law. This is mainly for the illuminated side of the
    simulation, where we have a very finely sampled grid which can result in
    duplicate values after applying the string formatter. This function thus
    does not alter the law in any way, but merely improves readability of the
    Cloudy .in file laws as it doesn't have many (obsolete) duplicate rows.

    law:        2D numpy array of a quantity 'on the Cloudy grid',
                i.e. with the first column the depth from the illuminated face
                and the second column the quantity, both in log10 units.
    fmt:        string formatter used when writing this law to a '.in' file.
                This function will remove floats that are duplicate up to the
                precision implied by this fmt formatter.
    '''

    nonduplicates = [0]
    for i in range(1, len(law)-1):
        if format(law[i,1], fmt) != format(law[i-1,1], fmt) or format(law[i,1], fmt) != format(law[i+1,1], fmt):
            nonduplicates.append(i)
    nonduplicates.append(-1)

    return law[nonduplicates]



def copyadd_Cloudy_in(oldsimname, newsimname, set_thickness=False,
                        dlaw=None, tlaw=None, alaw=None, pTmulaw=None, cextra=None, hextra=None,
                        coolextra=None, othercommands=None, outfiles=[], denspecies=[], selected_den_levels=False,
                        constantT=None, double_tau=False, hcfrac=None):
    '''
    This function makes a copy of a Cloudy in file, and it will append
    the given commands to this .in file.

    For explanation of arguments, see write_Cloudy_in(). Not all of the commands
    of write_Cloudy_in() are also in this function, as commands are only added here
    when they are needed. Most commands can freely be added if needed here.
    '''

    if denspecies != []:
        assert ".den" in outfiles and ".en" in outfiles
    if ".den" in outfiles or ".en" in outfiles:
        assert ".den" in outfiles and ".en" in outfiles and denspecies != []
    if constantT != None:
        assert not np.any(tlaw != None)

    copyfile(oldsimname+".in", newsimname+".in")

    with open(newsimname+".in", "a") as f:
        if set_thickness:
            f.write('\nstop thickness '+'{:.7f}'.format(dlaw[-1,0])+'\t#last dlaw point')
        if ".ovr" in outfiles:
            f.write('\nsave overview ".ovr" last')
        if ".cool" in outfiles:
            f.write('\nsave cooling ".cool" last')
        if ".coolH2" in outfiles:
            f.write('\nsave H2 cooling ".coolH2" last')
        if ".heat" in outfiles:
            f.write('\nsave heating ".heat" last')
        if ".con" in outfiles:
            f.write('\nsave continuum ".con" last units Hz')
        if ".den" in outfiles: #then ".en" is always there as well.
            f.write('\nsave species densities last ".den"\n'+speciesstring(denspecies, selected_levels=selected_den_levels)+"\nend")
            f.write('\nsave species energies last ".en"\n'+speciesstring(denspecies, selected_levels=selected_den_levels)+"\nend")
        if constantT != None:
            f.write('\nconstant temperature t= '+str(constantT)+' linear')
        if double_tau:
            f.write('\ndouble optical depths    #so radiation does not escape into planet core freely')
        if hcfrac:
            f.write('\nset WeakHeatCool '+str(hcfrac)+' #for .heat and .cool output files')
        if othercommands != None:
            f.write("\n"+othercommands)
        if np.any(dlaw != None):
            dlaw = remove_duplicates(dlaw, "1.7f")
            f.write("\n# ========= density law    ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ndlaw table depth\n")
            np.savetxt(f, dlaw, fmt='%1.7f')
            f.write('{:.7f}'.format(dlaw[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(dlaw[-1,1]))
            f.write("\nend of dlaw #last point added to prevent roundoff")
        if np.any(tlaw != None):
            tlaw = remove_duplicates(tlaw, "1.7f")
            f.write("\n# ========= temperature law    ============")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ntlaw table depth\n")
            np.savetxt(f, tlaw, fmt='%1.7f')
            f.write('{:.7f}'.format(tlaw[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(tlaw[-1,1]))
            f.write("\nend of tlaw #last point added to prevent roundoff")
        if np.any(alaw != None):
            alaw = remove_duplicates(alaw, "1.7f")
            f.write("\n# ========= advection law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\nadvectiontable depth\n")
            np.savetxt(f, alaw, fmt='%1.7f')
            f.write('{:.7f}'.format(alaw[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(alaw[-1,1]))
            f.write("\nend of advectiontable #last point added to prevent roundoff")
        if np.any(pTmulaw != None):
            pTmulaw = remove_duplicates(pTmulaw, "1.7f")
            f.write("\n# ========= previous T/mu law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\nprevTmu depth\n")
            np.savetxt(f, pTmulaw, fmt='%1.7f')
            f.write('{:.7f}'.format(pTmulaw[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(pTmulaw[-1,1]))
            f.write("\nend of prevTmu #last point added to prevent roundoff")
        if np.any(cextra != None):
            cextra = remove_duplicates(cextra, "1.7f")
            f.write("\n# ========= cextra law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ncextra table depth\n")
            np.savetxt(f, cextra, fmt='%1.7f')
            f.write('{:.7f}'.format(cextra[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(cextra[-1,1]))
            f.write("\nend of cextra #last point added to prevent roundoff")
        if np.any(hextra != None):
            hextra = remove_duplicates(hextra, "1.7f")
            f.write("\n# ========= hextra law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\nhextra table depth\n")
            np.savetxt(f, hextra, fmt='%1.7f')
            f.write('{:.7f}'.format(hextra[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(hextra[-1,1]))
            f.write("\nend of hextra #last point added to prevent roundoff")
        if np.any(coolextra != None):
            coolextra = remove_duplicates(coolextra, "1.7f")
            f.write("\n# ========= coolextra law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ncoolextra table depth\n")
            np.savetxt(f, coolextra, fmt='%1.7f')
            f.write('{:.7f}'.format(coolextra[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(coolextra[-1,1]))
            f.write("\nend of coolextra #last point added to prevent roundoff")


def write_Cloudy_in(simname, title=None, flux_scaling=None,
                    SED=None, set_thickness=True,
                    dlaw=None, tlaw=None, alaw=None, pTmulaw=None, cextra=None, hextra=None,
                    coolextra=None, othercommands=None, overwrite=False, iterate='convergence',
                    nend=3000, outfiles=['.ovr', '.cool'], denspecies=[], selected_den_levels=False,
                    constantT=None, double_tau=False, cosmic_rays=False, zdict=None, hcfrac=None,
                    comments=None):
    '''
    This function writes a Cloudy .in file for simulating an exoplanet atmosphere.
    Arguments:
        simname:        path+name of Cloudy simulation (without extension)
        title:          title of simulation
        flux_scaling:   nuF(nu) value and at which energy of the SED (required!)
                        should be given as a list with two values, e.g. [7.5, 0.3] meaning nuF(nu) = 7.5 at 0.3 Ryd
        SED:            filename of SED (should be present in Cloudy's data/SED/ or in the folder of the simulation)
        set_thickness:  if True, adds a 'stop thickness' argument based on the maximum depth reached in the given dlaw
        dlaw:           density law (2D numpy array)
        tlaw:           temperature law (2D numpy array)
        alaw:           advectiontable law (2D numpy array). This is for a command that I added to
                        the Cloudy source code.
        pTmulaw:        'previous T/mu law' (2D numpy array). This if for a command that I added to
                        the Cloudy source code.
        cextra:         cextra law (2D numpy array). Note that this command functions differently in my
                        Cloudy source code because I edited it to be able to add expansion cooling.
        hextra:         hextra law (2D numpy array).
        coolextra:      coolextra law (2D numpy array). This is for a command that I added to
                        the Cloudy source code.
        othercommands:  a string with any other commands to add that do not have their own treatment in this function.
        overwrite:      boolean whether to overwrite if the .in file already exists.
        iterate:        Cloudy's iterate command. Usually best to leave at convergence.
        nend:           how many zones to include. Cloudy by default has 1400 but that is too few to handle
                        some high-density profiles so I set it to 3000 standard.
        outfiles:       which output files to produce
        denspecies:     which species (atomic/ionic) to save the densities and energies of the excited levels for
                        (to be used for both .en and .den output)
        selected_den_levels:    boolean passed on to speciesstring() whether to include all excited levels or
                                only those for which we know we can match Cloudy's output to NIST
        constantT:      constant temperature to use
        double_tau:     'double optical depths' command (useful for 1D simulations so that radiation does
                        not escape the cloud/atmosphere at the back-side into the planet core)
        cosmic_rays:    whether to include cosmic rays
        zdict:          dictionary of scale factors for all elements
        hcfrac:         threshold fraction of the total heating/cooling rate for which the .heat and .cool files
                        should save agents. Cloudy's default is 0.05 (i.e. rates <0.05 of the total are not saved).
        comments:       will be written at the top of the file with a #
    '''

    assert flux_scaling != None #we need this to proceed. Give in format [F,E] like nuF(nu) = F at E Ryd
    assert SED != None
    if denspecies != []:
        assert ".den" in outfiles and ".en" in outfiles
    if ".den" in outfiles or ".en" in outfiles:
        assert ".den" in outfiles and ".en" in outfiles and denspecies != []
    if not overwrite:
        assert not os.path.isfile(simname+".in")
    if constantT != None:
        assert not np.any(tlaw != None)

    with open(simname+".in", "w") as f:
        if comments != None:
            f.write(comments+'\n')
        if title != None:
            f.write('title '+title)
        f.write("\n# ========= input spectrum ================")
        f.write("\nnuF(nu) = "+str(flux_scaling[0])+" at "+str(flux_scaling[1])+" Ryd")
        f.write('\ntable SED "'+SED+'"')
        if cosmic_rays:
            f.write('\ncosmic rays background')
        f.write("\n# ========= chemistry      ================")
        f.write("\n# solar abundances and metallicity is standard")
        if zdict != None:
            for element in zdict.keys():
                if zdict[element] == 0.:
                    f.write("\nelement "+element_names[element]+" off")
                elif zdict[element] != 1.: #only write it to Cloudy if the scale factor is not 1
                    f.write("\nelement scale factor "+element_names[element]+" "+str(zdict[element]))
        f.write("\n# ========= other          ================")
        if nend != None:
            f.write("\nset nend "+str(nend)+"   #models at high density need >1400 zones")
        f.write("\nset temperature floor 5 linear")
        f.write("\nstop temperature off     #otherwise it stops at 1e4 K")
        if iterate == 'convergence':
            f.write("\niterate to convergence")
        else:
            f.write("niterate "+str(iterate))
        f.write("\nprint last iteration")
        if set_thickness:
            f.write('\nstop thickness '+'{:.7f}'.format(dlaw[-1,0])+'\t#last dlaw point')
        if constantT != None:
            f.write('\nconstant temperature t= '+str(constantT)+' linear')
        if double_tau:
            f.write('\ndouble optical depths    #so radiation does not escape into planet core freely')
        if hcfrac:
            f.write('\nset WeakHeatCool '+str(hcfrac)+' #for .heat and .cool output files')
        if othercommands != None:
            f.write("\n"+othercommands)
        f.write("\n# ========= output         ================")
        if ".ovr" in outfiles:
            f.write('\nsave overview ".ovr" last')
        if ".cool" in outfiles:
            f.write('\nsave cooling ".cool" last')
        if ".coolH2" in outfiles:
            f.write('\nsave H2 cooling ".coolH2" last')
        if ".heat" in outfiles:
            f.write('\nsave heating ".heat" last')
        if ".con" in outfiles:
            f.write('\nsave continuum ".con" last units Hz')
        if ".den" in outfiles: #then ".en" is always there as well.
            f.write('\nsave species densities last ".den"\n'+speciesstring(denspecies, selected_levels=selected_den_levels)+"\nend")
            f.write('\nsave species energies last ".en"\n'+speciesstring(denspecies, selected_levels=selected_den_levels)+"\nend")
        if dlaw is not None:
            dlaw = remove_duplicates(dlaw, "1.7f")
            f.write("\n# ========= density law    ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ndlaw table depth\n")
            np.savetxt(f, dlaw, fmt='%1.7f')
            f.write('{:.7f}'.format(dlaw[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(dlaw[-1,1]))
            f.write("\nend of dlaw #last point added to prevent roundoff")
        if tlaw is not None:
            tlaw = remove_duplicates(tlaw, "1.7f")
            f.write("\n# ========= temperature law    ============")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ntlaw table depth\n")
            np.savetxt(f, tlaw, fmt='%1.7f')
            f.write('{:.7f}'.format(tlaw[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(tlaw[-1,1]))
            f.write("\nend of tlaw #last point added to prevent roundoff")
        if alaw is not None:
            alaw = remove_duplicates(alaw, "1.7f")
            f.write("\n# ========= advection law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\nadvectiontable depth\n")
            np.savetxt(f, alaw, fmt='%1.7f')
            f.write('{:.7f}'.format(alaw[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(alaw[-1,1]))
            f.write("\nend of advectiontable #last point added to prevent roundoff")
        if pTmulaw is not None:
            pTmulaw = remove_duplicates(pTmulaw, "1.7f")
            f.write("\n# ========= previous T/mu law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\nprevTmu depth\n")
            np.savetxt(f, pTmulaw, fmt='%1.7f')
            f.write('{:.7f}'.format(pTmulaw[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(pTmulaw[-1,1]))
            f.write("\nend of prevTmu #last point added to prevent roundoff")
        if cextra is not None:
            cextra = remove_duplicates(cextra, "1.7f")
            f.write("\n# ========= cextra law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ncextra table depth\n")
            np.savetxt(f, cextra, fmt='%1.7f')
            f.write('{:.7f}'.format(cextra[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(cextra[-1,1]))
            f.write("\nend of cextra #last point added to prevent roundoff")
        if hextra is not None:
            hextra = remove_duplicates(hextra, "1.7f")
            f.write("\n# ========= hextra law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\nhextra table depth\n")
            np.savetxt(f, hextra, fmt='%1.7f')
            f.write('{:.7f}'.format(hextra[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(hextra[-1,1]))
            f.write("\nend of hextra #last point added to prevent roundoff")
        if coolextra is not None:
            coolextra = remove_duplicates(coolextra, "1.7f")
            f.write("\n# ========= coolextra law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ncoolextra table depth\n")
            np.savetxt(f, coolextra, fmt='%1.7f')
            f.write('{:.7f}'.format(coolextra[-1,0]+0.1)+
                        ' '+'{:.7f}'.format(coolextra[-1,1]))
            f.write("\nend of coolextra #last point added to prevent roundoff")


def insertden_Cloudy_in(simname, denspecies, selected_den_levels=True, rerun=False):
    '''
    This function takes a Cloudy .in input file and adds species to the
    'save species densities' command. This is useful if you e.g. first went
    through the temperature convergence scheme, but later want to add additional
    species to the 'converged' simulation.

    simname:                see write_Cloudy_in().
    denspecies:             see write_Cloudy_in().
    selected_den_levels:    see write_Cloudy_in().
    rerun: [bool]           whether to rerun the just edited simulation through Cloudy.
    '''

    with open(simname+".in", "r") as f:
        oldcontent = f.readlines()

    newcontent = oldcontent
    indices = [i for i, s in enumerate(oldcontent) if 'save species densities' in s]
    if len(indices) == 0: #then there is no 'save species densities' command yet
        newcontent.append('\nsave species densities last ".den"\n'+speciesstring(denspecies, selected_levels=selected_den_levels)+"\nend")
        newcontent.append('\nsave species energies last ".en"\n'+speciesstring(denspecies, selected_levels=selected_den_levels)+"\nend")

    elif len(indices) == 1: #then there already is a 'save species densities' command with some species
        for sp in denspecies.copy():
            if len([i for i, s in enumerate(oldcontent) if sp+"[" in s]) != 0: #check if this species is already in the file
                denspecies.remove(sp)
                print(sp, "was already in the .in file so I did not add it again.")
        if len(denspecies) >= 1:
            newcontent.insert(indices[0]+1, speciesstring(denspecies, selected_levels=selected_den_levels)+"\n")
            #also add them to the 'save species energies' list
            indices2 = [i for i, s in enumerate(oldcontent) if 'save species energies' in s]
            newcontent.insert(indices2[0]+1, speciesstring(denspecies, selected_levels=selected_den_levels)+"\n")
        else:
            return

    else:
        print("There are multiple 'save species densities' commands in the .in file. This shouldn't be the case, please check.")
        return

    newcontent = "".join(newcontent) #turn list into string
    with open(simname+".in", "w") as f: #overwrite the old file
        f.write(newcontent)

    if rerun:
        path, name = os.path.split(simname)
        os.system("cd "+path+" && "+cloudyruncommand+' "'+name+'"')


'''
Useful classes
'''

class Parker:
    '''
    Class that stores a Parker wind profile and its parameters

    Arguments:
        plname:     [str] name of the planet
        T:          [int] isothermal temperature
        Mdot:       [str or float] log10 of the mass-loss rate in cgs
        filename:   [str] filename of the parker wind profile .txt file
        fH:         hydrogen fraction (for H/He-only profiles)
        zdict:      scale factor dictionary
        SED:        name of the spectrum used to construct the Parker profile
        readin:     [bool] whether to read in the Parker profile file.
                    If you don't want to read in, use filename=''
    '''

    def __init__(self, plname, T, Mdot, pdir, fH=None, zdict=None, SED=None, readin=True):
        self.plname = plname
        self.T = int(T)
        if type(Mdot) == str:
            self.Mdot = Mdot
            self.Mdotf = float(Mdot)
        elif type(Mdot) == float:
            self.Mdot = "%.3f" % Mdot
            self.Mdotf = Mdot
        if fH != None:
            self.fH = fH
        if zdict != None:
            self.zdict = zdict
        if SED != None:
            self.SED = SED
        if readin:
            self.prof = read_parker(plname, T, Mdot, pdir)


class Planet:
    '''
    Class that saves parameters for a given planet.

    Arguments:
        name:   [str] name of the planet. If this name appears in the 'planets.txt'
                    file, the class will automatically load its parameters.

        R:      [float] optional, radius in cm
        Rstar:  [float] optional, radius of the host star in cm
        a:      [float] optional, semimajor axis in cm
        M:      [float] optional, mass in g
        Mstar:  [float] optional, mass of the host star in g
        bp:     [float] optional, transit impact parameter (dimensionless, between 0 and 1 in units of Rstar)
        Rroche: [float] optional, Roche radius in cm,
                if not given (preffered), will be calculated based on other parameters
        SEDname:[str]   optional, name of the stellar SED used, including file extension

        If the planet name was found in 'planets.txt', but any of the other
        parameters are given as well, those values will overwrite the values
        of the 'planet.txt' file.
    '''

    def __init__(self, name, fullname=None, R=None, Rstar=None, a=None, M=None, Mstar=None, bp=None, SEDname=None):
        #check if we can fetch planet parameters from planets.txt:
        if name in planets_file['name'].values or name in planets_file['full name'].values:
            this_planet = planets_file[(planets_file['name'] == name) | (planets_file['full name'] == name)]
            assert len(this_planet) == 1, "Multiple entries were found in planets.txt for this planet name."
            
            self.name = this_planet['name'].values[0]
            self.fullname = this_planet['full name'].values[0]
            self.R = this_planet['R [RJ]'].values[0] * RJ #in cm
            self.Rstar = this_planet['Rstar [Rsun]'].values[0] *Rsun #in cm
            self.a = this_planet['a [AU]'].values[0] * AU #in cm
            self.M = this_planet['M [MJ]'].values[0] * MJ #in g
            self.Mstar = this_planet['Mstar [Msun]'].values[0] * Msun #in g
            self.bp = this_planet['transit impact parameter'].values[0] #dimensionless
            self.SEDname = this_planet['SEDname'].values[0].strip() #strip to remove whitespace from beginning and end

            #if any specified, overwrite values read from planets.txt
            if fullname != None:
                self.fullname = fullname
            if R != None:
                self.R = R
            if Rstar != None:
                self.Rstar = Rstar
            if a != None:
                self.a = a
            if M != None:
                self.M = M
            if Mstar != None:
                self.Mstar = Mstar
            if bp != None:
                self.bp = bp
            if SEDname != None:
                self.SEDname = SEDname

        else:
            print(f"Creating Planet object with name {name} that's not in the database.")
            self.name = name
            self.fullname = fullname
            self.R = R
            self.Rstar = Rstar
            self.a = a
            self.M = M
            self.Mstar = Mstar
            self.bp = bp
            self.SEDname = SEDname

        self.__update_Rroche()
        self.__update_phi()

    def set_var(self, name=None, fullname=None, R=None, Rstar=None, a=None, M=None, Mstar=None, bp=None, SEDname=None):
        '''
        To edit values after creation of the object.
        '''
        if name != None:
            self.name = name
        if R != None:
            self.R = R
            self.__update_phi()
        if Rstar != None:
            self.Rstar = Rstar
        if a != None:
            self.a = a
            self.__update_Rroche()
        if M != None:
            self.M = M
            self.__update_phi()
            self.__update_Rroche()
        if Mstar != None:
            self.Mstar = Mstar
            self.__update_Rroche()
        if bp != None:
            self.bp = bp
        if SEDname != None:
            self.SEDname = SEDname

    def __update_phi(self):
        '''
        Tries to update the gravitational potential.
        '''
        if (self.M != None) and (self.R != None):
            self.phi = G * self.M / self.R
        else:
            self.phi = None

    def __update_Rroche(self):
        '''
        Tries to update the Roche radius.
        '''
        if (self.a != None) and (self.M != None) and (self.Mstar != None):
            self.Rroche = roche_radius(self.a, self.M, self.Mstar)
        else:
            self.Rroche = None

    def print_params(self):
        print(f"Name: {self.name}")
        if self.fullname is not None:
            print(f"Full name: {self.fullname}")
        if self.R is not None:
            print(f"Planet radius: {self.R} cm, {self.R / RJ} RJ")
        if self.Rstar is not None:
            print(f"Star radius: {self.Rstar} cm, {self.Rstar / Rsun} Rsun")
        if self.a is not None:
            print(f"Semi-major axis: {self.a} cm, {self.a / AU} AU")
        if self.M is not None:
            print(f"Planet mass: {self.M} g, {self.M / MJ} MJ")
        if self.Mstar is not None:
            print(f"Star mass: {self.Mstar} g, {self.Mstar / Msun} Msun")
        if self.bp is not None:
            print(f"Transit impact parameter: {self.bp} Rstar")
        if self.SEDname is not None:
            print(f"Stellar spectrum name: {self.SEDname}")
        if self.Rroche is not None:
            print(f"Roche radius: {self.Rroche} cm, {self.Rroche / RJ} RJ, {self.Rroche / self.R} Rp")
        if self.phi is not None:
            print(f"log10(Gravitational potential): {np.log10(self.phi)} log10(erg/g)")

    def plot_transit_geometry(self, phase=0., altmax=None):
        fig, ax = plt.subplots(1)
        ax.plot(self.Rstar*np.cos(np.linspace(0, 2*np.pi, 100)), self.Rstar*np.sin(np.linspace(0, 2*np.pi, 100)), c='k')
        ax.plot(self.a*np.sin(2*np.pi*phase) + self.R*np.cos(np.linspace(0, 2*np.pi, 100)), self.bp*self.Rstar + self.R*np.sin(np.linspace(0, 2*np.pi, 100)), c='b')
        if self.Rroche is not None:
            ax.plot(self.a*np.sin(2*np.pi*phase) + self.Rroche*np.cos(np.linspace(0, 2*np.pi, 100)), self.bp*self.Rstar + self.Rroche*np.sin(np.linspace(0, 2*np.pi, 100)), c='b', linestyle='dotted')
        if altmax is not None:
            ax.plot(self.a*np.sin(2*np.pi*phase) + altmax*self.R*np.cos(np.linspace(0, 2*np.pi, 100)), self.bp*self.Rstar + altmax*self.R*np.sin(np.linspace(0, 2*np.pi, 100)), c='b', linestyle='dashed')
        plt.axis('equal')
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        ax.set_title(f"Phase: {phase}")
        plt.show()


class Sim:
    '''
    Main class for loading Cloudy simulations into Python.

    Arguments:
        simname:        [str] name of the Cloudy simulation without extension.
                        Usually safest to give the full path, i.e.
                        '/Users/dion/src/cloudy/sims/1D/HD209458b/Tstruc_fiducial/parker_8000_10.000/converged'
        altmax:         [int] maximum altitude of the Cloudy simulation, in units of Rp (optional)
        proceedFail:    [bool] proceed even if the targeted Cloudy simulation did not exit OK.
        files:          [list] which output files to read in, e.g. 'con', 'heat', 'ovr'
        planet:         [Planet] object of the simulated planet
        parker:         [Parker] object of the simulated Parker wind profile
    '''

    def __init__(self, simname, altmax=None, proceedFail=False, files=['all'], planet=None, parker=None):
        if not isinstance(simname, str):
            raise TypeError("simname must be set to a string")
        self.simname = simname

        #check the Cloudy version, and if the simulation did not crash.
        _succesful = False
        with open(simname+'.out', 'r') as f:
            _outfile_content = f.read()
            if "Cloudy exited OK" in _outfile_content:
                _succesful = True
            if "Cloudy 17" in _outfile_content:
                self.cloudy_version = "17"
            elif "Cloudy 23" in _outfile_content:
                self.cloudy_version = "23"
            else:
                raise TypeError(f"This simulation did not use Cloudy v17 or v23, which are the only supported versions: {simname}")
        if not _succesful and not proceedFail:
            raise FileNotFoundError(f"This simulation went wrong: {simname} Check the .out file!")

        #read the .in file to extract some sim info like changes to the chemical composition and altmax
        self.disabled_elements = []
        zelem = {}
        _parker_T, _parker_Mdot, _parker_dir = None, None, None #temp variables
        with open(simname+'.in', 'r') as f:
            for line in f:
                if line[0] == '#': #then it is a comment written by sunbather, extract info:
                    #check if a planet was defined
                    if 'plname' in line:
                        self.p = Planet(line.split('=')[-1].strip('\n'))
                    
                    #check if a Parker profile was defined
                    if 'parker_T' in line:
                        _parker_T = int(line.split('=')[-1].strip('\n'))
                    if 'parker_Mdot' in line:
                        _parker_Mdot = line.split('=')[-1].strip('\n')
                    if 'parker_dir' in line:
                        _parker_dir = line.split('=')[-1].strip('\n')
                    
                    #check if an altmax was defined
                    if 'altmax' in line:
                        self.altmax = int(line.split('=')[1].strip('\n'))
                
                #read SED
                if 'table SED' in line:
                    self.SEDname = line.split('"')[1]
                
                #read chemical composition
                if 'element scale factor' in line.rstrip():
                    zelem[element_symbols[line.split(' ')[3]]] = float(line.rstrip().split(' ')[-1])
                elif 'element' in line.rstrip() and 'off' in line.rstrip():
                    self.disabled_elements.append(element_symbols[line.split(' ')[1]])
                    zelem[element_symbols[line.split(' ')[1]]] = 0.
        
        #set zdict and abundances as attributes
        self.zdict = get_zdict(zelem=zelem)
        self.abundances = get_abundances(zdict=self.zdict)

        #overwrite/set manually given Planet object
        if planet != None:
            assert isinstance(planet, Planet)
            if hasattr(self, 'p'):
                print("I had already read out the Planet object from the .in file, but I will overwrite that with the object you have given.")
            self.p = planet

        #check if the SED of the Planet object matches the SED of the Cloudy simulation
        if hasattr(self, 'p') and hasattr(self, 'SEDname'):
            if self.p.SEDname != self.SEDname:
                print("I read in the .in file that the SED used is", self.SEDname, "which is different from the one of your Planet object. " \
                        "I will change the .SEDname attribute of the Planet object to match the one actually used in the simulation. Are you " \
                        "sure that also the associated Parker wind profile is correct?")
                self.p.set_var(SEDname = self.SEDname)

        #try to set a Parker object if the .in file had the required info for that
        if hasattr(self, 'p') and (_parker_T != None) and (_parker_Mdot != None) and (_parker_dir != None):
            self.par = Parker(self.p.name, _parker_T, _parker_Mdot, _parker_dir)
        
        #overwrite/set manually given Parker object
        if parker != None:
            assert isinstance(parker, Parker)
            if hasattr(self, 'par'):
                print("I had already read out the Parker object from the .in file, but I will overwrite that with the object you have given.")
            self.par = parker

        #overwrite/set manually given altmax
        if altmax != None:
            if not (isinstance(altmax, float) or isinstance(altmax, int)):
                raise TypeError("altmax must be set to a float or int") #can it actually be a float? I'm not sure if my code can handle it - check and try.
            if hasattr(self, 'altmax'):
                if self.altmax != altmax:
                    print("I read the altmax from the .in file, but the value you have explicitly passed is different. " \
                            "I will use your value, but please make sure it is correct.")
            self.altmax = altmax


        #temporary variables for adding the alt-columns to the pandas dataframes
        _Rp, _altmax = None, None
        if hasattr(self, 'p') and hasattr(self, 'altmax'):
            _Rp = self.p.R
            _altmax = self.altmax
        
        #read in the Cloudy simulation files
        self.simfiles = []
        for simfile in glob.glob(simname+'.*', recursive=True):
            filetype = simfile.split('.')[-1]
            if filetype=='ovr' and ('ovr' in files or 'all' in files):
                self.ovr = process_overview(self.simname+'.ovr', Rp=_Rp, altmax=_altmax, abundances=self.abundances)
                self.simfiles.append('ovr')
            if filetype=='con' and ('con' in files or 'all' in files):
                self.con = process_continuum(self.simname+'.con')
                self.simfiles.append('con')
            if filetype=='heat' and ('heat' in files or 'all' in files):
                self.heat = process_heating(self.simname+'.heat', Rp=_Rp, altmax=_altmax)
                self.simfiles.append('heat')
            if filetype=='cool' and ('cool' in files or 'all' in files):
                self.cool = process_cooling(self.simname+'.cool', Rp=_Rp, altmax=_altmax, cloudy_version=self.cloudy_version)
                self.simfiles.append('cool')
            if filetype=='coolH2' and ('coolH2' in files or 'all' in files):
                self.coolH2 = process_coolingH2(self.simname+'.coolH2', Rp=_Rp, altmax=_altmax)
                self.simfiles.append('coolH2')
            if filetype=='den' and ('den' in files or 'all' in files):
                self.den = process_densities(self.simname+'.den', Rp=_Rp, altmax=_altmax)
                self.simfiles.append('den')
            if filetype=='en' and ('en' in files or 'all' in files):
                self.en = process_energies(self.simname+'.en')
                self.simfiles.append('en')

        #set the velocity structure in .ovr if we have an associated Parker profile - needed for radiative transfer
        if hasattr(self, 'par'): 
            if hasattr(self.par, 'prof'):
                Sim.addv(self, self.par.prof.alt, self.par.prof.v)


    def get_simfile(self, simfile):
        '''
        Allows accessing the Cloudy simulation files.
        They can all also be accessed as an attribute,
        for example sim.ovr or sim.cool
        '''
        if simfile not in self.simfiles:
            raise FileNotFoundError("This simulation does not have a", simfile, "output file.")

        if simfile == 'ovr':
            return self.ovr
        elif simfile == 'con':
            return self.con
        elif simfile == 'heat':
            return self.heat
        elif simfile == 'cool':
            return self.cool
        elif simfile == 'coolH2':
            return self.coolH2
        elif simfile == 'den':
            return self.den
        elif simfile == 'en':
            return self.en
        elif simfile == 'ionFe':
            return self.ionFe
        elif simfile == 'ionNa':
            return self.ionNa


    def add_parker(self, parker):
        '''
        Adds a Parker profile object to the Sim, in case it wasn't added upon initialization.
        '''

        assert isinstance(parker, Parker)
        self.par = parker
        if hasattr(parker, 'prof'):
            Sim.addv(self, parker.prof.alt, parker.prof.v)


    def addv(self, alt, v, delete_negative=True):
        '''
        Adds velocity profile in cm/s on the Cloudy altitude grid. Will be added to the .ovr file,
        but also available as the .v attribute for potential backwards compatability.
        Called automatically when adding a Parker object to the sim.
        '''

        assert 'ovr' in self.simfiles

        if delete_negative:
            v[v < 0.] = 0.

        self.ovr['v'] = interp1d(alt, v)(self.ovr.alt)

        vseries = pd.Series(index=self.ovr.alt.index, dtype=float)
        vseries[self.ovr.alt.index] = interp1d(alt, v)(self.ovr.alt)
        self.v = vseries
