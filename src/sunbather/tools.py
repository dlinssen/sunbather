"""
Tools for sunbather
"""
import os
import glob
import re
from shutil import copyfile
from fractions import Fraction
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import scipy.stats as sps
from scipy.ndimage import gaussian_filter1d


# ######################################
# ########## GLOBAL CONSTANTS ##########
# ######################################

sunbatherpath = os.path.dirname(
    os.path.abspath(__file__)
)  # the absolute path where this code lives


def get_cloudy_path():
    try:
        # the path where Cloudy is installed
        cloudypath = os.environ["CLOUDY_PATH"]
    except KeyError as exc:
        cloudypath = f"{sunbatherpath}/cloudy/c23.01"
        if not os.path.exists(f"{cloudypath}/source/cloudy.exe"):
            raise KeyError(
                "The environment variable 'CLOUDY_PATH' is not set. "
                "Please set this variable in your .bashrc/.zshrc file "
                "to the path where the Cloudy installation is located. "
                "Do not point it to the /source/ subfolder, but to the main folder."
            ) from exc
    return cloudypath


def get_sunbather_project_path():
    try:
        projectpath = os.environ[
            "SUNBATHER_PROJECT_PATH"
        ]  # the path where you save your simulations and do analysis
    except KeyError as exc:
        projectpath = "./"
        if not os.path.exists(f"{projectpath}/planets.txt"):
            raise FileNotFoundError(
                "The environment variable 'SUNBATHER_PROJECT_PATH' is not set, and no "
                "planets.txt file found in current directory. Please set the "
                "'SUNBATHER_PROJECT_PATH' variable in your .bashrc/.zshrc file "
                "to the path where you want the sunbather models to be saved, "
                "and make sure that the 'planets.txt' file is present in that folder."
            ) from exc
    return projectpath


def get_planets_file():
    if os.path.exists(f"{get_sunbather_project_path()}/planets.txt"):
        # read planet parameters globally instead of in the Planets class (so we do it only
        # once)
        planets_file = pd.read_csv(
            f"{get_sunbather_project_path()}/planets.txt",
            dtype={
                "name": str,
                "full name": str,
                "R [RJ]": np.float64,
                "Rstar [Rsun]": np.float64,
                "a [AU]": np.float64,
                "M [MJ]": np.float64,
                "Mstar [Msun]": np.float64,
                "transit impact parameter": np.float64,
                "SEDname": str,
            },
            comment="#",
        )
        return planets_file
    raise FileNotFoundError(
        "The $SUNBATHER_PROJECT_PATH/planets.txt file cannot be found. "
        "Please check if your $SUNBATHER_PROJECT_PATH actually exists on your machine. "
        "Then, copy /sunbather/planets.txt to your project path."
    )

# define constants:
c = 2.99792458e10  # cm/s
h = 4.135667696e-15  # eV s, used to plot wavelengths in keV units
mH = 1.674e-24  # g
k = 1.381e-16  # erg/K
AU = 1.49597871e13  # cm
pc = 3.08567758e18  # cm
RJ = 7.1492e9  # cm
RE = 6.371e8  # cm
Rsun = 69634000000  # cm
Msun = 1.9891e33  # g
MJ = 1.898e30  # g
ME = 5.9722e27  # g
G = 6.6743e-8  # cm3/g/s2
Ldict = {
    "S": 0,
    "P": 1,
    "D": 2,
    "F": 3,
    "G": 4,
    "H": 5,
    "I": 6,
    "K": 7,
    "L": 8,
    "M": 9,
    "N": 10,
    "O": 11,
    "Q": 12,
    "R": 13,
    "T": 14,
}  # atom number of states per L orbital

element_names = {
    "H": "hydrogen",
    "He": "helium",
    "Li": "lithium",
    "Be": "beryllium",
    "B": "boron",
    "C": "carbon",
    "N": "nitrogen",
    "O": "oxygen",
    "F": "fluorine",
    "Ne": "neon",
    "Na": "sodium",
    "Mg": "magnesium",
    "Al": "aluminium",
    "Si": "silicon",
    "P": "phosphorus",
    "S": "sulphur",
    "Cl": "chlorine",
    "Ar": "argon",
    "K": "potassium",
    "Ca": "calcium",
    "Sc": "scandium",
    "Ti": "titanium",
    "V": "vanadium",
    "Cr": "chromium",
    "Mn": "manganese",
    "Fe": "iron",
    "Co": "cobalt",
    "Ni": "nickel",
    "Cu": "copper",
    "Zn": "zinc",
}
element_symbols = dict(
    (reversed(item) for item in element_names.items())
)  # reverse dictionary mapping e.g. 'hydrogen'->'H'

# number of corresponding energy levels between Cloudy and NIST - read txt file
# header for more info
species_enlim = pd.read_csv(sunbatherpath + "/species_enlim.txt", index_col=0, header=1)


# ######################################
# ##########  CLOUDY SPECIES  ##########
# ######################################


def get_specieslist(max_ion=6, exclude_elements=None):
    """
    Returns a list of atomic and ionic species names. Default returns all
    species up to 6+ ionization. Higher than 6+ ionization is rarely attained
    in an exoplanet atmosphere, but it can occur in high XUV flux scenarios
    such as young planetary systems.  The species list only includes species
    for which the NIST database has any spectral line coefficients, as there is
    little use saving other species as well.

    Parameters
    ----------
    max_ion : int, optional
        Maximum ionization degree of the included species, by default 6
    exclude_elements : str or list, optional
        Elements to include (in both atomic and ionic form), by default []

    Returns
    -------
    specieslist : list
        List of atomic and ionic species names in the string format expected by
        Cloudy.
    """
    if exclude_elements is None:
        exclude_elements = []

    if max_ion > 12:
        warnings.warn(
            "tools.get_specieslist(): You have set max_ion > 12, but "
            "sunbather is currently only able to process species up to 12+ ionized. "
            "However, this should typically be enough, even when using a strong "
            "XUV flux."
        )

    if isinstance(exclude_elements, str):  # turn into list with one element
        exclude_elements = [exclude_elements]

    specieslist = species_enlim.index.tolist()  # all species up to 12+

    for element in exclude_elements:
        specieslist = [sp for sp in specieslist if sp.split("+")[0] != element]

    for sp in specieslist[:]:
        sp_split = sp.split("+")

        if len(sp_split) == 1:
            deg_ion = 0
        elif sp_split[1] == "":
            deg_ion = 1
        else:
            deg_ion = int(sp_split[1])

        if deg_ion > max_ion:
            specieslist.remove(sp)

    return specieslist


def get_mass(species):
    """
    Returns the mass of an atomic or positive ion. For ions,
    it returns the mass of the atom, since the electron mass is negligible.

    Parameters
    ----------
    species : str
        Name of the species in the string format expected by Cloudy.

    Returns
    -------
    mass : float
        Mass of the species in units of g.
    """

    atom = species.split("+")[0]

    mass_dict = {
        "H": 1.6735575e-24,
        "He": 6.646477e-24,
        "Li": 1.15e-23,
        "Be": 1.4965082e-23,
        "B": 1.795e-23,
        "C": 1.9945e-23,
        "N": 2.3259e-23,
        "O": 2.6567e-23,
        "F": 3.1547e-23,
        "Ne": 3.35092e-23,
        "Na": 3.817541e-23,
        "Mg": 4.0359e-23,
        "Al": 4.48038988e-23,
        "Si": 4.6636e-23,
        "P": 5.14331418e-23,
        "S": 5.324e-23,
        "Cl": 5.887e-23,
        "Ar": 6.6335e-23,
        "K": 6.49243e-23,
        "Ca": 6.6551e-23,
        "Sc": 7.4651042e-23,
        "Ti": 7.9485e-23,
        "V": 8.45904e-23,
        "Cr": 8.63416e-23,
        "Mn": 9.1226768e-23,
        "Fe": 9.2733e-23,
        "Co": 9.786087e-23,
        "Ni": 9.74627e-23,
        "Cu": 1.0552e-22,
        "Zn": 1.086e-22,
    }  # g

    mass = mass_dict[atom]

    return mass


# ######################################
# ##########   CLOUDY FILES   ##########
# ######################################


def process_continuum(filename, nonzero=False):
    """
    Reads a .con file from the 'save continuum units angstrom' command.
    It renames the columns and adds a wavelength column.
    The flux units of the continuum are as follows:
    Take the SED in spectral flux density, so F(nu) instead of nu*F(nu), and
    find the total area by integration. Then multiply with the frequency,
    to get nu*F(nu), and normalize that by the total area found, and multiply
    with the total luminosity. Those are the units of Cloudy.

    Parameters
    ----------
    filename : str
        Filename of a 'save continuum' Cloudy output file.
    nonzero : bool, optional
        Whether to remove rows where the incident spectrum is 0 (i.e., not
        defined), by default False

    Returns
    -------
    con_df : pandas.DataFrame
        Parsed output of the 'save continuum' Cloudy command.
    """

    con_df = pd.read_table(filename)
    con_df.rename(columns={"#Cont  nu": "wav", "net trans": "nettrans"}, inplace=True)
    if nonzero:
        con_df = con_df[con_df.incident != 0]

    return con_df


def process_heating(filename, Rp=None, altmax=None, cloudy_version="17"):
    """
    Reads a .heat file from the 'save heating' command.
    If Rp and altmax are given, it adds an altitude/radius scale.
    For each unique heating agent, it adds a column with its rate at each radial bin.

    Parameters
    ----------
    filename : str
        Filename of a 'save heating' Cloudy output file.
    Rp : numeric, optional
        Planet radius in units of cm, by default None
    altmax : numeric, optional
        Maximum altitude of the simulation in units of planet radius, by default None
    cloudy_version : str, optional
        Major Cloudy release version, by default "17"

    Returns
    -------
    heat : pandas.DataFrame
        Parsed output of the 'save heating' Cloudy command.

    Raises
    ------
    TypeError
        If a Cloudy version was used that is not supported by sunbather.
    """

    # determine max number of columns (otherwise pd.read_table assumes it is the number
    # of the first row)
    max_columns = 0
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            num_columns = len(line.split("\t"))
            max_columns = max(max_columns, num_columns)
    # set up the column names
    if cloudy_version == "17":
        fixed_column_names = ["depth", "temp", "htot", "ctot"]
    elif cloudy_version == "23":
        fixed_column_names = ["depth", "temp", "htot", "ctot", "adv"]
    else:
        raise TypeError("Only C17.02 and C23.01 are currently supported.")
    num_additional_columns = (max_columns - 4) // 2
    additional_column_names = [
        f"htype{i}" for i in range(1, num_additional_columns + 1) for _ in range(2)
    ]
    additional_column_names[1::2] = [
        f"hfrac{i}" for i in range(1, num_additional_columns + 1)
    ]
    all_column_names = fixed_column_names + additional_column_names
    heat = pd.read_table(
        filename, delimiter="\t", skiprows=1, header=None, names=all_column_names
    )

    if heat["depth"].eq("#>>>>  Ionization not converged.").any():
        warnings.warn(
            f"The simulation you are reading in exited OK but does contain ionization "
            f"convergence failures: {filename[:-5]}"
        )
        heat = heat[
            heat["depth"] != "#>>>>  Ionization not converged."
        ]  # remove those extra lines from the heat DataFrame

    # remove the "second rows", which sometimes are in the .heat file and do
    # not give the heating at a given depth
    if isinstance(heat.depth.iloc[0], str):  # in some cases there are no second rows
        heat = heat[heat.depth.map(len) < 12]  # delete second rows

    heat.depth = pd.to_numeric(heat.depth)  # str to float
    heat.reset_index(
        drop=True, inplace=True
    )  # reindex so that it has same index as e.g. .ovr

    if Rp is not None and altmax is not None:  # add altitude scale
        heat["alt"] = altmax * Rp - heat.depth

    agents = []
    for column in heat.columns:
        if column.startswith("htype"):
            agents.extend(heat[column].unique())
    agents = list(
        set(agents)
    )  # all unique heating agents that appear somewhere in the .heat file

    for agent in agents:
        heat[agent] = np.nan  # add 'empty' column for each agent

    # now do a (probably sub-optimal) for-loop over the whole df to put all hfracs in
    # the corresponding column
    htypes = [f"htype{i+1}" for i in range(num_additional_columns)]
    hfracs = [f"hfrac{i+1}" for i in range(num_additional_columns)]
    for htype, hfrac in zip(htypes, hfracs):
        for index, agent in heat[htype].items():
            rate = heat.loc[index, hfrac]
            heat.loc[index, agent] = rate

    if (
        np.nan in heat.columns
    ):  # sometimes columns are partially missing, resulting in columns called nan
        heat.drop(columns=[np.nan], inplace=True)

    heat["sumfrac"] = heat.loc[:, [col for col in heat.columns if "hfrac" in col]].sum(
        axis=1
    )

    return heat


def process_cooling(filename, Rp=None, altmax=None, cloudy_version="17"):
    """
    Reads a .cool file from the 'save cooling' command.
    If Rp and altmax are given, it adds an altitude/radius scale.
    For each unique cooling agent, it adds a column with its rate at each radial bin.

    Parameters
    ----------
    filename : str
        Filename of a 'save cooling' Cloudy output file.
    Rp : numeric, optional
        Planet radius in units of cm, by default None
    altmax : numeric, optional
        Maximum altitude of the simulation in units of planet radius, by default None
    cloudy_version : str, optional
        Major Cloudy release version, by default "17"

    Returns
    -------
    cool : pandas.DataFrame
        Parsed output of the 'save cooling' Cloudy command.

    Raises
    ------
    TypeError
        If a Cloudy version was used that is not supported by sunbather.
    """

    # determine max number of columns (otherwise pd.read_table assumes it is
    # the number of the first row)
    max_columns = 0
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            num_columns = len(line.split("\t"))
            max_columns = max(max_columns, num_columns)
    # set up the column names
    if cloudy_version == "17":
        fixed_column_names = ["depth", "temp", "htot", "ctot"]
    elif cloudy_version == "23":
        fixed_column_names = ["depth", "temp", "htot", "ctot", "adv"]
    else:
        raise Exception("Only C17.02 and C23.01 are currently supported.")
    num_additional_columns = (max_columns - 4) // 2
    additional_column_names = [
        f"ctype{i}" for i in range(1, num_additional_columns + 1) for _ in range(2)
    ]
    additional_column_names[1::2] = [
        f"cfrac{i}" for i in range(1, num_additional_columns + 1)
    ]
    all_column_names = fixed_column_names + additional_column_names
    cool = pd.read_table(
        filename, delimiter="\t", skiprows=1, header=None, names=all_column_names
    )

    if cool["depth"].eq("#>>>>  Ionization not converged.").any():
        warnings.warn(
            f"The simulation you are reading in exited OK but does contain ionization "
            f"convergence failures: {filename[:-5]}"
        )
        # remove those extra lines from the cool DataFrame
        cool = cool[cool["depth"] != "#>>>>  Ionization not converged."]
        cool["depth"] = cool["depth"].astype(float)
        cool = cool.reset_index(drop=True)  # so it matches other dfs like .ovr

    if Rp is not None and altmax is not None:  # add altitude scale
        cool["alt"] = altmax * Rp - cool.depth

    agents = []
    for column in cool.columns:
        if column.startswith("ctype"):
            agents.extend(cool[column].unique())
    agents = list(
        set(agents)
    )  # all unique cooling agents that appear somewhere in the .cool file

    for agent in agents:
        cool[agent] = np.nan  # add 'empty' column for each agent

    # now do a (probably sub-optimal) for-loop over the whole df to put all cfracs in
    # the corresponding column
    ctypes = [f"ctype{i+1}" for i in range(num_additional_columns)]
    cfracs = [f"cfrac{i+1}" for i in range(num_additional_columns)]
    for ctype, cfrac in zip(ctypes, cfracs):
        for index, agent in cool[ctype].items():
            rate = cool.loc[index, cfrac]
            cool.loc[index, agent] = rate

    if (
        np.nan in cool.columns
    ):  # sometimes columns are partially missing, resulting in columns called nan
        cool.drop(columns=[np.nan], inplace=True)

    cool["sumfrac"] = cool.loc[:, [col for col in cool.columns if "cfrac" in col]].sum(
        axis=1
    )

    return cool


def process_coolingH2(filename, Rp=None, altmax=None):
    """
    Reads a .coolH2 file from the 'save H2 cooling' command,
    which keeps track of cooling and heating processes unique to the
    H2 molecule, when using the 'database H2' command.
    From the Cloudy source code "mole_h2_io.cpp" the columns are:
    depth, Temp, ctot/htot, H2 destruction rate Solomon TH85,
    H2 destruction rate Solomon big H2, photodis heating,
    heating dissoc. electronic exited states,
    cooling collisions in X (neg = heating),
    "HeatDexc"=net heat, "-HeatDexc/abundance"=net cool per particle

    If Rp and altmax are given, it adds an altitude/radius scale.

    Parameters
    ----------
    filename : str
        Filename of a 'save H2 cooling' Cloudy output file.
    Rp : numeric, optional
        Planet radius in units of cm, by default None
    altmax : numeric, optional
        Maximum altitude of the simulation in units of planet radius, by default None

    Returns
    -------
    coolH2 : pandas.DataFrame
        Parsed output of the 'save H2 cooling' Cloudy command.
    """

    coolH2 = pd.read_table(
        filename,
        names=[
            "depth",
            "Te",
            "ctot",
            "desTH85",
            "desbigH2",
            "phdisheat",
            "eedisheat",
            "collcool",
            "netheat",
            "netcoolpp",
        ],
        header=1,
    )
    if Rp is not None and altmax is not None:
        coolH2["alt"] = altmax * Rp - coolH2["depth"]

    return coolH2


def process_overview(filename, Rp=None, altmax=None, abundances=None):
    """
    Reads in a '.ovr' file from the 'save overview' command.
    If Rp and altmax are given, it adds an altitude/radius scale.
    It also adds the mass density, the values of which are only correct if
    the correct abundances are passed.

    Parameters
    ----------
    filename : str
        Filename of a 'save overview' Cloudy output file.
    Rp : numeric, optional
        Planet radius in units of cm, by default None
    altmax : numeric, optional
        Maximum altitude of the simulation in units of planet radius, by default None
    abundances : dict, optional
        Dictionary with the abudance of each element, expressed as a fraction of the
        total.  Can be easily created with get_abundances(). By default None, which
        results in solar composition.

    Returns
    -------
    ovr : pandas.DataFrame
        Parsed output of the 'save overview' Cloudy command.
    """

    ovr = pd.read_table(filename)
    ovr.rename(columns={"#depth": "depth"}, inplace=True)
    ovr["rho"] = hden_to_rho(ovr.hden, abundances=abundances)  # Hdens to total dens
    if Rp is not None and altmax is not None:
        ovr["alt"] = altmax * Rp - ovr["depth"]
    ovr["mu"] = calc_mu(ovr.rho, ovr.eden, abundances=abundances)

    if (
        (ovr["2H_2/H"].max() > 0.1)
        or (ovr["CO/C"].max() > 0.1)
        or (ovr["H2O/O"].max() > 0.1)
    ):
        warnings.warn(
            f"Molecules are significant, the calculated mean particle mass could be "
            f"inaccurate: {filename}"
        )

    return ovr


def process_densities(filename, Rp=None, altmax=None):
    """
    Reads a .den file from the 'save species densities' command.
    If Rp and altmax are given, it adds an altitude/radius scale.

    Parameters
    ----------
    filename : str
        Filename of a 'save species densities' Cloudy output file.
    Rp : numeric, optional
        Planet radius in units of cm, by default None
    altmax : numeric, optional
        Maximum altitude of the simulation in units of planet radius, by default None

    Returns
    -------
    den : pandas.DataFrame
        Parsed output of the 'save species densities' Cloudy command.
    """

    den = pd.read_table(filename)
    den.rename(columns={"#depth densities": "depth"}, inplace=True)

    if Rp is not None and altmax is not None:
        den["alt"] = altmax * Rp - den["depth"]

    return den


def process_energies(filename, rewrite=True, cloudy_version="17"):
    """
    Reads a '.en' file from the 'save species energies' command.
    This command must always be used alongside the 'save species densities' command,
    since they give the associated energy of each level printed in the
    densities file. Without saving the energies, it is for example not clear
    which atomic configuration / energy level 'He[52]' corresponds to.
    This function returns a dictionary mapping the column names of
    the .den file to their corresponding atomic configurations.
    The atomic configuration is needed to identify the spectral lines originating
    from this level during radiative transfer.

    Parameters
    ----------
    filename : str
        Filename of a 'save species energies' Cloudy output file.
    rewrite : bool, optional
        Whether to rewrite the file to only keeping only the first row. Normally,
        the energies of each energy level are stored per depth cell of the simulation,
        but they should be the same at each depth. Retaining only the values of the
        first row in this way helps to compress file size. By default True.
    cloudy_version : str, optional
        Major Cloudy release version, by default "17"

    Returns
    -------
    en_df : dict
        Dictionary mapping the column names of the .den file to their atomic configurations.

    Raises
    ------
    ValueError
        If the energy values are not the same at each depth.
    """

    en = pd.read_table(
        filename, float_precision="round_trip"
    )  # use round_trip to prevent exp numerical errors

    if (
        en.columns.values[0][0] == "#"
    ):  # condition checks whether it has already been rewritten, if not, we do all following stuff:

        for i, col in enumerate(en.columns):  # check if all rows are the same
            if len(en.iloc[:, col].unique()) != 1:
                raise ValueError(
                    "In reading .en file, found a column with not identical values!"
                    + " filename:",
                    filename,
                    "col:",
                    col,
                    "colname:",
                    en.columns[col],
                    "unique values:",
                    en.iloc[:, col].unique(),
                )

        en.rename(
            columns={en.columns.values[0]: en.columns.values[0][10:]}, inplace=True
        )  # rename the column

        if rewrite:  # save with only first row to save file size
            en.iloc[[0], :].to_csv(filename, sep="\t", index=False, float_format="%.5e")

    en_df = pd.DataFrame(index=en.columns.values)
    en_df["species"] = [
        k.split("[")[0] for k in en_df.index.values
    ]  # we want to match 'He12' to species='He', for example
    en_df["energy"] = en.iloc[0, :].values
    en_df["configuration"] = ""
    en_df["term"] = ""
    en_df["J"] = ""

    # the & set action takes the intersection of all unique species of the .en file, and those known with NIST levels
    unique_species = list(set(en_df.species.values) & set(species_enlim.index.tolist()))

    for species in unique_species:
        species_levels = pd.read_table(
            sunbatherpath + "/RT_tables/" + species + "_levels_processed.txt"
        )  # get the NIST levels
        species_energies = en_df[
            en_df.species == species
        ].energy  # get Cloudy's energies

        # tolerance of difference between Cloudy's and NISTs energy levels. They usually differ at the decimal level so we need some tolerance.
        atol = species_enlim.loc[species, f"atol_C{cloudy_version}"]
        # start by assuming we can match this many energy levels
        n_matching = species_enlim.loc[species, f"idx_C{cloudy_version}"]

        for n in range(n_matching):
            if (
                not np.abs(species_energies.iloc[n] - species_levels.energy.iloc[n])
                < atol
            ):
                warnings.warn(
                    f"In {filename} while getting atomic states for species {species}, I expected to be able to match the first {n_matching} "
                    f"energy levels between Cloudy and NIST to a precision of {atol} but I have an energy mismatch at energy level {n+1}. "
                    f"This should not introduce bugs, as I will now only parse the first {n} levels."
                )

                # for debugging, you can print the energy levels of Cloudy and NIST:
                # print("\nCloudy, NIST, Match?")
                # for i in range(n_matching):
                #    print(species_energies.iloc[i], species_levels.energy.iloc[i], np.isclose(species_energies.iloc[:n_matching], species_levels.energy.iloc[:n_matching], rtol=0.0, atol=atol)[i])

                n_matching = n  # reset n_matching to how many actually match

                break

        # Now assign the first n_matching columns to their expected values as given by the NIST species_levels DataFrame
        first_iloc = np.where(en_df.species == species)[0][
            0
        ]  # iloc at which the species (e.g. He or Ca+3) starts.
        en_df.iloc[
            first_iloc: first_iloc + n_matching, en_df.columns.get_loc("configuration")
        ] = species_levels.configuration.iloc[:n_matching].values
        en_df.iloc[
            first_iloc: first_iloc + n_matching, en_df.columns.get_loc("term")
        ] = species_levels.term.iloc[:n_matching].values
        en_df.iloc[first_iloc: first_iloc + n_matching, en_df.columns.get_loc("J")] = (
            species_levels.J.iloc[:n_matching].values
        )

    return en_df


def find_line_lowerstate_in_en_df(species, lineinfo, en_df, verbose=False):
    """
    Finds the column name of the .den file that corresponds to
    the ground state of the given line. So for example if species='He',
    and we are looking for the metastable helium line,
    it will return 'He[2]', meaning the 'He[2]' column of the .den file contains
    the number densities of the metastable helium atom.

    Additionally, it calculates a multiplication factor <1 for the number
    density of this energy level. This is for spectral lines that originate from a
    specific J (total angular momentum quantum number) configuration, but Cloudy
    does not save the densities of this specific J-value, only of the parent LS state.
    In this case, we use a statistical argument to guess how many of the particles
    are in each J-state. For this, we use that each J-state has 2*J+1 substates,
    and then assuming all substates are equally populated, we can calculate the
    population of each J-level. The assumption of equal population may not always be strictly
    valid. In LTE, the population should in principle be calculated form the Boltzmann
    distribution, but equal populations will be a good approximation at high temperature
    or when the energy levels of the J-substates are close together. In NLTE, the
    assumption is less valid due to departure from the Boltzmann equation.

    Parameters
    ----------
    species : str
        Name of the atomic or ionic species in the string format expected by Cloudy.
    lineinfo : pandas.DataFrame
        One row containing the spectral line coefficients from NIST, from the
        RT.read_NIST_lines() function.
    en_df : dict
        Dictionary mapping the column names of the .den file to their atomic configurations,
        from the process_energies() function.
    verbose : bool, optional
        Whether to print out , by default False

    Returns
    -------
    match : str
        Column name of the .den file that contains the number densities of the energy
        level that this spectral line originates from.
    lineweight : float
        Multiplication factor <1 for the number density of this energy level, to get
        the number density of the specific J-state that the spectral line originates from.
    """

    en_df = en_df[
        en_df.species == species
    ]  # keep only the part for this species to not mix up the energy levels of different ones
    match, lineweight = None, None  # start with the assumption that we cannot match it

    # check if the line originates from a J sublevel, a term, or only principal quantum number
    if str(lineinfo["term_i"]) != "nan" and str(lineinfo["J_i"]) != "nan":
        linetype = "J"  # then now match with configuration and term:
        matchedrow = en_df[
            (en_df.configuration == lineinfo.conf_i)
            & (en_df.term == lineinfo.term_i)
            & (en_df.J == lineinfo.J_i)
        ]
        assert len(matchedrow) <= 1

        if len(matchedrow) == 1:
            match = matchedrow.index.item()
            # since the Cloudy column is for this J specifically, we don't need
            # to downweigh the density
            lineweight = 1.0

        elif len(matchedrow) == 0:
            # the exact J was not found in Cloudy's levels, but maybe the term
            # is there in Cloudy, just not resolved.
            matchedtermrow = en_df[
                (en_df.configuration == lineinfo.conf_i)
                & (en_df.term == lineinfo.term_i)
            ]

            if len(matchedtermrow) == 1:
                if str(matchedtermrow.J.values[0]) == "nan":
                    # This can only happen if the Cloudy level is a term with
                    # no J resolved.  Then we use statistical weights to guess
                    # how many of the atoms in this term state would be in the
                    # J state of the level and use this as lineweight
                    L = Ldict[
                        "".join(
                            x
                            for x in matchedtermrow.loc[:, "term"].item()
                            if x.isalpha()
                        )[-1]
                    ]  # last letter in term string
                    S = (
                        float(
                            re.search(
                                r"\d+", matchedtermrow.loc[:, "term"].item()
                            ).group()
                        )
                        - 1.0
                    ) / 2.0  # first number in term string
                    J_states = np.arange(np.abs(L - S), np.abs(L + S) + 1, 1.0)
                    J_statweights = 2 * J_states + 1
                    J_probweights = J_statweights / np.sum(J_statweights)

                    lineweight = J_probweights[
                        J_states == Fraction(lineinfo.loc["J_i"])
                    ][0]

                    match = matchedtermrow.index.item()
                else:
                    verbose_print(
                        f"One J level of the term is resolved, but not the one of this line: {species} "
                        + lineinfo.conf_i,
                        verbose=verbose,
                    )

            else:
                verbose_print(
                    f"Multiple J levels of the term are resolved, but not the one of this line: {species} "
                    + lineinfo.conf_i,
                    verbose=verbose,
                )

    elif str(lineinfo["term_i"]) != "nan":
        linetype = "LS"

        verbose_print(
            "Currently not able to do lines originating from LS state without J number.",
            verbose=verbose,
        )
        verbose_print(
            f"Lower state configuration: {species} " + lineinfo.conf_i, verbose=verbose
        )
    else:
        linetype = "n"

        verbose_print(
            "Currently not able to do lines originating from n state without term. "
            "This is not a problem if this line is also in the NIST database with its "
            "different term components, such as for e.g. H n=2, but only if they "
            "aren't such as for H n>6, or if they go to an upper level n>6 from any "
            "given level.",
            verbose=verbose,
        )
        verbose_print(
            f"Lower state configuration: {species} " + lineinfo.conf_i, verbose=verbose
        )

        """
        DEVELOPERS NOTE:
        If we do decide to make this functionality, for example by summing the densities
        of all sublevels of a particular n, we also need to tweak the cleaning of
        hydrogen lines algorithm. Right now, we remove double lines only for the upper
        state, so e.g. for Ly alpha, we remove the separate 2p 3/2 and 2p 1/2 etc.
        component and leave only the one line with upper state n=2.  However, we don't
        do this for lower states, which is not a problem yet because the lower n state
        lines are ignored as stated above. However if we make the functionality, we
        should also remove double lines in the lower level.
        """

    return match, lineweight


# ######################################
# ##########  MISCELLANEOUS  ###########
# ######################################


def verbose_print(message, verbose=False):
    """
    Prints the provided string only if verbose is True.

    Parameters
    ----------
    message : str
        String to optionally print.
    verbose : bool, optional
        Whether to print the provided message, by default False
    """

    if verbose:
        print(message)


def get_SED_norm_1AU(SEDname):
    """
    Reads in an SED file and returns the normalization in monochromatic flux
    (i.e., nu*F_nu or lambda*F_lambda) and Ryd units.
    These are needed because Cloudy does not preserve the normalization of
    user-specified SEDs. To do a simulation of an atmosphere, the normalization
    of the SED must afterwards still be scaled to the planet distance.
    Then, the log10 of nuFnu can be passed to Cloudy using the
    "nuFnu(nu) = ... at ... Ryd" command.
    This function requires that the units of the SED are Å and
    monochromatic flux (i.e., nu*F_nu or lambda*F_lambda).

    Parameters
    ----------
    SEDname : str
        Name of a SED file located in $CLOUDY_PATH/data/SED/.

    Returns
    -------
    nuFnu : float
        Monochromatic flux specified at the energy of the Ryd output variable.
    Ryd : float
        Energy where the monochromatic flux of the nuFnu output variable is specified.
    """

    with open(f"{get_cloudy_path()}/data/SED/{SEDname}", "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):  # skip through the comments at the top
                assert ("angstrom" in line) or ("Angstrom" in line)  # verify the units
                assert "nuFnu" in line  # verify the units
                break
        data = np.genfromtxt(
            f, skip_header=1
        )  # skip first line, which has extra words specifying the units

    ang, nuFnu = data[-2, 0], data[-2, 1]  # read out intensity somewhere
    Ryd = 911.560270107676 / ang  # convert wavelength in Å to energy in Ryd

    return nuFnu, Ryd


def speciesstring(specieslist, selected_levels=False, cloudy_version="17"):
    """
    Takes a list of species names and returns a long string with those species
    between quotes and [:] added (or [:maxlevel] if selected_levels=True),
    and \n between them. This string can then be used in a Cloudy input
    script for .den and .en files. The maxlevel is the number of energy levels
    that can be matched between Cloudy and NIST. Saving higher levels than that is not
    really useful since they cannot be postprocessed by the radiative transfer module.

    Parameters
    ----------
    specieslist : list
        Species to include.
    selected_levels : bool, optional
        If True, only energy levels up to the number that can be matched to NIST
        will be included. If False, all energy levels of each species will be
        included, regardless of whether we can match them to NIST. By default False.
    cloudy_version : str, optional
        Major Cloudy release version, by default "17"

    Returns
    -------
    speciesstr : str
        One long string containing the species and the energy level numbers.
    """

    if not selected_levels:  # so just all levels available in cloudy
        speciesstr = '"' + specieslist[0] + '[:]"'
        if len(specieslist) > 1:
            for species in specieslist[1:]:
                speciesstr += '\n"' + species + '[:]"'

    elif (
        selected_levels
    ):  # then we read out the max level that we expect to match the energy of
        speciesstr = (
            '"'
            + specieslist[0]
            + "[:"
            + str(species_enlim.loc[specieslist[0], f"idx_C{cloudy_version}"])
            + ']"'
        )
        if len(specieslist) > 1:
            for species in specieslist[1:]:
                speciesstr += (
                    '\n"'
                    + species
                    + "[:"
                    + str(species_enlim.loc[species, f"idx_C{cloudy_version}"])
                    + ']"'
                )

    return speciesstr


def read_parker(plname, T, Mdot, pdir, filename=None):
    """
    Reads an isothermal Parker wind profile as generated by the construct_parker.py module.

    Parameters
    ----------
    plname : str
        Planet name (must have parameters stored in
        $SUNBATHER_PROJECT_PATH/planets.txt).
    T : str or numeric
        Temperature in units of K.
    Mdot : str or numeric
        log of the mass-loss rate in units of g s-1.
    pdir : str
        Directory as $SUNBATHER_PROJECT_PATH/parker_profiles/*plname*/*pdir*/
        where the isothermal parker wind density and velocity profiles are saved.
        Different folders may exist there for a given planet, to separate for example
        profiles with different assumptions such as stellar SED/semi-major
        axis/composition.
    filename : str, optional
        If None, the profile as specified by plname, T, Mdot, pdir is read. If not None,
        filename must specify the full path + filename of the isothermal Parker wind
        profile to read in. By default None.

    Returns
    -------
    pprof : pandas.DataFrame
        Radial density, velocity and mean particle mass profiles of the isothermal
        Parker wind profile.
    """

    if filename is None:
        Mdot = f"{float(Mdot):.3f}"
        T = str(int(T))
        filename = (
            f"{get_sunbather_project_path()}/parker_profiles/{plname}/"
            f"{pdir}/pprof_{plname}_T={T}_M={Mdot}.txt"
        )

    pprof = pd.read_table(
        filename, names=["alt", "rho", "v", "mu"], dtype=np.float64, comment="#"
    )
    pprof["drhodr"] = np.gradient(pprof["rho"], pprof["alt"])

    return pprof


def calc_mu(rho, ne, abundances=None, mass=False):
    """
    Calculates the mean particle mass of an atomic/ionic gas mixture,
    but neglecting molecules (and the negligible mass contributed by
    electrons). Based on formula:
    mu = sum(ni*mi) / (sum(ni) + ne)
    where ni and mi are the number density and mass of element i, and
    ne is the electron number density.
    Use ni = ntot * fi   and   ntot = rho / sum(fi*mi)
    where ntot is the total number density, fi the abundance of element i
    expressed as a 0<fraction<1, and rho the mass density.
    Substitution yields:
    mu = sum(fi*mi) / (1 + (ne * sum(fi*mi))/rho)

    Parameters
    ----------
    rho : array-like or numeric
        Mass density in units of g cm-3.
    ne : array-like or numeric
        Electron number density in units of cm-3.
    abundances : dict, optional
        Dictionary with the abundance of each element, expressed as a fraction of the
        total.  Can be easily created with get_abundances(). By default None, which
        results in solar composition.
    mass : bool, optional
        If True returns mu in units of g, if False returns mu in units of amu, by
        default False.

    Returns
    -------
    mu : array-like or numeric
        Mean particle mass.
    """

    if abundances is None:
        abundances = get_abundances()

    sum_all = 0.0
    for element in abundances.keys():
        sum_all += abundances[element] * get_mass(element)

    mu = sum_all / (1 + ne * sum_all / rho)  # mu in g
    if not mass:
        mu = mu / mH  # mu in amu

    return mu


def get_zdict(z=1.0, zelem=None):
    """
    Returns a dictionary of the scale factors of each element relative to solar.

    Parameters
    ----------
    z : numeric, optional
        Metallicity relative to solar (in linear units, i.e., z=1 is solar), by default 1.
    zelem : dict, optional
        Dictionary of scale factors for specific elements. For example,
        {'C':2} to get two times the solar carbon abundance. By default {}.

    Returns
    -------
    zdict : dict
        Dictionary with the scale factors of all elements relative
        to the default solar composition.
    """
    if zelem is None:
        zelem = {}

    assert (
        "H" not in zelem.keys()
    ), "You cannot scale hydrogen, scale everything else instead."

    zdict = {
        "He": 1.0,
        "Li": z,
        "Be": z,
        "B": z,
        "C": z,
        "N": z,
        "O": z,
        "F": z,
        "Ne": z,
        "Na": z,
        "Mg": z,
        "Al": z,
        "Si": z,
        "P": z,
        "S": z,
        "Cl": z,
        "Ar": z,
        "K": z,
        "Ca": z,
        "Sc": z,
        "Ti": z,
        "V": z,
        "Cr": z,
        "Mn": z,
        "Fe": z,
        "Co": z,
        "Ni": z,
        "Cu": z,
        "Zn": z,
    }

    for element in zelem.keys():
        zdict[element] *= zelem[element]

    return zdict


def get_abundances(zdict=None):
    """
    Returns a dictionary of the fractional abundances of each element (sums to 1).

    Parameters
    ----------
    zdict : dict, optional
        Dictionary with the scale factors of all elements relative
        to a solar composition. Can be easily created with get_zdict().
        Default is None, which results in a solar composition.

    Returns
    -------
    abundances : dict
        Dictionary with the abundances of all elements, expressed as a fraction of the total.
    """

    # solar abundance relative to hydrogen (Hazy table 7.1):
    rel_abundances = {
        "H": 1.0,
        "He": 0.1,
        "Li": 2.04e-9,
        "Be": 2.63e-11,
        "B": 6.17e-10,
        "C": 2.45e-4,
        "N": 8.51e-5,
        "O": 4.9e-4,
        "F": 3.02e-8,
        "Ne": 1e-4,
        "Na": 2.14e-6,
        "Mg": 3.47e-5,
        "Al": 2.95e-6,
        "Si": 3.47e-5,
        "P": 3.2e-7,
        "S": 1.84e-5,
        "Cl": 1.91e-7,
        "Ar": 2.51e-6,
        "K": 1.32e-7,
        "Ca": 2.29e-6,
        "Sc": 1.48e-9,
        "Ti": 1.05e-7,
        "V": 1e-8,
        "Cr": 4.68e-7,
        "Mn": 2.88e-7,
        "Fe": 2.82e-5,
        "Co": 8.32e-8,
        "Ni": 1.78e-6,
        "Cu": 1.62e-8,
        "Zn": 3.98e-8,
    }

    if zdict is not None:
        assert (
            "H" not in zdict.keys()
        ), "You cannot scale hydrogen, scale everything else instead."
        for element in zdict.keys():
            rel_abundances[element] *= zdict[element]

    total = sum(list(rel_abundances.values()))
    abundances = {k: v / total for k, v in rel_abundances.items()}

    return abundances


def rho_to_hden(rho, abundances=None):
    """
    Converts a mass density in units of g cm-3 to a hydrogen number density
    in units of cm-3, for a given chemical composition. Based on formula:
    rho = nH*mH + ntot*sum(fj*mj)
    where nH is the hydrogen number density, mH the hydrogen atom mass,
    mj and fj the mass and abundance (=fraction) of element j (the sum excludes hydrogen)
    and ntot=rho/sum(fi*mi)
    where the sum runs over every element i including hydrogen.
    Substitution yields:
    nH = rho/mH * (1 - sum(fj*mj)/sum(fi*mi))

    Parameters
    ----------
    rho : array-like or numeric
        Mass density in units of g cm-3.
    abundances : dict, optional
        Dictionary with the abundance of each element, expressed as a fraction of the total.
        Can be easily created with get_abundances(). By default None, which
        results in solar composition.

    Returns
    -------
    hden : array-like or numeric
        Hydrogen number density in units of cm-3.
    """

    if abundances is None:
        abundances = get_abundances()  # get a solar composition

    sum_all = 0.0
    for element in abundances.keys():
        sum_all += abundances[element] * get_mass(element)

    sum_noH = sum_all - abundances["H"] * get_mass(
        "H"
    )  # subtract hydrogen to get the sum without H

    hden = rho / mH * (1 - sum_noH / sum_all)

    return hden


def hden_to_rho(hden, abundances=None):
    """
    Converts a hydrogen number density in units of cm-3 to a mass density
    in units of g cm-3, for a given chemical composition. Based on formula:
    rho = nH*mH + ntot*sum(fj*mj)
    where nH is the hydrogen number density, mH the hydrogen atom mass,
    mj and fj the mass and abundance (=fraction) of element j (the sum excludes hydrogen)
    and ntot=rho/sum(fi*mi)
    where the sum runs over every element i including hydrogen.
    Substitution yields:
    rho = nH*mH / (1 - sum(fj*mj)/sum(fi*mi))

    Parameters
    ----------
    hden : array-like or numeric
        Hydrogen number density in units of cm-3.
    abundances : dict, optional
        Dictionary with the abundance of each element, expressed as a fraction of the total.
        Can be easily created with get_abundances(). By default None, which
        results in solar composition.

    Returns
    -------
    rho : array-like or numeric
        Mass density in units of g cm-3.
    """

    if abundances is None:
        abundances = get_abundances()  # get a solar composition

    sum_all = 0.0
    for element in abundances.keys():
        sum_all += abundances[element] * get_mass(element)

    sum_noH = sum_all - abundances["H"] * get_mass(
        "H"
    )  # subtract hydrogen to get the sum without H

    rho = hden * mH / (1 - sum_noH / sum_all)

    return rho


def roche_radius(a, Mp, Mstar):
    """
    Returns the Hill/Roche radius. This is an approximation valid for
    small Mp / Mstar.

    Parameters
    ----------
    a : numeric
        Semi-major axis in units of cm.
    Mp : numeric
        Planet mass in units of g.
    Mstar : numeric
        Star mass in units of g.

    Returns
    -------
    Rroche : float
        Hill/Roche radius in units of cm.
    """

    Rroche = a * pow(Mp / (3.0 * (Mstar + Mp)), 1.0 / 3.0)

    return Rroche


def set_alt_ax(ax, altmax=8, labels=True):
    """
    Sets the xscale of a figure to represent altitude/radius in units
    of planetary radius. Sets the x-axis to log, adds an xlabel and
    sets convenient xticks.

    Parameters
    ----------
    ax : matplotlib.Axes
        Figure axis to configure.
    altmax : int, optional
        Maximum altitude of the simulation in units of planet radius, by default 8.
    labels : bool, optional
        Whether to use an xlabel and xticklabels, by default True
    """

    ax.set_xscale("log")
    ax.set_xlim(1, altmax)
    ticks = np.concatenate((np.arange(1, 2, 0.1), np.arange(2, altmax + 1, 1)))
    if altmax <= 3:
        ticklabels = [
            "1",
            "1.1",
            "1.2",
            "1.3",
            "1.4",
            "1.5",
            "1.6",
            "1.7",
            "1.8",
            "1.9",
        ]
        ticklabels2 = ["%i" % t for t in np.arange(2, altmax + 1, 1).astype(int)]
    elif altmax <= 10:
        ticklabels = ["1", "", "", "", "", "1.5", "", "", "", ""]
        ticklabels2 = ["%i" % t for t in np.arange(2, altmax + 1, 1).astype(int)]
    elif altmax <= 14:
        ticklabels = [
            "1",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "2",
            "3",
            "4",
            "5",
            "",
            "7",
            "",
            "",
            "10",
        ]
        ticklabels2 = [""] * (altmax - 10)
    else:
        ticklabels = [
            "1",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "2",
            "3",
            "4",
            "5",
            "",
            "7",
            "",
            "",
            "10",
        ]
        ticklabels2 = [""] * (altmax - 10)
        ticklabels2b = np.arange(15, altmax + 0.1, 5).astype(int)
        index = 4
        for t2b in ticklabels2b:
            ticklabels2[index] = str(t2b)
            index += 5

    ticklabels = ticklabels + ticklabels2

    ax.set_xticks(ticks)
    if labels:
        ax.set_xticklabels(ticklabels)
        ax.set_xlabel(r"Radius [$R_p$]")
    else:
        ax.set_xticklabels([])


def alt_array_to_Cloudy(alt, quantity, altmax, Rp, nmax, log=True):
    """
    Takes an atmospheric quantity as a function of altitude/radius,
    and returns it as a 2D array of that same quantity as a function of
    distance from the top of the atmosphere. The latter is the format
    in which Cloudy expects quantities to be given with for example the
    'dlaw' and 'tlaw' commands, since it works from the illuminated face
    of the 'cloud' towards the planet core.

    Parameters
    ----------
    alt : array-like
        Altitude/radius grid in units of cm. Must be in ascending order.
    quantity : array-like
        Quantity defined at the positions of the alt grid.
    altmax : numeric
        Maximum altitude of the simulation in units of planet radius.
    Rp : numeric
        Planet radius in units of cm.
    nmax : int
        Number of grid points to use.
    log : bool, optional
        Whether to return the log10 of the depth and quantity values,
        which is what Cloudy expects, by default True

    Returns
    -------
    law : numpy.ndarray
        The quantity on a 'depth'-grid as a 2D array.
    """

    if isinstance(alt, pd.Series):
        alt = alt.values
    if isinstance(quantity, pd.Series):
        quantity = quantity.values

    assert alt[1] > alt[0]  # should be in ascending alt order
    assert (
        alt[-1] - altmax * Rp > -1.0
    )  # For extrapolation: the alt scale should extend at least to within 1 cm of altmax*Rp

    if not np.isclose(alt[0], Rp, rtol=1e-2, atol=0.0):
        warnings.warn(
            f"Are you sure the altitude array starts at Rp? alt[0]/Rp = {alt[0]/Rp}"
        )

    depth = altmax * Rp - alt
    ifunc = interp1d(depth, quantity, fill_value="extrapolate")

    Clgridr1 = np.logspace(np.log10(alt[0]), np.log10(altmax * Rp), num=int(0.8 * nmax))
    Clgridr1[0], Clgridr1[-1] = (
        alt[0],
        altmax * Rp,
    )  # reset these for potential log-numerical errors
    Clgridr1 = (Clgridr1[-1] - Clgridr1)[::-1]
    # sample the first 10 points better since Cloudy messes up with log-space interpolation there
    Clgridr2 = np.logspace(-2, np.log10(Clgridr1[9]), num=nmax - len(Clgridr1))
    Clgridr = np.concatenate((Clgridr2, Clgridr1[10:]))
    Clgridr[0] = 1e-35

    Clgridq = ifunc(Clgridr)
    law = np.column_stack((Clgridr, Clgridq))
    if log:
        law[law[:, 1] == 0.0, 1] = 1e-100
        law = np.log10(law)

    return law


def smooth_gaus_savgol(y, size=None, fraction=None):
    """
    Smooth an array using a Gaussian filter, but smooth the start and
    end of the array with a Savitzky-Golay filter.

    Parameters
    ----------
    y : array-like
        Array to smooth.
    size : int, optional
        Smoothing size expressed as a number of points that will serve as the Gaussian
        standard deviation. If None, instead a fraction must be provided, by default None
    fraction : float, optional
        Smoothing size expressed as a fraction of the total array length
        that will serve as the Gaussian standard deviation. If None, instead
        a size must be provided, by default None

    Returns
    -------
    ysmooth : numpy.ndarray
        Smoothed array.

    Raises
    ------
    ValueError
        If neither or both size and fraction were provided.
    """

    if size is not None and fraction is None:
        size = max(3, size)
    elif fraction is not None and size is None:
        assert (
            0.0 < fraction < 1.0
        ), "fraction must be greater than 0 and smaller than 1"
        size = int(np.ceil(len(y) * fraction) // 2 * 2 + 1)  # make it odd
        size = max(3, size)
    else:
        raise ValueError("Please provide either 'size' or 'fraction'.")

    ygaus = gaussian_filter1d(y, size)
    ysavgol = savgol_filter(y, 2 * int(size / 2) + 1, polyorder=2)

    savgolweight = np.zeros(len(y))
    savgolweight += sps.norm.pdf(range(len(y)), 0, size)
    savgolweight += sps.norm.pdf(range(len(y)), len(y), size)
    savgolweight /= np.max(savgolweight)  # normalize
    gausweight = 1 - savgolweight

    ysmooth = ygaus * gausweight + ysavgol * savgolweight

    return ysmooth


# ######################################
# ##########    CLOUDY I/O    ##########
# ######################################


def run_Cloudy(filename, folder=None):
    """
    Run a Cloudy simulation from within Python.

    Parameters
    ----------
    filename : str
        Name of the simulation input file. If the folder argument is not
        specfied, filename must include the full path to the simulation.
        If the folder argument is specified, the filename should only
        specify the filename.
    folder : str, optional
        Full path to the directory where the file is located, excluding
        the filename itself, which must be specified with the filename
        argument. If folder is None, filename must also include the
        full path. By default None.
    """

    if folder is None:  # then the folder should be in the simname
        folder, filename = os.path.split(filename)

    if filename.endswith(".in"):
        filename = filename[:-3]  # filename should not contain the extension

    os.system(
        f"cd {folder} && {get_cloudy_path()}/source/cloudy.exe -p {filename}"
    )


def remove_duplicates(law, fmt):
    """
    Takes a Cloudy law (e.g., dlaw or tlaw) and a formatter, and removes
    duplicate rows from the law. This is mainly for the illuminated side of the
    simulation, where we have a very finely sampled grid which can result in
    duplicate values after applying the string formatter. This function thus
    does not alter the law in any way, but merely improves readability of the
    Cloudy .in file laws as the many (obsolete) duplicate rows are removed.

    Parameters
    ----------
    law : numpy.ndarray
        Quantity on a 'depth'-grid as a 2D array, in the format that Cloudy expects it.
    fmt : str
        String formatter specifying a float precision. This function will remove
        floats that are duplicate up to the precision implied by this fmt formatter.

    Returns
    -------
    new_law : numpy.ndarray
        Same quantity but with rows removed that have the same float precision
        under the provided fmt formatter.
    """

    nonduplicates = [0]
    for i in range(1, len(law) - 1):
        if format(law[i, 1], fmt) != format(law[i - 1, 1], fmt) or format(
            law[i, 1], fmt
        ) != format(law[i + 1, 1], fmt):
            nonduplicates.append(i)
    nonduplicates.append(-1)

    new_law = law[nonduplicates]

    return new_law


def copyadd_Cloudy_in(
    oldsimname,
    newsimname,
    set_thickness=False,
    dlaw=None,
    tlaw=None,
    cextra=None,
    hextra=None,
    othercommands=None,
    outfiles=None,
    denspecies=None,
    selected_den_levels=False,
    constantT=None,
    double_tau=False,
    hcfrac=None,
    cloudy_version="17",
):
    """
    Makes a copy of a Cloudy input file and appends commands.

    Parameters
    ----------
    oldsimname : str
        Full path + name of the Cloudy input file to copy, without the file extension.
    newsimname : str
        Full path + name of the target Cloudy input file, without the file extension.
    set_thickness : bool, optional
        Whether to include a command that ends the simulation at a depth equal
        to the length of the dlaw, by default True
    dlaw : numpy.ndarray, optional
        Hydrogen number density in units of cm-3, as a 2D array where dlaw[:,0]
        specifies the log10 of the depth into the cloud in cm, and dlaw[:,1]
        specifies the log10 of the hydrogen number density in units of cm-3, by default
        None
    tlaw : numpy.ndarray, optional
        Temperature in units of K as a 2D array where tlaw[:,0]
        specifies the log10 of the depth into the cloud in cm, and tlaw[:,1]
        specifies the log10 of the temperature in units of K, by default None
    cextra : numpy.ndarray, optional
        Extra unspecified cooling in units of erg s-1 cm-3, as a 2D array where
        cextra[:,0] specifies the log10 of the depth into the cloud in cm,
        and cextra[:,1] specifies the log10 of the cooling rate in units of
        erg s-1 cm-3, by default None
    hextra : numpy.ndarray, optional
        Extra unspecified heating in units of erg s-1 cm-3, as a 2D array where
        hextra[:,0] specifies the log10 of the depth into the cloud in cm,
        and hextra[:,1] specifies the log10 of the heating rate in units of
        erg s-1 cm-3, by default None
    othercommands : str, optional
        String to include in the input file. Any command not otherwise supported
        by this function can be included here, by default None
    outfiles : list, optional
        List of file extensions indicating which Cloudy output to save. For example,
        include '.heat' to include the 'save heating' command, by default ['.ovr',
        '.cool']
    denspecies : list, optional
        List of atomic/ionic species for which to save densities and energies, which
        are needed to do radiative transfer. The list can easily be created by the
        get_specieslist() function. By default [].
    selected_den_levels : bool, optional
        If True, only energy levels up to the number that can be matched to NIST
        will be included in the 'save densities' command. If False, all energy levels
        of each species will be included, regardless of whether we can match them
        to NIST. By default False.
    constantT : str or numeric, optional
        Constant temperature in units of K, by default None
    double_tau : bool, optional
        Whether to use the 'double optical depths' command. This command is useful
        for 1D simulations, ensuring that radiation does not escape the atmosphere
        at the back-side into the planet core. By default False
    hcfrac : str or numeric, optional
        Threshold fraction of the total heating/cooling rate for which the .heat and
        .cool files should save agents. Cloudy's default is 0.05, so that individual
        heating and cooling processes contributing <0.05 of the total are not saved.
        By default None, so that Cloudy's default of 0.05 is used.
    cloudy_version : str, optional
        Major Cloudy release version, used only in combination with the denspecies
        argument, by default "17".
    """
    if outfiles is None:
        outfiles = []
    if denspecies is None:
        denspecies = []

    if denspecies != []:
        assert ".den" in outfiles and ".en" in outfiles
    if ".den" in outfiles or ".en" in outfiles:
        assert ".den" in outfiles and ".en" in outfiles
    if constantT is not None:
        assert not np.any(tlaw is not None)

    copyfile(oldsimname + ".in", newsimname + ".in")

    with open(newsimname + ".in", "a", encoding="utf-8") as f:
        if set_thickness:
            f.write(
                "\nstop thickness "
                + "{:.7f}".format(dlaw[-1, 0])
                + "\t#last dlaw point"
            )
        if ".ovr" in outfiles:
            f.write('\nsave overview ".ovr" last')
        if ".cool" in outfiles:
            f.write('\nsave cooling ".cool" last')
        if ".coolH2" in outfiles:
            f.write('\nsave H2 cooling ".coolH2" last')
        if ".heat" in outfiles:
            f.write('\nsave heating ".heat" last')
        if ".con" in outfiles:
            f.write('\nsave continuum ".con" last units angstrom')
        if (
            ".den" in outfiles
        ):  # then ".en" is always there as well due to the assertion above
            if denspecies != []:
                f.write(
                    '\nsave species densities last ".den"\n'
                    + speciesstring(
                        denspecies,
                        selected_levels=selected_den_levels,
                        cloudy_version=cloudy_version,
                    )
                    + "\nend"
                )
                f.write(
                    '\nsave species energies last ".en"\n'
                    + speciesstring(
                        denspecies,
                        selected_levels=selected_den_levels,
                        cloudy_version=cloudy_version,
                    )
                    + "\nend"
                )
        if constantT is not None:
            f.write("\nconstant temperature t= " + str(constantT) + " linear")
        if double_tau:
            f.write(
                "\ndouble optical depths    #so radiation does not escape into planet "
                "core freely"
            )
        if hcfrac:
            f.write(
                "\nset WeakHeatCool "
                + str(hcfrac)
                + " #for .heat and .cool output files"
            )
        if othercommands is not None:
            f.write("\n" + othercommands)
        if np.any(dlaw is not None):
            dlaw = remove_duplicates(dlaw, "1.7f")
            f.write("\n# ========= density law    ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ndlaw table depth\n")
            np.savetxt(f, dlaw, fmt="%1.7f")
            f.write(
                "{:.7f}".format(dlaw[-1, 0] + 0.1) + " " + "{:.7f}".format(dlaw[-1, 1])
            )
            f.write("\nend of dlaw #last point added to prevent roundoff")
        if np.any(tlaw is not None):
            tlaw = remove_duplicates(tlaw, "1.7f")
            f.write("\n# ========= temperature law    ============")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ntlaw table depth\n")
            np.savetxt(f, tlaw, fmt="%1.7f")
            f.write(
                "{:.7f}".format(tlaw[-1, 0] + 0.1) + " " + "{:.7f}".format(tlaw[-1, 1])
            )
            f.write("\nend of tlaw #last point added to prevent roundoff")
        if np.any(cextra is not None):
            cextra = remove_duplicates(cextra, "1.7f")
            f.write("\n# ========= cextra law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ncextra table depth\n")
            np.savetxt(f, cextra, fmt="%1.7f")
            f.write(
                "{:.7f}".format(cextra[-1, 0] + 0.1)
                + " "
                + "{:.7f}".format(cextra[-1, 1])
            )
            f.write("\nend of cextra #last point added to prevent roundoff")
        if np.any(hextra is not None):
            hextra = remove_duplicates(hextra, "1.7f")
            f.write("\n# ========= hextra law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\nhextra table depth\n")
            np.savetxt(f, hextra, fmt="%1.7f")
            f.write(
                "{:.7f}".format(hextra[-1, 0] + 0.1)
                + " "
                + "{:.7f}".format(hextra[-1, 1])
            )
            f.write("\nend of hextra #last point added to prevent roundoff")


def write_Cloudy_in(
    simname,
    title=None,
    flux_scaling=None,
    SED=None,
    set_thickness=True,
    dlaw=None,
    tlaw=None,
    cextra=None,
    hextra=None,
    othercommands=None,
    overwrite=False,
    iterate="convergence",
    nend=3000,
    outfiles=None,
    denspecies=None,
    selected_den_levels=False,
    constantT=None,
    double_tau=False,
    cosmic_rays=False,
    zdict=None,
    hcfrac=None,
    comments=None,
    cloudy_version="17",
):
    """
    Writes a Cloudy input file for simulating an exoplanet atmosphere.

    Parameters
    ----------
    simname : str
        Full path + name of the Cloudy simulation, without the file extension.
    title : str, optional
        Title of simulation, by default None
    flux_scaling : tuple, optional
        Normalization of the SED, as a tuple with the monochromatic flux
        and energy in Ryd where it is specified, by default None
    SED : str, optional
        Name of a SED file located in $CLOUDY_PATH/data/SED/, by default None
    set_thickness : bool, optional
        Whether to include a command that ends the simulation at a depth equal
        to the length of the dlaw, by default True
    dlaw : numpy.ndarray, optional
        Hydrogen number density in units of cm-3, as a 2D array where dlaw[:,0]
        specifies the log10 of the depth into the cloud in cm, and dlaw[:,1]
        specifies the log10 of the hydrogen number density in units of cm-3, by default None
    tlaw : numpy.ndarray, optional
        Temperature in units of K as a 2D array where tlaw[:,0]
        specifies the log10 of the depth into the cloud in cm, and tlaw[:,1]
        specifies the log10 of the temperature in units of K, by default None
    cextra : numpy.ndarray, optional
        Extra unspecified cooling in units of erg s-1 cm-3, as a 2D array where
        cextra[:,0] specifies the log10 of the depth into the cloud in cm,
        and cextra[:,1] specifies the log10 of the cooling rate in units of
        erg s-1 cm-3, by default None
    hextra : numpy.ndarray, optional
        Extra unspecified heating in units of erg s-1 cm-3, as a 2D array where
        hextra[:,0] specifies the log10 of the depth into the cloud in cm,
        and hextra[:,1] specifies the log10 of the heating rate in units of
        erg s-1 cm-3, by default None
    othercommands : str, optional
        String to include in the input file. Any command not otherwise supported
        by this function can be included here, by default None
    overwrite : bool, optional
        Whether to overwrite the simname if it already exists, by default False
    iterate : str or int, optional
        Argument to Cloudy's 'iterate' command, either a number or 'convergence',
        by default 'convergence'
    nend : int, optional
        Argument to Cloudy's 'set nend' command, which sets the maximum number of Cloudy
        cells. Cloudy's default is 1400 which can often be too few. For this function,
        by default 3000.
    outfiles : list, optional
        List of file extensions indicating which Cloudy output to save. For example,
        include '.heat' to include the 'save heating' command, by default ['.ovr', '.cool']
    denspecies : list, optional
        List of atomic/ionic species for which to save densities and energies, which
        are needed to do radiative transfer. The list can easily be created by the
        get_specieslist() function. By default [].
    selected_den_levels : bool, optional
        If True, only energy levels up to the number that can be matched to NIST
        will be included in the 'save densities' command. If False, all energy levels
        of each species will be included, regardless of whether we can match them
        to NIST. By default False.
    constantT : str or numeric, optional
        Constant temperature in units of K, by default None
    double_tau : bool, optional
        Whether to use the 'double optical depths' command. This command is useful
        for 1D simulations, ensuring that radiation does not escape the atmosphere
        at the back-side into the planet core. By default False
    cosmic_rays : bool, optional
        Whether to include cosmic rays, by default False
    zdict : dict, optional
        Dictionary with the scale factors of all elements relative
        to a solar composition. Can be easily created with get_zdict().
        Default is None, which results in a solar composition.
    hcfrac : str or numeric, optional
        Threshold fraction of the total heating/cooling rate for which the .heat and
        .cool files should save agents. Cloudy's default is 0.05, so that individual
        heating and cooling processes contributing <0.05 of the total are not saved.
        By default None, so that Cloudy's default of 0.05 is used.
    comments : str, optional
        Comments to write at the top of the input file. Make sure to include hashtags
        in the string, by default None
    cloudy_version : str, optional
        Major Cloudy release version, used only in combination with the denspecies
        argument, by default "17".
    """
    if outfiles is None:
        outfiles = [".ovr", ".cool"]

    if denspecies is None:
        denspecies = []

    assert (
        flux_scaling is not None
    )  # we need this to proceed. Give in format [F,E] like nuF(nu) = F at E Ryd
    assert SED is not None
    if denspecies != []:
        assert ".den" in outfiles and ".en" in outfiles
    if ".den" in outfiles or ".en" in outfiles:
        assert ".den" in outfiles and ".en" in outfiles and denspecies != []
    if not overwrite:
        assert not os.path.isfile(simname + ".in")
    if constantT is not None:
        assert not np.any(tlaw is not None)

    with open(simname + ".in", "w", encoding="utf-8") as f:
        if comments is not None:
            f.write(comments + "\n")
        if title is not None:
            f.write("title " + title)
        f.write("\n# ========= input spectrum ================")
        f.write(
            "\nnuF(nu) = "
            + str(flux_scaling[0])
            + " at "
            + str(flux_scaling[1])
            + " Ryd"
        )
        f.write('\ntable SED "' + SED + '"')
        if cosmic_rays:
            f.write("\ncosmic rays background")
        f.write("\n# ========= chemistry      ================")
        f.write("\n# solar abundances and metallicity is standard")
        if zdict is not None:
            for element in zdict.keys():
                if zdict[element] == 0.0:
                    f.write("\nelement " + element_names[element] + " off")
                elif (
                    zdict[element] != 1.0
                ):  # only write it to Cloudy if the scale factor is not 1
                    f.write(
                        "\nelement scale factor "
                        + element_names[element]
                        + " "
                        + str(zdict[element])
                    )
        f.write("\n# ========= other          ================")
        if nend is not None:
            f.write(
                "\nset nend "
                + str(nend)
                + "   #models at high density need >1400 zones"
            )
        f.write("\nset temperature floor 5 linear")
        f.write("\nstop temperature off     #otherwise it stops at 1e4 K")
        if iterate == "convergence":
            f.write("\niterate to convergence")
        else:
            f.write("niterate " + str(iterate))
        f.write("\nprint last iteration")
        if set_thickness:
            f.write(
                "\nstop thickness "
                + "{:.7f}".format(dlaw[-1, 0])
                + "\t#last dlaw point"
            )
        if constantT is not None:
            f.write("\nconstant temperature t= " + str(constantT) + " linear")
        if double_tau:
            f.write(
                "\ndouble optical depths    #so radiation does not escape into planet core freely"
            )
        if hcfrac:
            f.write(
                "\nset WeakHeatCool "
                + str(hcfrac)
                + " #for .heat and .cool output files"
            )
        if othercommands is not None:
            f.write("\n" + othercommands)
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
            f.write('\nsave continuum ".con" last units angstrom')
        if ".den" in outfiles:  # then ".en" is always there as well.
            f.write(
                '\nsave species densities last ".den"\n'
                + speciesstring(
                    denspecies,
                    selected_levels=selected_den_levels,
                    cloudy_version=cloudy_version,
                )
                + "\nend"
            )
            f.write(
                '\nsave species energies last ".en"\n'
                + speciesstring(
                    denspecies,
                    selected_levels=selected_den_levels,
                    cloudy_version=cloudy_version,
                )
                + "\nend"
            )
        if dlaw is not None:
            dlaw = remove_duplicates(dlaw, "1.7f")
            f.write("\n# ========= density law    ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ndlaw table depth\n")
            np.savetxt(f, dlaw, fmt="%1.7f")
            f.write(
                "{:.7f}".format(dlaw[-1, 0] + 0.1) + " " + "{:.7f}".format(dlaw[-1, 1])
            )
            f.write("\nend of dlaw #last point added to prevent roundoff")
        if tlaw is not None:
            tlaw = remove_duplicates(tlaw, "1.7f")
            f.write("\n# ========= temperature law    ============")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ntlaw table depth\n")
            np.savetxt(f, tlaw, fmt="%1.7f")
            f.write(
                "{:.7f}".format(tlaw[-1, 0] + 0.1) + " " + "{:.7f}".format(tlaw[-1, 1])
            )
            f.write("\nend of tlaw #last point added to prevent roundoff")
        if cextra is not None:
            cextra = remove_duplicates(cextra, "1.7f")
            f.write("\n# ========= cextra law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\ncextra table depth\n")
            np.savetxt(f, cextra, fmt="%1.7f")
            f.write(
                "{:.7f}".format(cextra[-1, 0] + 0.1)
                + " "
                + "{:.7f}".format(cextra[-1, 1])
            )
            f.write("\nend of cextra #last point added to prevent roundoff")
        if hextra is not None:
            hextra = remove_duplicates(hextra, "1.7f")
            f.write("\n# ========= hextra law     ================")
            f.write("\n#depth sets distances from edge of cloud")
            f.write("\nhextra table depth\n")
            np.savetxt(f, hextra, fmt="%1.7f")
            f.write(
                "{:.7f}".format(hextra[-1, 0] + 0.1)
                + " "
                + "{:.7f}".format(hextra[-1, 1])
            )
            f.write("\nend of hextra #last point added to prevent roundoff")


def insertden_Cloudy_in(
    simname, denspecies, selected_den_levels=True, rerun=False, cloudy_version="17"
):
    """
    Takes a Cloudy .in input file and adds extra species to the
    'save species densities' command. This is useful for example if you first went
    through the convergeT_parker.py temperature convergence scheme,
    but later want to add additional species to the 'converged' simulation.

    Parameters
    ----------
    simname : str
        Full path + name of the Cloudy simulation, without the file extension.
    denspecies : list, optional
        List of atomic/ionic species for which to save densities and energies, which
        are needed to do radiative transfer. The list can easily be created by the
        get_specieslist() function.
    selected_den_levels : bool, optional
        If True, only energy levels up to the number that can be matched to NIST
        will be included in the 'save densities' command. If False, all energy levels
        of each species will be included, regardless of whether we can match them
        to NIST. By default True.
    rerun : bool, optional
        Whether to run the new Cloudy input file, by default False
    cloudy_version : str, optional
        Major Cloudy release version, by default "17".

    Raises
    ------
    ValueError
        If there are multiple 'save species densities' commands in the Cloudy input file.
    """

    with open(simname + ".in", "r", encoding="utf-8") as f:
        oldcontent = f.readlines()

    newcontent = oldcontent
    indices = [i for i, s in enumerate(oldcontent) if "save species densities" in s]
    if len(indices) == 0:  # then there is no 'save species densities' command yet
        newcontent.append(
            '\nsave species densities last ".den"\n'
            + speciesstring(
                denspecies,
                selected_levels=selected_den_levels,
                cloudy_version=cloudy_version,
            )
            + "\nend"
        )
        newcontent.append(
            '\nsave species energies last ".en"\n'
            + speciesstring(
                denspecies,
                selected_levels=selected_den_levels,
                cloudy_version=cloudy_version,
            )
            + "\nend"
        )

    elif (
        len(indices) == 1
    ):  # then there already is a 'save species densities' command with some species
        for sp in denspecies.copy():
            if (
                len([i for i, s in enumerate(oldcontent) if sp + "[" in s]) != 0
            ):  # check if this species is already in the file
                denspecies.remove(sp)
                print(sp, "was already in the .in file so I did not add it again.")
        if len(denspecies) >= 1:
            newcontent.insert(
                indices[0] + 1,
                speciesstring(
                    denspecies,
                    selected_levels=selected_den_levels,
                    cloudy_version=cloudy_version,
                )
                + "\n",
            )
            # also add them to the 'save species energies' list
            indices2 = [
                i for i, s in enumerate(oldcontent) if "save species energies" in s
            ]
            newcontent.insert(
                indices2[0] + 1,
                speciesstring(
                    denspecies,
                    selected_levels=selected_den_levels,
                    cloudy_version=cloudy_version,
                )
                + "\n",
            )
        else:
            return

    else:
        raise ValueError(
            "There are multiple 'save species densities' commands in the .in file. This shouldn't be the case, please check."
        )

    newcontent = "".join(newcontent)  # turn list into string
    with open(simname + ".in", "w", encoding="utf-8") as f:  # overwrite the old file
        f.write(newcontent)

    if rerun:
        run_Cloudy(simname)


# ######################################
# ##########     CLASSES     ###########
# ######################################


class Parker:
    """
    Class that stores a Parker wind profile and its parameters.
    """

    def __init__(
        self, plname, T, Mdot, pdir, fH=None, zdict=None, SED=None, readin=True
    ):
        """
        Parameters
        ----------
        plname : str
            Name of the planet
        T : str or numeric
            Temperature in units of K.
        Mdot : str or numeric
            log10 of the mass-loss rate in units of g s-1.
        pdir : str
            Directory as $SUNBATHER_PROJECT_PATH/parker_profiles/*plname*/*pdir*/
            where the isothermal parker wind density and velocity profiles are saved.
            Different folders may exist there for a given planet, to separate for example profiles
            with different assumptions such as stellar SED/semi-major axis/composition.
        fH : float, optional
            Hydrogen abundance fraction, in case of a H/He composition, by default None
        zdict : dict, optional
            Dictionary with the scale factors of all elements relative
            to a solar composition. Can be easily created with get_zdict().
            Default is None, which results in a solar composition.
        SED : str, optional
            Stellar SED name, by default None
        readin : bool, optional
            Whether to read in the atmospheric profile, by default True
        """

        self.plname = plname
        self.T = int(T)
        if isinstance(Mdot, str):
            self.Mdot = Mdot
            self.Mdotf = float(Mdot)
        elif isinstance(Mdot, (float, int)):
            self.Mdot = "%.3f" % Mdot
            self.Mdotf = Mdot
        if fH is not None:
            self.fH = fH
        if zdict is not None:
            self.zdict = zdict
        if SED is not None:
            self.SED = SED
        if readin:
            self.prof = read_parker(plname, T, Mdot, pdir)


class Planet:
    """
    Class that stores planet/star parameters.
    """

    def __init__(
        self,
        name,
        fullname=None,
        R=None,
        Rstar=None,
        a=None,
        M=None,
        Mstar=None,
        bp=None,
        SEDname=None,
    ):
        """
        Parameters
        ----------
        name : str
            Planet name. Typically does not include spaces. If this name appears in the
            $SUNBATHER_PROJECT_PATH/planets.txt file, those parameters are automatically
            fetched. Specific values can then be changed by providing them as arguments.
            If the planet name does not appear in $SUNBATHER_PROJECT_PATH/planets.txt,
            all parameters must be provided upon initialization.
        fullname : str, optional
            Full planet name, can include spaces and other special characters, by default None
        R : float, optional
            Planet radius in units of cm, by default None
        Rstar : float, optional
            Star radius in units of cm, by default None
        a : float, optional
            Semi-major axis in units of cm, by default None
        M : float, optional
            Planet mass in units of g, by default None
        Mstar : float, optional
            Star mass in units of g, by default None
        bp : float, optional
            Transit impact parameter, in units of the star radius, by default None
        SEDname : str, optional
            Stellar SED name, by default None
        """

        # check if we can fetch planet parameters from planets.txt:
        planets_file = get_planets_file()
        if (
            name in planets_file["name"].values
            or name in planets_file["full name"].values
        ):
            this_planet = planets_file[
                (planets_file["name"] == name) | (planets_file["full name"] == name)
            ]
            assert (
                len(this_planet) == 1
            ), "Multiple entries were found in planets.txt for this planet name."

            self.name = this_planet["name"].values[0]
            self.fullname = this_planet["full name"].values[0]
            self.R = this_planet["R [RJ]"].values[0] * RJ  # in cm
            self.Rstar = this_planet["Rstar [Rsun]"].values[0] * Rsun  # in cm
            self.a = this_planet["a [AU]"].values[0] * AU  # in cm
            self.M = this_planet["M [MJ]"].values[0] * MJ  # in g
            self.Mstar = this_planet["Mstar [Msun]"].values[0] * Msun  # in g
            self.bp = this_planet["transit impact parameter"].values[0]  # dimensionless
            self.SEDname = (
                this_planet["SEDname"].values[0].strip()
            )  # strip to remove whitespace from beginning and end

            # if any specified, overwrite values read from planets.txt
            if fullname is not None:
                self.fullname = fullname
            if R is not None:
                self.R = R
            if Rstar is not None:
                self.Rstar = Rstar
            if a is not None:
                self.a = a
            if M is not None:
                self.M = M
            if Mstar is not None:
                self.Mstar = Mstar
            if bp is not None:
                self.bp = bp
            if SEDname is not None:
                self.SEDname = SEDname

        else:
            assert (
                fullname is not None
                and R is not None
                and Rstar is not None
                and a is not None
                and M is not None
                and Mstar is not None
                and bp is not None
                and SEDname is not None
            ), "I'm trying to make a Planet that is not in the planets.txt file, but I don't have all required arguments."
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
        self.__update_Kp()

    def set_var(
        self,
        name=None,
        fullname=None,
        R=None,
        Rstar=None,
        a=None,
        M=None,
        Mstar=None,
        bp=None,
        SEDname=None,
    ):
        """
        Change planet/star parameters after initialization.
        """

        if name is not None:
            self.name = name
        if R is not None:
            self.R = R
            self.__update_phi()
        if Rstar is not None:
            self.Rstar = Rstar
        if a is not None:
            self.a = a
            self.__update_Rroche()
            self.__update_Kp()
        if M is not None:
            self.M = M
            self.__update_phi()
            self.__update_Rroche()
            self.__update_Kp()
        if Mstar is not None:
            self.Mstar = Mstar
            self.__update_Rroche()
            self.__update_Kp()
        if bp is not None:
            self.bp = bp
        if SEDname is not None:
            self.SEDname = SEDname

    def __update_phi(self):
        """
        Tries to set/update the gravitational potential.
        """

        if (self.M is not None) and (self.R is not None):
            self.phi = G * self.M / self.R
        else:
            self.phi = None

    def __update_Rroche(self):
        """
        Tries to set/update the Roche radius.
        """

        if (self.a is not None) and (self.M is not None) and (self.Mstar is not None):
            self.Rroche = roche_radius(self.a, self.M, self.Mstar)
        else:
            self.Rroche = None

    def __update_Kp(self):
        """
        Tries to set/update the orbital velocity semi-amplitude.
        """

        if (self.a is not None) and (self.M is not None) and (self.Mstar is not None):
            self.Kp = np.sqrt(G * (self.M + self.Mstar) / self.a)
        else:
            self.Kp = None

    def print_params(self):
        """
        Prints out all parameters in read-friendly format.
        """

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
            print(
                f"Roche radius: {self.Rroche} cm, {self.Rroche / RJ} RJ, "
                f"{self.Rroche / self.R} Rp"
            )
        if self.phi is not None:
            print(f"log10(Gravitational potential): {np.log10(self.phi)} log10(erg/g)")
        if self.Kp is not None:
            print(
                f"Orbital velocity semi-amplitude: {self.Kp} cm/s, {self.Kp/1e5} km/s"
            )

    def plot_transit_geometry(self, phase=0.0, altmax=None):
        """
        Plots a schematic of the transit geometry. Helpful to understand where the
        planet and its atmosphere are relative to the stellar disk, for a given planet
        impact parameter and phase. The dotted line shows the planet Roche radius. The
        altmax argument can be used to draw another dashed line in units of the planet
        radius, for example the extent of the sunbather simulation (typically 8 Rp).
        """

        fig, ax = plt.subplots(1)
        title = ""
        # draw star
        ax.plot(
            self.Rstar * np.cos(np.linspace(0, 2 * np.pi, 100)),
            self.Rstar * np.sin(np.linspace(0, 2 * np.pi, 100)),
            c="k",
            zorder=0,
        )
        ax.text(
            1 / np.sqrt(2) * self.Rstar,
            -1 / np.sqrt(2) * self.Rstar,
            r"$R_s$",
            color="k",
            ha="left",
            va="top",
            zorder=0,
        )

        # draw planet
        pl_zorder = -1 if (phase % 1 > 0.25 and phase % 1 < 0.75) else 1
        ax.plot(
            self.a * np.sin(2 * np.pi * phase)
            + self.R * np.cos(np.linspace(0, 2 * np.pi, 100)),
            self.bp * self.Rstar + self.R * np.sin(np.linspace(0, 2 * np.pi, 100)),
            c="b",
            zorder=pl_zorder,
        )
        ax.text(
            self.a * np.sin(2 * np.pi * phase) + 1 / np.sqrt(2) * self.R,
            self.bp * self.Rstar - 1 / np.sqrt(2) * self.R,
            r"$R_P$",
            color="b",
            ha="left",
            va="top",
            zorder=pl_zorder,
        )

        # draw planet vy direction
        if phase % 1 > 0.75 or phase % 1 < 0.25:
            ax.text(
                self.a * np.sin(2 * np.pi * phase) + self.R,
                self.bp * self.Rstar,
                r"$\rightarrow$",
                color="b",
                ha="left",
                va="top",
                zorder=pl_zorder,
            )
            title = f"Phase: {phase} mod 1 = {phase % 1}"
        elif phase % 1 > 0.25 and phase % 1 < 0.75:
            ax.text(
                self.a * np.sin(2 * np.pi * phase) - self.R,
                self.bp * self.Rstar,
                r"$\leftarrow$",
                color="b",
                ha="right",
                va="top",
                zorder=pl_zorder,
            )
            title = f"Phase: {phase} mod 1 = {phase % 1} (planet behind star)"
        else:  # at 0.25 or 0.75, only vx velocity
            pass

        # draw Roche indication
        if self.Rroche is not None:
            ax.plot(
                self.a * np.sin(2 * np.pi * phase)
                + self.Rroche * np.cos(np.linspace(0, 2 * np.pi, 100)),
                self.bp * self.Rstar
                + self.Rroche * np.sin(np.linspace(0, 2 * np.pi, 100)),
                c="b",
                linestyle="dotted",
            )
            ax.text(
                self.a * np.sin(2 * np.pi * phase) + 1 / np.sqrt(2) * self.Rroche,
                self.bp * self.Rstar - 1 / np.sqrt(2) * self.Rroche,
                r"$R_{Roche}$",
                color="b",
                ha="left",
                va="top",
                zorder=pl_zorder,
            )

        # draw altmax indication
        if altmax is not None:
            ax.plot(
                self.a * np.sin(2 * np.pi * phase)
                + altmax * self.R * np.cos(np.linspace(0, 2 * np.pi, 100)),
                self.bp * self.Rstar
                + altmax * self.R * np.sin(np.linspace(0, 2 * np.pi, 100)),
                c="b",
                linestyle="dashed",
            )
            ax.text(
                self.a * np.sin(2 * np.pi * phase) + altmax / np.sqrt(2) * self.R,
                self.bp * self.Rstar - altmax / np.sqrt(2) * self.R,
                "altmax",
                color="b",
                ha="left",
                va="top",
                zorder=pl_zorder,
            )

        plt.axis("equal")
        ax.set_xlabel("y [cm]")
        ax.set_ylabel("z [cm]")
        ax.set_title(title)
        plt.show()

    def max_T0(self, mu_bar=1.0):
        """
        Calculates the maximum isothermal temperature T0 that the Parker wind can have,
        for it to still be transonic. If T0 is higher than this value,
        Rp > Rs which breaks the assumption of the Parker wind.
        See Vissapragada et al. (2024) on TOI-1420 b.
        """

        maxT0 = G * self.M * mH * mu_bar / (2 * self.R * k)

        return maxT0


class Sim:
    """
    Loads the output of a Cloudy simulation. Tailored towards simulations of
    an escaping exoplanet atmosphere.
    """

    def __init__(
        self,
        simname,
        altmax=None,
        proceedFail=False,
        files="all",
        planet=None,
        parker=None,
    ):
        """
        Parameters
        ----------
        simname : str
            Full path + simulation name excluding file extension.
        altmax : int, optional
            Maximum altitude of the simulation in units of the planet radius. Will also
            be automatically read from the input file if written as a comment.  By
            default None.
        proceedFail : bool, optional
            Whether to proceed loading the simulation if Cloudy did not exit OK, by
            default False
        files : list, optional
            List of file extensions of Cloudy output to load. For example, include
            '.heat' to read the output of the 'save heating' command.  By default
            ['all'], which reads in all output files present that are understood by this
            class.
        planet : Planet, optional
            Object storing planet parameters. Will also be automatically read from the
            input file if written as a comment. By default None.
        parker : Parker, optional
            Object storing the isothermal Parker wind atmospheric profiles and
            parameters. Will also be automatically read from the input file if written
            as a comment. By default None.

        Raises
        ------
        TypeError
            If the simname argument is not a string.
        TypeError
            If a Cloudy version was used that is not supported by sunbather.
        FileNotFoundError
            If the Cloudy simulation did not exit OK and proceedFail = False.
        TypeError
            If the altmax argument is not numeric.
        """
        if isinstance(files, str):
            files = [files]

        if not isinstance(simname, str):
            raise TypeError("simname must be set to a string")
        self.simname = simname

        # check the Cloudy version, and if the simulation did not crash.
        _succesful = False
        with open(simname + ".out", "r", encoding="utf-8") as f:
            _outfile_content = f.read()
            _succesful = "Cloudy exited OK" in _outfile_content

            if "Cloudy 17" in _outfile_content:
                self.cloudy_version = "17"
            elif "Cloudy 23" in _outfile_content or "Cloudy (c23" in _outfile_content:
                self.cloudy_version = "23"
            elif _succesful:
                raise TypeError(
                    f"This simulation did not use Cloudy v17 or v23, which are the "
                    f"only supported versions: {simname}"
                )
        if not _succesful and not proceedFail:
            raise FileNotFoundError(
                f"This simulation went wrong: {simname} Check the .out file!"
            )

        # read the .in file to extract some sim info like changes to the chemical
        # composition and altmax
        self.disabled_elements = []
        zelem = {}
        _parker_T, _parker_Mdot, _parker_dir = None, None, None  # temp variables
        with open(simname + ".in", "r", encoding="utf-8") as f:
            for line in f:
                if (
                    line[0] == "#"
                ):  # then it is a comment written by sunbather, extract info:
                    # check if a planet was defined
                    if "plname" in line:
                        self.p = Planet(line.split("=")[-1].strip("\n"))

                    # check if a Parker profile was defined
                    if "parker_T" in line:
                        _parker_T = int(line.split("=")[-1].strip("\n"))
                    if "parker_Mdot" in line:
                        _parker_Mdot = line.split("=")[-1].strip("\n")
                    if "parker_dir" in line:
                        _parker_dir = line.split("=")[-1].strip("\n")

                    # check if an altmax was defined
                    if "altmax" in line:
                        self.altmax = int(line.split("=")[1].strip("\n"))

                # read SED
                if "table SED" in line:
                    self.SEDname = line.split('"')[1]

                # read chemical composition
                if "element scale factor" in line.rstrip():
                    zelem[element_symbols[line.split(" ")[3]]] = float(
                        line.rstrip().split(" ")[-1]
                    )
                elif "element" in line.rstrip() and "off" in line.rstrip():
                    self.disabled_elements.append(element_symbols[line.split(" ")[1]])
                    zelem[element_symbols[line.split(" ")[1]]] = 0.0

        # set zdict and abundances as attributes
        self.zdict = get_zdict(zelem=zelem)
        self.abundances = get_abundances(zdict=self.zdict)

        # overwrite/set manually given Planet object
        if planet is not None:
            assert isinstance(planet, Planet)
            if hasattr(self, "p"):
                warnings.warn(
                    "I had already read out the Planet object from the .in file, but I "
                    "will overwrite that with the object you have given."
                )
            self.p = planet

        # check if the SED of the Planet object matches the SED of the Cloudy simulation
        if hasattr(self, "p") and hasattr(self, "SEDname"):
            if self.p.SEDname != self.SEDname:
                warnings.warn(
                    f"I read in the .in file that the SED used is {self.SEDname} which "
                    f"is different from the one of your Planet object. "
                    f"I will change the .SEDname attribute of the Planet object to "
                    f"match the one actually used in the simulation. Are you "
                    f"sure that also the associated Parker wind profile is correct?"
                )
                self.p.set_var(SEDname=self.SEDname)

        # try to set a Parker object if the .in file had the required info for that
        if (
            hasattr(self, "p")
            and (_parker_T is not None)
            and (_parker_Mdot is not None)
            and (_parker_dir is not None)
        ):
            self.par = Parker(self.p.name, _parker_T, _parker_Mdot, _parker_dir)

        # overwrite/set manually given Parker object
        if parker is not None:
            assert isinstance(parker, Parker)
            if hasattr(self, "par"):
                warnings.warn(
                    "I had already read out the Parker object from the .in file, but I "
                    "will overwrite that with the object you have given."
                )
            self.par = parker

        # overwrite/set manually given altmax
        if altmax is not None:
            if not isinstance(altmax, (float, int)):
                # can it actually be a float? I'm not sure if the code can handle it -
                # check and try.
                raise TypeError("altmax must be set to a float or int")
            if hasattr(self, "altmax"):
                if self.altmax != altmax:
                    warnings.warn(
                        "I read the altmax from the .in file, but the value you have "
                        "explicitly passed is different. "
                        "I will use your value, but please make sure it is correct."
                    )
            self.altmax = altmax

        # temporary variables for adding the alt-columns to the pandas dataframes
        _Rp, _altmax = None, None
        if hasattr(self, "p") and hasattr(self, "altmax"):
            _Rp = self.p.R
            _altmax = self.altmax

        # read in the Cloudy simulation files
        self.simfiles = []
        for simfile in glob.glob(simname + ".*", recursive=True):
            filetype = simfile.split(".")[-1]
            if filetype == "ovr" and ("ovr" in files or "all" in files):
                self.ovr = process_overview(
                    self.simname + ".ovr",
                    Rp=_Rp,
                    altmax=_altmax,
                    abundances=self.abundances,
                )
                self.simfiles.append("ovr")
            if filetype == "con" and ("con" in files or "all" in files):
                self.con = process_continuum(self.simname + ".con")
                self.simfiles.append("con")
            if filetype == "heat" and ("heat" in files or "all" in files):
                self.heat = process_heating(
                    self.simname + ".heat",
                    Rp=_Rp,
                    altmax=_altmax,
                    cloudy_version=self.cloudy_version,
                )
                self.simfiles.append("heat")
            if filetype == "cool" and ("cool" in files or "all" in files):
                self.cool = process_cooling(
                    self.simname + ".cool",
                    Rp=_Rp,
                    altmax=_altmax,
                    cloudy_version=self.cloudy_version,
                )
                self.simfiles.append("cool")
            if filetype == "coolH2" and ("coolH2" in files or "all" in files):
                self.coolH2 = process_coolingH2(
                    self.simname + ".coolH2", Rp=_Rp, altmax=_altmax
                )
                self.simfiles.append("coolH2")
            if filetype == "den" and ("den" in files or "all" in files):
                self.den = process_densities(
                    self.simname + ".den", Rp=_Rp, altmax=_altmax
                )
                self.simfiles.append("den")
            if filetype == "en" and ("en" in files or "all" in files):
                self.en = process_energies(
                    self.simname + ".en", cloudy_version=self.cloudy_version
                )
                self.simfiles.append("en")

        # set the velocity structure in .ovr if we have an associated Parker profile -
        # needed for radiative transfer
        if hasattr(self, "par") and hasattr(self, "ovr"):
            if hasattr(self.par, "prof") and hasattr(self.ovr, "alt"):
                Sim.addv(self, self.par.prof.alt, self.par.prof.v)

    def get_simfile(self, simfile):
        """
        Returns the output of the requested simulation output file.
        These can also be accessed as an attribute,
        for example mysim.ovr or mysim.cool for a Sim object called mysim
        """

        if simfile not in self.simfiles:
            raise FileNotFoundError(
                "This simulation does not have a", simfile, "output file."
            )

        if simfile == "ovr":
            return self.ovr
        if simfile == "con":
            return self.con
        if simfile == "heat":
            return self.heat
        if simfile == "cool":
            return self.cool
        if simfile == "coolH2":
            return self.coolH2
        if simfile == "den":
            return self.den
        if simfile == "en":
            return self.en
        if simfile == "ionFe":
            return self.ionFe
        if simfile == "ionNa":
            return self.ionNa
        return None

    def add_parker(self, parker):
        """
        Adds a Parker profile object to the Sim, in case it wasn't added upon
        initialization.
        """

        assert isinstance(parker, Parker)
        self.par = parker
        if hasattr(parker, "prof"):
            Sim.addv(self, parker.prof.alt, parker.prof.v)

    def addv(self, alt, v, delete_negative=True):
        """
        Adds a velocity profile in cm s-1 on the Cloudy grid. Will be added to the .ovr
        file, but also available as the .v attribute for backwards compatibility of
        sunbather.  Called automatically when adding a Parker object to the Sim.
        """

        assert "ovr" in self.simfiles, "Simulation must have a 'save overview .ovr file"
        assert "alt" in self.ovr.columns, (
            "The .ovr file must have an altitude column (which in turn requires a "
            "known Rp and altmax)"
        )

        if delete_negative:
            v[v < 0.0] = 0.0

        self.ovr["v"] = interp1d(alt, v)(self.ovr.alt)

        vseries = pd.Series(index=self.ovr.alt.index, dtype=float)
        vseries[self.ovr.alt.index] = interp1d(alt, v)(self.ovr.alt)
        self.v = vseries
