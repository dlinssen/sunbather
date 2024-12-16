"""
ConvergeT_parker module of sunbather
"""
import sys
import multiprocessing
from shutil import copyfile
import time
import os
import re
import argparse
import traceback
import pandas as pd
import numpy as np

# sunbather imports
from sunbather import tools, solveT


def find_close_model(parentfolder, T, Mdot, tolT=2000, tolMdot=1.0):
    """
    Takes a parent folder where multiple 1D parker profiles have been ran,
    and for given T and Mdot it looks for another model that is already
    finished and closest to the given model, so that we can start our new
    simulation from that converged temperature structure. It returns the T and
    Mdot of the close converged folder, or None if there aren't any (within the
    tolerance).

    Parameters
    ----------
    parentfolder : str
        Parent folder containing sunbather simulations within folders with the
        parker_*T0*_*Mdot* name format.
    T : numeric
        Target isothermal temperature in units of K.
    Mdot : numeric
        log of the target mass-loss rate in units of g s-1.
    tolT : numeric, optional
        Maximum T0 difference with the target temperature, by default 2000 K
    tolMdot : numeric, optional
        Maximum log10(Mdot) difference with the target mass-loss rate, by
        default 1 dex

    Returns
    -------
    clconv : list
        [T0, Mdot] of the closest found finished model, or [None, None] if none
        were found within the tolerance.
    """

    pattern = re.compile(
        r"parker_\d+_\d+\.\d{3}$"
    )  # this is how folder names should be
    all_files_and_folders = os.listdir(parentfolder)
    allfolders = [
        os.path.join(parentfolder, folder) + "/"
        for folder in all_files_and_folders
        if pattern.match(folder) and os.path.isdir(os.path.join(parentfolder, folder))
    ]

    convergedfolders = (
        []
    )  # stores the T and Mdot values of all folders with 0.out files
    for folder in allfolders:
        if os.path.isfile(folder + "converged.out"):
            folderparams = folder.split("/")[-2].split("_")
            convergedfolders.append([int(folderparams[1]), float(folderparams[2])])

    if [
        int(T),
        float(Mdot),
    ] in convergedfolders:  # if the current folder is found, remove it
        convergedfolders.remove([int(T), float(Mdot)])

    if not convergedfolders:  # then we default to constant starting value
        clconv = [None, None]
    else:  # find closest converged profile
        dist = (
            lambda x, y: (x[0] - y[0]) ** 2 + (2000 * (x[1] - y[1])) ** 2
        )  # 1 order of magnitude Mdot is now 'equal weighted' to 2000K
        clconv = min(
            convergedfolders, key=lambda fol: dist(fol, [int(T), float(Mdot)])
        )  # closest converged [T, Mdot]
        if (np.abs(clconv[0] - int(T)) > tolT) or (
            np.abs(clconv[1] - float(Mdot)) > tolMdot
        ):
            clconv = [None, None]

    return clconv


def run_s(
    plname,
    Mdot,
    T,
    itno,
    fc,
    workingdir,
    SEDname,
    overwrite,
    startT,
    pdir,
    zdict=None,
    altmax=8,
    save_sp=None,
    constantT=False,
    maxit=16,
):
    """
    Solves for a nonisothermal temperature profile of a single isothermal
    Parker wind (density and velocity) profile.

    Parameters
    ----------
    plname : str
        Planet name (must have parameters stored in
        $SUNBATHER_PROJECT_PATH/planets.txt).
    Mdot : str or numeric
        log of the mass-loss rate in units of g s-1.
    T : str or int
        Temperature in units of g s-1.
    itno : int
        Iteration number to start from (can only be different from 1
        if this same model has been ran before, and then also
        overwrite = True needs to be set). If value is 0, will automatically
        look for the highest iteration number to start from.
    fc : numeric
        H/C convergence factor, see Linssen et al. (2024). A sensible value is 1.1.
    workingdir : str
        Directory as $SUNBATHER_PROJECT_PATH/sims/1D/planetname/*workingdir*/
        where the temperature profile will be solved. A folder named
        parker_*T*_*Mdot*/ will be made there.
    SEDname : str
        Name of SED file to use. If SEDname='real', we use the name as
        given in the planets.txt file, but if SEDname is something else,
        we advice to use a separate dir folder for this.
    overwrite : bool
        Whether to overwrite if this simulation already exists.
    startT : str
        Either 'constant', 'free' or 'nearby'. Sets the initial
        temperature profile guessed/used for the first iteration.
        'constant' sets it equal to the parker wind isothermal value.
        'free' lets Cloudy solve it, so you will get the radiative equilibrium
        structure.  'nearby' looks in the workingdir folder for previously solved
        Parker wind profiles and starts from a converged one. Then, if no
        converged ones are available, uses 'free' instead.
    pdir : str
        Directory as $SUNBATHER_PROJECT_PATH/parker_profiles/planetname/*pdir*/
        where we take the isothermal parker wind density and velocity profiles from.
        Different folders may exist there for a given planet, to separate for
        example profiles with different assumptions such as stellar
        SED/semi-major axis/composition.
    zdict : dict, optional
        Dictionary with the scale factors of all elements relative
        to the default solar composition. Can be easily created with tools.get_zdict().
        Default is None, which results in a solar composition.
    altmax : int, optional
        Maximum altitude of the simulation in units of planet radius, by default 8
    save_sp : list, optional
        A list of atomic/ionic species to let Cloudy save the number density profiles
        for. Those are needed when doing radiative transfer to produce
        transmission spectra. For example, to be able to make
        metastable helium spectra, 'He' needs to be in the save_sp list. By default [].
    constantT : bool, optional
        If True, instead of sovling for a nonisothermal temperature profile,
        the Parker wind profile is ran at the isothermal value. By default False.
    maxit : int, optional
        Maximum number of iterations, by default 16.
    """
    if save_sp is None:
        save_sp = []

    Mdot = f"{float(Mdot):.3f}"  # enforce this format to get standard file names.
    T = str(T)

    # set up the planet object
    planet = tools.Planet(plname)
    if SEDname != "real":
        planet.set_var(SEDname=SEDname)

    # set up the folder structure
    projectpath = tools.get_sunbather_project_path()
    pathTstruc = projectpath + "/sims/1D/" + planet.name + "/" + workingdir + "/"
    path = pathTstruc + "parker_" + T + "_" + Mdot + "/"

    # check if this parker profile exists in the given pdir
    try:
        pprof = tools.read_parker(planet.name, T, Mdot, pdir)
    except FileNotFoundError:
        print(
            "This parker profile does not exist:",
            projectpath
            + "/parker_profiles/"
            + planet.name
            + "/"
            + pdir
            + "/pprof_"
            + planet.name
            + "_T="
            + str(T)
            + "_M="
            + Mdot
            + ".txt",
        )
        return  # quit the run_s function but not the code

    # check for overwriting
    if os.path.isdir(path):  # the simulation exists already
        if not overwrite:
            print(
                "Simulation already exists and overwrite = False:",
                plname, workingdir, Mdot, T
            )
            # this quits the function but if we're running a grid, it doesn't
            # quit the whole Python code
            return
    else:
        os.makedirs(path[:-1])  # make the folder

    # get profiles and parameters we need for the input file
    alt = pprof.alt.values
    hden = tools.rho_to_hden(pprof.rho.values, abundances=tools.get_abundances(zdict))
    dlaw = tools.alt_array_to_Cloudy(alt, hden, altmax, planet.R, 1000, log=True)

    nuFnu_1AU_linear, Ryd = tools.get_SED_norm_1AU(planet.SEDname)
    nuFnu_a_log = np.log10(
        nuFnu_1AU_linear / ((planet.a - altmax * planet.R) / tools.AU) ** 2
    )

    comments = (
        "# plname="
        + planet.name
        + "\n# parker_T="
        + str(T)
        + "\n# parker_Mdot="
        + str(Mdot)
        + "\n# parker_dir="
        + pdir
        + "\n# altmax="
        + str(altmax)
    )

    # this will run the profile at the isothermal T value instead of converging
    # a nonisothermal profile
    if (
        constantT
    ):
        if save_sp == []:
            tools.write_Cloudy_in(
                path + "constantT",
                title=planet.name
                + " 1D Parker with T="
                + str(T)
                + " and log(Mdot)="
                + str(Mdot),
                flux_scaling=[nuFnu_a_log, Ryd],
                SED=planet.SEDname,
                dlaw=dlaw,
                double_tau=True,
                overwrite=overwrite,
                cosmic_rays=True,
                zdict=zdict,
                comments=comments,
                constantT=T,
            )
        else:
            tools.write_Cloudy_in(
                path + "constantT",
                title=planet.name
                + " 1D Parker with T="
                + str(T)
                + " and log(Mdot)="
                + str(Mdot),
                flux_scaling=[nuFnu_a_log, Ryd],
                SED=planet.SEDname,
                dlaw=dlaw,
                double_tau=True,
                overwrite=overwrite,
                cosmic_rays=True,
                zdict=zdict,
                comments=comments,
                constantT=T,
                outfiles=[".den", ".en"],
                denspecies=save_sp,
                selected_den_levels=True,
            )

        tools.run_Cloudy("constantT", folder=path)  # run the Cloudy simulation
        return

    # if we got to here, we are not doing a constantT simulation, so we set up
    # the convergence scheme files
    # write Cloudy template input file - each iteration will add their current
    # temperature structure to this template
    tools.write_Cloudy_in(
        path + "template",
        title=planet.name
        + " 1D Parker with T="
        + str(T)
        + " and log(Mdot)="
        + str(Mdot),
        flux_scaling=[nuFnu_a_log, Ryd],
        SED=planet.SEDname,
        dlaw=dlaw,
        double_tau=True,
        overwrite=overwrite,
        cosmic_rays=True,
        zdict=zdict,
        comments=comments,
    )

    if (
        itno == 0
    ):  # this means we resume from the highest found previously ran iteration
        pattern = (
            r"iteration(\d+)\.out"  # search pattern: iteration followed by an integer
        )
        max_iteration = -1  # set an impossible number
        for filename in os.listdir(path):  # loop through all files/folder in the path
            if os.path.isfile(
                os.path.join(path, filename)
            ):  # if it is a file (not a folder)
                if re.search(pattern, filename):  # if it matches the pattern
                    iteration_number = int(
                        re.search(pattern, filename).group(1)
                    )  # extract the iteration number
                    max_iteration = max(max_iteration, iteration_number)
        if max_iteration == -1:  # this means no files were found
            print(
                f"This folder does not contain any iteration files {path}, so I cannot "
                f"resume from the highest one. Will instead start at itno = 1."
            )
            itno = 1
        else:
            print(
                f"Found the highest iteration {path}iteration{max_iteration}, will "
                f"resume at that same itno."
            )
            itno = max_iteration

    if itno == 1:
        # get starting temperature structure
        clconv = find_close_model(
            pathTstruc, T, Mdot
        )  # find if there are any nearby models we can start from
        if startT == "constant":  # then we start with the isothermal value
            tools.copyadd_Cloudy_in(path + "template", path + "iteration1", constantT=T)

        elif (
            clconv == [None, None] or startT == "free"
        ):  # then we start in free (=radiative eq.) mode
            copyfile(path + "template.in", path + "iteration1.in")

        # then clconv cannot be [None, None] and we start from a previous
        # converged T(r)
        elif (
            startT == "nearby"
        ):
            print(
                f"Model {path} starting from previously converged temperature profile: "
                f"T0 = {clconv[0]}, Mdot = {clconv[1]}"
            )
            prev_conv_T = pd.read_table(
                pathTstruc
                + "parker_"
                + str(clconv[0])
                + "_"
                + f"{clconv[1]:.3f}"
                + "/converged.txt",
                delimiter=" ",
            )
            Cltlaw = tools.alt_array_to_Cloudy(
                prev_conv_T.R * planet.R, prev_conv_T.Te, altmax, planet.R, 1000
            )
            tools.copyadd_Cloudy_in(path + "template", path + "iteration1", tlaw=Cltlaw)

    # with everything in order, run the actual temperature convergence scheme
    solveT.run_loop(path, itno, fc, save_sp, maxit)


def run(
    plname,
    mdot=None,
    temp=None,
    itno=1,
    fc=1.1,
    workingdir=None,
    sedname="real",
    overwrite=False,
    start_temp="nearby",
    pdir=None,
    z=None,
    zelem=None,
    altmax=8,
    save_sp=None,
    constant_temp=False,
    maxit=20,
):
    if zelem is None:
        zelem = {}
    zdict = tools.get_zdict(z=z, zelem=zelem)
    run_s(
        plname,
        mdot,
        temp,
        itno,
        fc,
        workingdir,
        sedname,
        overwrite,
        start_temp,
        pdir,
        zdict=zdict,
        altmax=altmax,
        save_sp=save_sp,
        constantT=constant_temp,
        maxit=maxit,
    )


def catch_errors_run_s(*args):
    """
    Executes the run_s() function with provided arguments, while catching
    errors more gracefully.
    """

    try:
        run_s(*args)
    except Exception as e:
        traceback.print_exc()


def run_g(
    plname,
    cores,
    Mdot_l,
    Mdot_u,
    Mdot_s,
    T_l,
    T_u,
    T_s,
    fc,
    workingdir,
    SEDname,
    overwrite,
    startT,
    pdir,
    zdict,
    altmax,
    save_sp,
    constantT,
    maxit,
):
    """
    Solves for a nonisothermal temperature profile of a grid of isothermal
    Parker wind models, by executing the run_s() function in parallel.

    Parameters
    ----------
    plname : str
        Planet name (must have parameters stored in
        $SUNBATHER_PROJECT_PATH/planets.txt).
    cores : int
        Number of parallel processes to spawn (i.e., number of CPU cores).
    Mdot_l : str or numeric
        Lower bound on the log10(mass-loss rate) grid in units of g s-1.
    Mdot_u : str or numeric
        Upper bound on the log10(mass-loss rate) grid in units of g s-1.
    Mdot_s : str or numeric
        Step size of the log10(mass-loss rate) grid in units of g s-1.
    T_l : str or numeric
        Lower bound on the temperature grid in units of K.
    T_u : str or numeric
        Upper bound on the temperature grid in units of K.
    T_s : str or numeric
        Step size of the temperature grid in units of K.
    fc : numeric
        H/C convergence factor, see Linssen et al. (2024). A sensible value is 1.1.
    workingdir : str
        Directory as $SUNBATHER_PROJECT_PATH/sims/1D/planetname/*workingdir*/
        where the temperature profile will be solved. A folder named
        parker_*T*_*Mdot*/ will be made there.
    SEDname : str
        Name of SED file to use. If SEDname is 'real', we use the name as
        given in the planets.txt file, but if SEDname is something else,
        we advice to use a separate dir folder for this.
    overwrite : bool
        Whether to overwrite if this simulation already exists.
    startT : str
        Either 'constant', 'free' or 'nearby'. Sets the initial
        temperature profile guessed/used for the first iteration.
        'constant' sets it equal to the parker wind isothermal value.
        'free' lets Cloudy solve it, so you will get the radiative equilibrium
        structure.
        'nearby' looks in the workingdir folder for previously solved
        Parker wind profiles and starts from a converged one. Then, if no converged
        ones are available, uses 'free' instead.
    pdir : str
        Directory as $SUNBATHER_PROJECT_PATH/parker_profiles/planetname/*pdir*/
        where we take the isothermal parker wind density and velocity profiles from.
        Different folders may exist there for a given planet, to separate for
        example profiles with different assumptions such as stellar
        SED/semi-major axis/composition.
    zdict : dict, optional
        Dictionary with the scale factors of all elements relative
        to the default solar composition. Can be easily created with tools.get_zdict().
        Default is None, which results in a solar composition.
    altmax : int, optional
        Maximum altitude of the simulation in units of planet radius, by default 8
    save_sp : list, optional
        A list of atomic/ionic species to let Cloudy save the number density profiles
        for. Those are needed when doing radiative transfer to produce
        transmission spectra. For example, to be able to make
        metastable helium spectra, 'He' needs to be in the save_sp list. By default [].
    constantT : bool, optional
        If True, instead of sovling for a nonisothermal temperature profile,
        the Parker wind profile is ran at the isothermal value. By default False.
    maxit : int, optional
        Maximum number of iterations, by default 16.
    """

    with multiprocessing.Pool(processes=cores) as pool:
        pars = []
        for Mdot in np.arange(
            float(Mdot_l), float(Mdot_u) + 1e-6, float(Mdot_s)
        ):  # 1e-6 so that upper bound is inclusive
            for T in np.arange(int(T_l), int(T_u) + 1e-6, int(T_s)).astype(int):
                pars.append(
                    (
                        plname,
                        Mdot,
                        T,
                        1,
                        fc,
                        workingdir,
                        SEDname,
                        overwrite,
                        startT,
                        pdir,
                        zdict,
                        altmax,
                        save_sp,
                        constantT,
                        maxit,
                    )
                )
        pool.starmap(catch_errors_run_s, pars)
        pool.close()
        pool.join()


def new_argument_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Runs the temperature convergence for 1D Parker profile(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    class OneOrThreeAction(argparse.Action):
        """
        Custom class for an argparse argument with exactly 1 or 3 values.
        """

        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) not in (1, 3):
                parser.error("Exactly one or three values are required.")
            setattr(namespace, self.dest, values)

    class AddDictAction(argparse.Action):
        """
        Custom class to add an argparse argument to a dictionary.
        """

        def __call__(self, parser, namespace, values, option_string=None):
            if (
                not hasattr(namespace, self.dest)
                or getattr(namespace, self.dest) is None
            ):
                setattr(namespace, self.dest, {})
            for value in values:
                key, val = value.split("=")
                getattr(namespace, self.dest)[key] = float(val)

    parser.add_argument(
        "-plname", required=True, help="planet name (must be in planets.txt)"
    )
    parser.add_argument(
        "-dir",
        required=True,
        type=str,
        dest="workingdir",
        help=(
            "folder where the temperature structures are solved. e.g. Tstruc_fH_0.9 or "
            "Tstruc_z_100_3xEUV etc."
        ),
    )
    parser.add_argument(
        "-pdir",
        required=True,
        type=str,
        help="parker profile folder/dir to use, e.g. fH_0.9 or z_100.",
    )
    parser.add_argument(
        "-Mdot",
        required=True,
        type=float,
        nargs="+",
        action=OneOrThreeAction,
        help=(
            "log10(mass-loss rate), or three values specifying a grid of "
            "mass-loss rates: lowest, highest, stepsize. -Mdot will be rounded to "
            "three decimal places."
        ),
    )
    parser.add_argument(
        "-T",
        required=True,
        type=int,
        nargs="+",
        action=OneOrThreeAction,
        help=(
            "temperature, or three values specifying a grid of temperatures: lowest, "
            "highest, stepsize."
        ),
    )
    parser.add_argument(
        "-cores", type=int, default=1, help="number of parallel runs"
    )
    parser.add_argument(
        "-fc",
        type=float,
        default=1.1,
        help="convergence factor (heat/cool should be below this value)",
    )
    parser.add_argument(
        "-startT",
        choices=["nearby", "free", "constant"],
        default="nearby",
        help=(
            "initial T structure, either 'constant', 'free' or 'nearby'"
        ),
    )
    parser.add_argument(
        "-itno",
        type=int,
        default=1,
        help=(
            "starting iteration number (itno != 1 only works with -overwrite). As a "
            "special use, you can pass -itno 0 which will automatically find the "
            "highest previously ran iteration number"
        ),
    )
    parser.add_argument(
        "-maxit",
        type=int,
        default=20,
        help="maximum number of iterations",
    )
    parser.add_argument(
        "-SEDname",
        type=str,
        default="real",
        help=(
            "name of SED to use. Must be in Cloudy's data/SED/ folder"
        ),
    )
    parser.add_argument(
        "-overwrite",
        action="store_true",
        help="overwrite existing simulation if passed",
    )
    parser.add_argument(
        "-z",
        type=float,
        default=1.0,
        help=(
            "metallicity (=scale factor relative to solar for all elements except H "
            "and He)"
        ),
    )
    parser.add_argument(
        "-zelem",
        action=AddDictAction,
        nargs="+",
        default={},
        help=(
            "abundance scale factor for specific elements, e.g. -zelem Fe=10 -zelem "
            "He=0.01. Can also be used to toggle elements off, e.g. -zelem Ca=0. "
            "Combines with -z argument. Using this command results in running p_winds "
            "in an iterative scheme where Cloudy updates the mu parameter."
        ),
    )
    parser.add_argument(
        "-altmax",
        type=int,
        default=8,
        help="maximum altitude of the simulation in units of Rp.",
    )
    parser.add_argument(
        "-save_sp",
        type=str,
        nargs="+",
        default=["all"],
        help=(
            "atomic or ionic species to save densities for (needed for radiative "
            "transfer). You can add multiple as e.g. -save_sp He Ca+ Fe3+ Passing "
            "'all' includes all species that weren't turned off. In that case, you can "
            "set the maximum degree of ionization with the -save_sp_max_ion flag. "
        ),
    )
    parser.add_argument(
        "-save_sp_max_ion",
        type=int,
        default=6,
        help=(
            "only used when you set -save_sp all   This command sets the maximum "
            "degree of ionization that will be saved. [default=6] but using lower "
            "values saves significant file size if high ions are not needed. The "
            "maximum number is 12, but such highly ionized species only occur at very "
            "high XUV flux, such as in young systems."
        ),
    )
    parser.add_argument(
        "-constantT",
        action="store_true",
        help=(
            "run the profile at the isothermal temperature instead of converging upon "
            "the temperature structure."
        ),
    )

    return parser


def main(**kwargs):
    """
    Main function for the convergeT_parker.py script
    """

    t0 = time.time()

    parser = new_argument_parser()
    if not kwargs:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = kwargs

    zdict = tools.get_zdict(z=args.z, zelem=args.zelem)

    if "all" in args.save_sp:
        args.save_sp = tools.get_specieslist(
            exclude_elements=[sp for sp, zval in zdict.items() if zval == 0.0],
            max_ion=args.save_sp_max_ion,
        )

    # set up the folder structure if it doesn't exist yet
    projectpath = tools.get_sunbather_project_path()
    if not os.path.isdir(projectpath + "/sims/"):
        os.mkdir(projectpath + "/sims")
    if not os.path.isdir(projectpath + "/sims/1D/"):
        os.mkdir(projectpath + "/sims/1D")
    if not os.path.isdir(projectpath + "/sims/1D/" + args.plname + "/"):
        os.mkdir(projectpath + "/sims/1D/" + args.plname)
    if not os.path.isdir(
        projectpath + "/sims/1D/" + args.plname + "/" + args.dir + "/"
    ):
        os.mkdir(projectpath + "/sims/1D/" + args.plname + "/" + args.dir)

    if len(args.T) == 1 and len(args.Mdot) == 1:  # then we run a single model
        run_s(
            args.plname,
            args.Mdot[0],
            str(args.T[0]),
            args.itno,
            args.fc,
            args.workingdir,
            args.SEDname,
            args.overwrite,
            args.startT,
            args.pdir,
            zdict,
            args.altmax,
            args.save_sp,
            args.constantT,
            args.maxit,
        )
    elif (
        len(args.T) == 3 and len(args.Mdot) == 3
    ):  # then we run a grid over both parameters
        run_g(
            args.plname,
            args.cores,
            args.Mdot[0],
            args.Mdot[1],
            args.Mdot[2],
            args.T[0],
            args.T[1],
            args.T[2],
            args.fc,
            args.workingdir,
            args.SEDname,
            args.overwrite,
            args.startT,
            args.pdir,
            zdict,
            args.altmax,
            args.save_sp,
            args.constantT,
            args.maxit,
        )
    elif len(args.T) == 3 and len(args.Mdot) == 1:  # then we run a grid over only T
        run_g(
            args.plname,
            args.cores,
            args.Mdot[0],
            args.Mdot[0],
            args.Mdot[0],
            args.T[0],
            args.T[1],
            args.T[2],
            args.fc,
            args.workingdir,
            args.SEDname,
            args.overwrite,
            args.startT,
            args.pdir,
            zdict,
            args.altmax,
            args.save_sp,
            args.constantT,
            args.maxit,
        )
    elif len(args.T) == 1 and len(args.Mdot) == 3:  # then we run a grid over only Mdot
        run_g(
            args.plname,
            args.cores,
            args.Mdot[0],
            args.Mdot[1],
            args.Mdot[2],
            args.T[0],
            args.T[0],
            args.T[0],
            args.fc,
            args.workingdir,
            args.SEDname,
            args.overwrite,
            args.startT,
            args.pdir,
            zdict,
            args.altmax,
            args.save_sp,
            args.constantT,
            args.maxit,
        )

    print(
        "\nCalculations took",
        int(time.time() - t0) // 3600,
        "hours, ",
        (int(time.time() - t0) % 3600) // 60,
        "minutes and ",
        (int(time.time() - t0) % 60),
        "seconds.\n",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
