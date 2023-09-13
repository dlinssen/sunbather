#sunbather imports
import tools
import solveT

#other imports
import pandas as pd
import numpy as np
import multiprocessing
from shutil import copyfile
import time
import glob
import os
import argparse


def find_close_model(parentfolder, T, Mdot, tolT=2000, tolMdot=1.0):
    '''
    This function takes a parent folder where multiple 1D parker profiles have been ran,
    and for given T and Mdot it looks for another model that is already finished and closest
    to the given model, so that we can start our new simulation from that converged temperature
    structure. It returns the T and Mdot
    of the close converged folder, or None if there aren't any (within the tolerance).
    '''

    allfolders = glob.glob(parentfolder+'parker_*/')
    convergedfolders = [] #stores the T and Mdot values of all folders with 0.out files
    for folder in allfolders:
        if os.path.isfile(folder+'converged.out'):
            folderparams = folder.split('/')[-2].split('_')
            convergedfolders.append([int(folderparams[1]), float(folderparams[2])])

    if [int(T), float(Mdot)] in convergedfolders: #if the current folder is found, remove it
        convergedfolders.remove([int(T), float(Mdot)])

    if convergedfolders == []: #then we default to constant starting value
        clconv = [None, None]
    else: #find closest converged profile
        dist = lambda x, y: (x[0]-y[0])**2 + (2000*(x[1]-y[1]))**2 #1 order of magnitude Mdot is now 'equal weighted' to 2000K
        clconv = min(convergedfolders, key=lambda fol: dist(fol, [int(T), float(Mdot)])) #closest converged [T, Mdot]
        if (np.abs(clconv[0] - int(T)) > tolT) or (np.abs(clconv[1] - float(Mdot)) > tolMdot):
            clconv = [None, None]

    return clconv


def run_s(plname, Mdot, T, itno, fc, dir, SEDname, overwrite, startT, pdir, zdict=None, altmax=8, save_sp=[], constantT=False):
    '''
    Solves for the converged temperature structure of a single parker wind profile.

    Arguments:
        plname: [str]       planet name that occurs in planets.txt
        Mdot: [str/float]   parker wind log10 of the mass loss rate
        T: [str/int]        parker wind isothermal temperature
        itno: [int]         iteration number to start (can only be different from 1
                            if this same model has been ran before, and then also
                            overwrite = True needs to be set.)
        fc: [float]         convergence factor, default should be 1.1. Sets the difference
                            in temperature structures between successive iterations
        dir: [str]          direction as projectpath/sims/1D/planetname/dir/
                            where the profile will be solved. A folder as
                            parker_T_Mdot/ will be made there and Cloudy is ran within.
        SEDname: [str]      name of SED file to use. if SEDname='real', we use the name as
                            given in the planets.txt file, but if SEDname is something else,
                            we advice to use a separate dir folder for this.
        overwrite: [bool]   whether to overwrite if this simulation (i.e. folder)
                            already exists.
        startT: [str]       either 'constant', 'free' or 'nearby'. Sets the initial
                            temperature structure used for the first iteration.
                            'constant' sets it equal to the parker wind isothermal value.
                            'free' lets Cloudy solve it and thus you get the rad. eq. structure.
                            'nearby' looks in the dir folder for previously solved
                            parker wind profiles and starts from a converged one.
                            If no converged ones are available, uses 'free' instead.
        pdir: [str]         direction as projectpath/parker_profiles/planetname/pdir/
                            where we take the parker wind profiles from. Different folders
                            may exist there for a given planet for parker wind profiles
                            with different assumptions such as SED/a/z/fH, etc.
        zdict: [dict]       dictionary with the scale factors of all elements relative
                            to the default solar composition.
        save_sp: [list]     a list of atomic/ionic species to save the density structures
                            for. This is needed when doing radiative transfer to produce
                            transmission spectra later. For example, to be able to make
                            metastable helium spectra, 'He' needs to be in the save_sp list.
        constantT: [bool]   if True, instead of converging the temperature structure,
                            the Parker wind profile is ran at the isothermal value.
    '''

    Mdot = "%.3f" % float(Mdot) #enforce this format to get standard file names.
    T = str(T)

    #set up the planet object
    planet = tools.Planet(plname)
    if SEDname != 'real':
        planet.set_var(SEDname=SEDname)

    #set up the folder structure
    pathTstruc = tools.projectpath+'/sims/1D/'+planet.name+'/'+dir+'/'
    path = pathTstruc+'parker_'+T+'_'+Mdot+'/'

    #check if this parker profile exists in the given pdir
    try:
        pprof = tools.read_parker(planet.name, T, Mdot, dir=pdir)
    except FileNotFoundError:
        print("This parker profile does not exist:", tools.projectpath+'/parker_profiles/'+planet.name+'/'+pdir+'/pprof_'+planet.name+'_T='+str(T)+'_M='+Mdot+'.txt')
        return #quit the run_s function but not the code

    #check for overwriting
    if os.path.isdir(path): #the simulation exists already
        if not overwrite:
            print("Simulation already exists and overwrite = False:", plname, dir, Mdot, T)
            return #this quits the function but if we're running a grid, it doesn't quit the whole Python code
    else:
        os.mkdir(path[:-1]) #make the folder

    #get profiles and parameters we need for the input file
    hdenprof, cextraprof, advecprof = tools.cl_table(pprof.alt.values, pprof.rho.values, pprof.v.values,
                                            altmax, planet.R, 1000, zdict=zdict)

    nuFnu_1AU_linear, Ryd = tools.get_SED_norm_1AU(planet.SEDname)
    nuFnu_a_log = np.log10(nuFnu_1AU_linear / (planet.a - altmax*planet.R/tools.AU)**2)

    comments = '# plname='+planet.name+'\n# parker_T='+str(T)+'\n# parker_Mdot='+str(Mdot)+'\n# parker_dir='+pdir+'\n# altmax='+str(altmax)

    if constantT: #this will run the profile at the isothermal T value instead of converging a nonisothermal profile
        tools.write_Cloudy_in(path+'constantT', title=planet.name+' 1D Parker with T='+str(T)+' and log(Mdot)='+str(Mdot),
                                    flux_scaling=[nuFnu_a_log, Ryd], SED=planet.SEDname, dlaw=hdenprof, double_tau=True,
                                    overwrite=overwrite, cosmic_rays=True, zdict=zdict, comments=comments, constantT=T)
        os.system("cd "+path+" && "+tools.cloudyruncommand+" constantT")
        return

    #else we converge T:
    #write Cloudy template input file - each iteration will add their current temperature structure to this template
    tools.write_Cloudy_in(path+'template', title=planet.name+' 1D Parker with T='+str(T)+' and log(Mdot)='+str(Mdot),
                                flux_scaling=[nuFnu_a_log, Ryd], SED=planet.SEDname, dlaw=hdenprof, double_tau=True,
                                overwrite=overwrite, cosmic_rays=True, zdict=zdict, comments=comments)

    #write clocs file that keeps track of from where we construct - see solveT.py
    if itno == 1:
        with open(path+"clocs.txt", "w") as f:
            f.write("1 0")

        #get starting temperature structure
        clconv = find_close_model(pathTstruc, T, Mdot) #find if there are any nearby models we can start from
        if startT == 'constant': #then we start with the isothermal value
            tools.copyadd_Cloudy_in(path+'template', path+'iteration1', constantT=T)

        elif clconv == [None, None] or startT == 'free': #then we start in free (=radiative eq.) mode
            copyfile(path+'template.in', path+'iteration1.in')

        elif startT == 'nearby': #then clconv cannot be [None, None] and we start from a previous converged T(r)
            print("Model", T, Mdot, "starting from previously converged profile:", clconv)
            prev_conv_T = pd.read_table(pathTstruc+'parker_'+str(clconv[0])+'_'+"{:.3f}".format(clconv[1])+'/converged.txt', delimiter=' ')
            Cltlaw = tools.alt_array_to_Cloudy(prev_conv_T.R * planet.R, prev_conv_T.Te, altmax, planet.R, 1000)
            tools.copyadd_Cloudy_in(path+'template', path+'iteration1', tlaw=Cltlaw)


    #with everything in order, run the actual temperature convergence scheme
    solveT.run_loop(path, itno, fc, altmax, planet.R, cextraprof, advecprof, zdict, save_sp)


def run_g(plname, cores, Mdot_l, Mdot_u, Mdot_s, T_l, T_u, T_s, fc, dir, SEDname, overwrite, startT, pdir, zdict, altmax, save_sp, constantT):
    '''
    Runs the function run_s in parallel for a given grid of Mdots and T, and
    for given number of cores (=parallel processes).
    '''

    p = multiprocessing.Pool(cores)

    pars = []
    for Mdot in np.arange(float(Mdot_l), float(Mdot_u)+float(Mdot_s), float(Mdot_s)):
        for T in np.arange(int(T_l), int(T_u)+int(T_s), int(T_s)).astype(int):
            pars.append((plname, Mdot, T, 1, fc, dir, SEDname, overwrite, startT, pdir, zdict, altmax, save_sp, constantT))

    p.starmap(run_s, pars)
    p.close()
    p.join()




if __name__ == '__main__':

    class OneOrThreeAction(argparse.Action):
        '''
        Custom class for an argparse argument with exactly 1 or 3 values.
        '''
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) not in (1, 3):
                parser.error("Exactly one or three values are required.")
            setattr(namespace, self.dest, values)

    class AddDictAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if not hasattr(namespace, self.dest) or getattr(namespace, self.dest) is None:
                setattr(namespace, self.dest, {})
            for value in values:
                key, val = value.split('=')
                getattr(namespace, self.dest)[key] = float(val)


    t0 = time.time()

    parser = argparse.ArgumentParser(description="Runs the temperature convergence for 1D Parker profile(s).")

    parser.add_argument("-plname", required=True, help="planet name (must be in planets.txt)")
    parser.add_argument("-dir", required=True, type=str, help="folder where the temperature structures are solved. e.g. Tstruc_fH_0.9 or Tstruc_z_100_3xEUV etc.")
    parser.add_argument("-pdir", required=True, type=str, help="parker profile folder/dir to use, e.g. fH_0.9 or z_100.")
    parser.add_argument("-Mdot", required=True, type=float, nargs='+', action=OneOrThreeAction, help="log10(mass-loss rate), or three values specifying a grid of " \
                                    "mass-loss rates: lowest, highest, stepsize. -Mdot will be rounded to three decimal places.")
    parser.add_argument("-T", required=True, type=int, nargs='+', action=OneOrThreeAction, help="temperature, or three values specifying a grid of temperatures: lowest, highest, stepsize.")
    parser.add_argument("-cores", type=int, default=1, help="number of parallel runs [default=1]")
    parser.add_argument("-fc", type=float, default=1.1, help="convergence factor (heat/cool should be below this value) [default=1.1]")
    parser.add_argument("-startT", choices=["nearby", "free", "constant"], default="nearby", help="initial T structure, either 'constant', 'free' or 'nearby' [default=nearby]")
    parser.add_argument("-itno", type=int, default=1, help="starting iteration number (itno != 1 only works with -overwrite). As a special use, you can pass " \
                                    "-itno 0 which will automatically find the highest previously ran iteration number [default=1]")
    parser.add_argument("-SEDname", type=str, default='real', help="name of SED to use. Must be in Cloudy's data/SED/ folder [default=SEDname set in planet.txt file]")
    parser.add_argument("-overwrite", action='store_true', help="overwrite existing simulation if passed [default=False]")
    parser.add_argument("-z", type=float, default=1., help="metallicity (=scale factor relative to solar for all elements except H and He) [default=1.]")
    parser.add_argument("-zelem", action = AddDictAction, nargs='+', default = {}, help="abundance scale factor for specific elements, e.g. -zelem Fe=10 -zelem He=0.01. " \
                                        "Can also be used to toggle elements off, e.g. -zelem Ca=0. Combines with -z argument. Using this " \
                                        "command results in running p_winds in an an iterative scheme where Cloudy updates the mu parameter.")
    parser.add_argument("-altmax", type=int, default=8, help="maximum altitude of the simulation in units of Rp. [default=8]")
    parser.add_argument("-save_sp", type=str, nargs='+', default=[], help="atomic or ionic species to save densities for (needed for radiative transfer). " \
                                    "You can add multiple as e.g. -save_sp He Ca+ Fe3+ Passing 'all' includes all species that weren't turned off. In that case, you can "\
                                    "set the maximum degree of ionization with the -save_sp_max_ion flag. default=[] i.e. none.")
    parser.add_argument("-save_sp_max_ion", type=int, default=6, help="only used when you set -save_sp all   This command sets the maximum degree of ionization "\
                                    "that will be saved. [default=6] but using lower values saves significant file size if high ions are not needed.")
    parser.add_argument("-constantT", action='store_true', help="run the profile at the isothermal temperature instead of converging upon the temperature structure. [default=False]")


    args = parser.parse_args()

    zdict = tools.get_zdict(z=args.z, zelem=args.zelem)

    if 'all' in args.save_sp:
        args.save_sp = tools.get_specieslist(exclude_elements=[sp for sp,zval in zdict.items() if zval == 0.], max_ion=args.save_sp_max_ion)

    #set up the folder structure if it doesn't exist yet
    if not os.path.isdir(tools.projectpath+'/sims/'):
        os.mkdir(tools.projectpath+'/sims')
    if not os.path.isdir(tools.projectpath+'/sims/1D/'):
        os.mkdir(tools.projectpath+'/sims/1D')
    if not os.path.isdir(tools.projectpath+'/sims/1D/'+args.plname+'/'):
        os.mkdir(tools.projectpath+'/sims/1D/'+args.plname)
    if not os.path.isdir(tools.projectpath+'/sims/1D/'+args.plname+'/'+args.dir+'/'):
        os.mkdir(tools.projectpath+'/sims/1D/'+args.plname+'/'+args.dir)

    if (len(args.T) == 1 and len(args.Mdot) == 1): #then we run a single model
        run_s(args.plname, args.Mdot[0], str(args.T[0]), args.itno, args.fc, args.dir, args.SEDname, args.overwrite, args.startT, args.pdir, zdict, args.altmax, args.save_sp, args.constantT)
    elif (len(args.T) == 3 and len(args.Mdot) == 3): #then we run a grid over both parameters
        run_g(args.plname, args.cores, args.Mdot[0], args.Mdot[1], args.Mdot[2], args.T[0], args.T[1], args.T[2], args.fc, args.dir, args.SEDname, args.overwrite, args.startT, args.pdir, zdict, args.altmax, args.save_sp, args.constantT)
    elif (len(args.T) == 3 and len(args.Mdot) == 1): #then we run a grid over only T
        run_g(args.plname, args.cores, args.Mdot[0], args.Mdot[0], args.Mdot[0], args.T[0], args.T[1], args.T[2], args.fc, args.dir, args.SEDname, args.overwrite, args.startT, args.pdir, zdict, args.altmax, args.save_sp, args.constantT)
    elif (len(args.T) == 1 and len(args.Mdot) == 3): #then we run a grid over only Mdot
        run_g(args.plname, args.cores, args.Mdot[0], args.Mdot[1], args.Mdot[2], args.T[0], args.T[0], args.T[0], args.fc, args.dir, args.SEDname, args.overwrite, args.startT, args.pdir, zdict, args.altmax, args.save_sp, args.constantT)

    print("\nCalculations took", int(time.time()-t0) // 3600, "hours, ", (int(time.time()-t0)%3600) // 60, "minutes and ", (int(time.time()-t0)%60), "seconds.\n")
