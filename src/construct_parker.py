#sunbather imports
import tools

#other imports
import numpy as np
import os
import time
from shutil import copyfile
import matplotlib.pyplot as plt
import astropy.units as u
from p_winds import tools as pw_tools
from p_winds import parker as pw_parker
from p_winds import hydrogen as pw_hydrogen
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
import argparse
import multiprocessing


def cloudy_spec_to_pwinds(SEDfilename, dist_SED, dist_planet):
    '''
        Reads a spectrum file in the format that we give it to Cloudy (angstroms and nuFnu units)
        and converts it to a spectrum dictionary that p-winds uses.
        This is basically an equivalent of the p_winds.parker.make_spectrum_from_file() function.
    '''

    with open(SEDfilename, 'r') as f:
        for line in f:
            if not line.startswith('#'): #skip through the comments at the top
                assert ('angstrom' in line) or ('Angstrom' in line) #verify the units
                assert 'nuFnu' in line #verify the units
                first_spec_point = np.array(line.split(' ')[:2]).astype(float)
                break
        rest_data = np.genfromtxt(f, skip_header=1)

    SED = np.concatenate(([first_spec_point], rest_data)) #rejoin with the first spectrum point that we read separately

    flux = SED[:,1] / SED[:,0] #from nuFnu = wavFwav to Fwav in erg s-1 cm-2 A-1
    flux = flux * (dist_SED / dist_planet)**2 #scale to planet distance

    assert SED[1,0] > SED[0,0] #check ascending wavelengths

    #make a dictionary like p_winds expects it
    spectrum = {'wavelength':SED[:,0],
                'flux_lambda':flux,
                'wavelength_unit':u.angstrom,
                'flux_unit':u.erg / u.s / u.cm ** 2 / u.angstrom,
                'SEDname':SEDfilename.split('/')[-1][:-5]} #SEDname added by me (without extension)

    return spectrum


def save_plain_parker_profile(planet, Mdot, T, spectrum, h_fraction=0.9, dir='fH=0.9'):
    '''
    Uses the p-winds code (dos Santos et al. 2022)
    Runs p_winds and saves a 'pprof' txt file with the r, rho, v, mu structure.
    Most of this code is taken from the tutorial found via the github:
    https://colab.research.google.com/drive/1mTh6_YEgCRl6DAKqnmRp2XMOW8CTCvm7?usp=sharing

    This function uses just p-winds and the ionization/mu structure calculated by it,
    so it creates H/He atmospheric profiles, contrary to the save_cloudy_parker_profile() function below.

    Sometimes when the solver cannot find a solution, you may want to change
    initial_f_ion to 0.5 or 1.0.

    arguments:
        planet: [Planet]    tools.Planet object
        Mdot: [str/float]   log10 of the mass-loss rate
        T: [str/int]        temperature
        spectrum: [dict]    dictionary with the spectrum, units, name.
                            made with cloudy_spec_to_pwinds() function.
        h_fraction: [float] fraction of hydrogen
        dir: [str]          directory within parker_profiles/planetname/ to
                            store the parker profile in. So e.g. make a
                            directory called fH=0.9 or fH=0.99 and store
                            the corresponding models there.
    '''

    print("Making Parker wind profile with p_winds...")

    if isinstance(Mdot, str):
        Mdot = float(Mdot)
    if isinstance(T, str):
        T = int(T)

    R_pl = planet.R / tools.RJ #because my planet object has it in cm
    M_pl = planet.M #already in MJ

    m_dot = 10 ** Mdot  # Total atmospheric escape rate in g / s
    r = np.logspace(0, np.log10(20), 200)  # Radial distance profile in unit of planetary radii

    # A few assumptions about the planet's atmosphere
    he_fraction = 1 - h_fraction  # He number fraction
    he_h_fraction = he_fraction / h_fraction
    mean_f_ion = 0.0  # Mean ionization fraction (will be self-consistently calculated later)
    mu_0 = (1 + 4 * he_h_fraction) / (1 + he_h_fraction + mean_f_ion)
    # mu_0 is the constant mean molecular weight (assumed for now, will be updated later)


    try:
        initial_f_ion = 0.
        f_r, mu_bar = pw_hydrogen.ion_fraction(r, R_pl, T, h_fraction,
                                    m_dot, M_pl, mu_0,
                                    spectrum_at_planet=spectrum, exact_phi=True,
                                    initial_f_ion=initial_f_ion, relax_solution=True,
                                    return_mu=True, atol=1e-8, rtol=1e-5)

    except RuntimeError as e: #sometimes the solver cannot find a solution
        print("We got this runtime error:", e)
        print("So I will try to construct again, using initial_f_ion = 1.")
        initial_f_ion = 1.0
        f_r, mu_bar = pw_hydrogen.ion_fraction(r, R_pl, T, h_fraction,
                                    m_dot, M_pl, mu_0,
                                    spectrum_at_planet=spectrum, exact_phi=True,
                                    initial_f_ion=initial_f_ion, relax_solution=True,
                                    return_mu=True, atol=1e-8, rtol=1e-5)

    #print("mu_bar:", mu_bar) #temporarily commented out, clutters output when doing grids.

    vs = pw_parker.sound_speed(T, mu_bar)  # Speed of sound (km/s, assumed to be constant)
    rs = pw_parker.radius_sonic_point(M_pl, vs)  # Radius at the sonic point (jupiterRad)
    rhos = pw_parker.density_sonic_point(m_dot, rs, vs)  # Density at the sonic point (g/cm^3)

    r_array = r * R_pl / rs
    v_array, rho_array = pw_parker.structure(r_array)
    mu_array = ((1-h_fraction)*4.0 + h_fraction)/(h_fraction*(1+f_r)+(1-h_fraction)) #this assumes no Helium ionization

    save_array = np.column_stack((r*planet.R, rho_array*rhos, v_array*vs*1e5, mu_array))
    save_name = tools.projectpath+'/parker_profiles/'+planet.name+'/'+dir+'/pprof_'+planet.name+'_T='+str(int(T))+'_M='+ \
                                "%.1f" %Mdot +".txt"
    np.savetxt(save_name, save_array, delimiter='\t', header="alt rho v mu")
    print("Parker wind profile done.")


def save_temp_parker_profile(planet, Mdot, T, spectrum, zdict, dir, mu_bar=None, mu_struc=None):
    '''
    Uses the p-winds code (dos Santos et al. 2022)
    Runs p_winds and saves a 'pprof' txt file with the r, rho, v, mu structure.
    Most of this code is taken from the tutorial found via the github:
    https://colab.research.google.com/drive/1mTh6_YEgCRl6DAKqnmRp2XMOW8CTCvm7?usp=sharing

    The difference with save_plain_parker_profile() is that this function can
    be fed a mu_bar value (e.g. from what Cloudy reports) and construct a
    profile based on that.

    arguments:
        planet: [Planet]    tools.Planet object
        Mdot: [str/float]   log10 of the mass-loss rate
        T: [str/int]        temperature
        spectrum: [dict]    dictionary with the spectrum, units, name.
                            made with cloudy_spec_to_pwinds() function.
        zdict: [dict]       scale factor dictionary with the scale factors
                            for all elements. It only uses the H and He
                            abundances to deduce the H/He ratio.
        dir: [str]          directory within parker_profiles/planetname/ to
                            store the parker profile in. So e.g. make a
                            directory called fH=0.9 or fH=0.99 and store
                            the corresponding models there.
    '''

    if isinstance(Mdot, str):
        Mdot = float(Mdot)
    if isinstance(T, str):
        T = int(T)

    R_pl = planet.R / tools.RJ #radius in RJ (my planet object has it in cm)
    M_pl = planet.M #already in MJ

    m_dot = 10 ** Mdot  # Total atmospheric escape rate in g / s
    r = np.logspace(0, np.log10(20), 200)  # Radial distance profile in unit of planetary radii


    if mu_bar == None: #if not given by a Cloudy run, let p-winds calculate it (used the first iteration)
        #pretend that the metals don't exist and just calculate the h_fraction with only H and He abundances
        abundances = tools.get_abundances(zdict) #solar abundances
        h_fraction = abundances['H'] / (abundances['H'] + abundances['He']) #approximate it by this for now, later Cloudy will give mu

        # A few assumptions about the planet's atmosphere
        he_fraction = 1 - h_fraction  # He number fraction
        he_h_fraction = he_fraction / h_fraction
        mean_f_ion = 0.0  # Mean ionization fraction (will be self-consistently calculated later)
        mu_0 = (1 + 4 * he_h_fraction) / (1 + he_h_fraction + mean_f_ion)
        # mu_0 is the constant mean molecular weight (assumed for now, will be updated later)

        initial_f_ion = 0.

        f_r, mu_bar = pw_hydrogen.ion_fraction(r, R_pl, T, h_fraction,
                                    m_dot, M_pl, mu_0,
                                    spectrum_at_planet=spectrum, exact_phi=True,
                                    initial_f_ion=initial_f_ion, relax_solution=True,
                                    return_mu=True, atol=1e-8, rtol=1e-5,
                                    convergence=0.0001, max_n_relax=30) #I personally think we can use more than 0.01 convergence

        mu_array = ((1-h_fraction)*4.0 + h_fraction)/(h_fraction*(1+f_r)+(1-h_fraction)) #this assumes no Helium ionization

        print("mu_bar as reported by p-winds:", mu_bar)


    else: #used later iterations
        assert np.abs(mu_struc[0,0] - 1.) < 0.01 and np.abs(mu_struc[-1,0] - 20.) < 0.01 #only then extrapolation is safe
        mu_array = interp1d(mu_struc[:,0], mu_struc[:,1], fill_value='extrapolate')(r)

    vs = pw_parker.sound_speed(T, mu_bar)  # Speed of sound (km/s, assumed to be constant)
    rs = pw_parker.radius_sonic_point(M_pl, vs)  # Radius at the sonic point (jupiterRad)
    rhos = pw_parker.density_sonic_point(m_dot, rs, vs)  # Density at the sonic point (g/cm^3)

    r_array = r * R_pl / rs
    v_array, rho_array = pw_parker.structure(r_array)

    save_array = np.column_stack((r*planet.R, rho_array*rhos, v_array*vs*1e5, mu_array))
    save_name = tools.projectpath+'/parker_profiles/'+planet.name+'/'+dir+'/temp/pprof_'+planet.name+'_T='+str(int(T))+'_M='+"%.1f" %Mdot +".txt"
    zdictstr = "abundance scale factors relative to solar:"
    for sp in zdict.keys():
        zdictstr += " "+sp+"="+"%.1f" %zdict[sp]
    np.savetxt(save_name, save_array, delimiter='\t', header=zdictstr+"\nalt rho v mu")

    return save_name, mu_bar


def run_parker_with_cloudy(filename, T, planet, zdict):
    '''
    Runs a parker profile with Cloudy.
    '''
    pprof = tools.read_parker('', '', '', filename=filename)

    altmax = 20
    hdenprof, _, _ = tools.cl_table(pprof.alt.values, pprof.rho.values, pprof.v.values,
                                            altmax, planet.R, 1000, zdict=zdict)

    nuFnu_1AU_linear, Ryd = tools.get_SED_norm_1AU(planet.SEDname)
    nuFnu_a_log = np.log10(nuFnu_1AU_linear / (planet.a - altmax*planet.R/tools.AU)**2)

    simname = filename.split('.txt')[0]
    tools.write_Cloudy_in(simname, title='Simulation of '+filename, overwrite=True,
                                flux_scaling=[nuFnu_a_log, Ryd], SED=planet.SEDname,
                                dlaw=hdenprof, double_tau=True, cosmic_rays=True, zdict=zdict, constantT=T, outfiles=['.ovr'])

    os.system("cd "+filename.rpartition('/')[0]+" && "+tools.cloudyruncommand+" "+simname.split('/')[-1]+" && cd "+tools.projectpath)

    return simname, pprof


def calc_mu_bar(sim, temperature):
    '''
    Adapted from p_winds.parker.average_molecular_weight() to calculate mu_bar of
    a Cloudy simulation Sim object. Based on Eq. A.3 of Lampon et al. 2020.
    '''
    # Converting units
    m_planet = sim.p.M * tools.MJ / 1000. #planet mass in kg
    r = sim.ovr.alt.values[::-1] / 100.  # Radius profile in m
    v_r = sim.ovr.v.values[::-1] / 100.  # Velocity profile in unit of m / s

    # Physical constants
    k_b = 1.380649e-23  # Boltzmann's constant in J / K
    grav = 6.6743e-11  # Gravitational constant in m ** 3 / kg / s ** 2

    # Mean molecular weight in function of radial distance r
    mu_r = sim.ovr.mu.values[::-1]

    # Eq. A.3 of Lampón et al. 2020 is a combination of several integrals, which
    # we calculate here
    int_1 = simps(mu_r / r ** 2, r)
    int_2 = simps(mu_r * v_r, v_r)
    int_3 = trapz(mu_r, 1 / mu_r)
    int_4 = simps(1 / r ** 2, r)
    int_5 = simps(v_r, v_r)
    int_6 = 1 / mu_r[-1] - 1 / mu_r[0]
    term_1 = grav * m_planet * int_1 + int_2 + k_b * temperature * int_3
    term_2 = grav * m_planet * int_4 + int_5 + k_b * temperature * int_6
    mu_bar = term_1 / term_2

    print("mu_bar as reported by Cloudy:", mu_bar)

    return mu_bar


def save_cloudy_parker_profile(planet, Mdot, T, spectrum, zdict, dir, convergence=0.01, maxit=7, cleantemp=False):
    '''
    Calculates a Parker wind profile with any composition by iteratively
    running the p-winds code (dos Santos et al. 2022) and Cloudy (Ferland 1998; 2017).
    p_winds calculates a profile, Cloudy gives the mean molecular weight structure,
    we calculate a weighted mu_bar value based on that and feed that to p_winds
    to generate a new profile until we converge. Saves a 'pprof' txt file with the r, rho, v, mu structure.

    arguments:
        planet: [Planet]    tools.Planet object
        Mdot: [str/float]   log10 of the mass-loss rate
        T: [int/float]      temperature
        spectrum: [dict]    dictionary with the spectrum, units, name.
                            made with cloudy_spec_to_pwinds() function.
        zdict: [dict]       scale factor dictionary with the scale factors
                            for all elements relative to solar abundance.
        dir: [str]          directory within parker_profiles/planetname/ to
                            store the parker profile in. So e.g. make a
                            directory called z=10 or Fe=100 and store
                            the corresponding models there.
        convergence:[float] fractional difference in mu_bar between successive
                            iterations needed for convergence.
        maxit: [int]        maximum number of iterations (even if convergence
                            threshold is not reached)
        cleantemp: [bool]   whether to remove the temporary folder /temp/ after
                            the calculations. The temp folder stores Cloudy
                            files and intermediate parker profiles.
    '''


    print("Making initial parker profile with p-winds...")
    filename, previous_mu_bar = save_temp_parker_profile(planet, Mdot, T, spectrum, zdict, dir, mu_bar=None)
    print("Saved temp parker profile.")

    for itno in range(maxit):
        print("Iteration number:", itno+1)

        print("Running parker profile through Cloudy...")
        simname, pprof = run_parker_with_cloudy(filename, T, planet, zdict)
        print("Cloudy run done.")

        sim = tools.Sim(simname, altmax=20, planet=planet)
        sim.addv(pprof.alt, pprof.v) #add the velocity structure to the sim, so that calc_mu_bar() works.

        mu_bar = calc_mu_bar(sim, T)
        print("Making new parker profile with p-winds based on Cloudy's reported mu_bar...")
        mu_struc = np.column_stack((sim.ovr.alt.values[::-1]/planet.R, sim.ovr.mu[::-1].values)) #pass Cloudy's mu structure to save in the pprof
        filename, mu_bar = save_temp_parker_profile(planet, Mdot, T, spectrum, zdict, dir, mu_bar=mu_bar, mu_struc=mu_struc)
        print("Saved temp parker profile.")

        if np.abs(mu_bar - previous_mu_bar)/previous_mu_bar < convergence:
            print("mu_bar converged.")
            break
        else:
            previous_mu_bar = mu_bar

    copyfile(filename, filename.split('temp/')[0] + filename.split('temp/')[1])
    print("Copied final parker profile from temp to parent folder.")

    if cleantemp: #then we remove the temp files
        print("Will remove temporarary files.")
        os.remove(simname+'.in')
        os.remove(simname+'.out')
        os.remove(simname+'.ovr')
        os.remove(filename)
        print("Temporary files removed.")


def run_s(plname, pdir, Mdot, T, SEDname, fH, zdict, mu_conv, mu_maxit):
    p = tools.Planet(plname)
    if SEDname != 'real':
        planet.set_var(SEDname=SEDname)        
    spectrum = cloudy_spec_to_pwinds(tools.cloudypath+'/data/SED/'+p.SEDname, 1., p.a - 20*p.R / tools.AU) #assumes SED is at 1 AU

    if fH != None: #then run p_winds standalone
        save_plain_parker_profile(p, Mdot, T, spectrum, h_fraction=fH, dir=pdir)
    else: #then run p_winds/Cloudy iterative scheme
        save_cloudy_parker_profile(p, Mdot, T, spectrum, zdict, pdir, convergence=mu_conv, maxit=mu_maxit, cleantemp=True)


def run_g(plname, pdir, cores, Mdot_l, Mdot_u, Mdot_s, T_l, T_u, T_s, SEDname, fH, zdict, mu_conv, mu_maxit):
    '''
    Runs the function run_s in parallel for a given grid of Mdots and T, and
    for given number of cores (=parallel processes).
    '''

    p = multiprocessing.Pool(cores)

    pars = []
    for Mdot in np.arange(float(Mdot_l), float(Mdot_u)+float(Mdot_s), float(Mdot_s)):
        for T in np.arange(int(T_l), int(T_u)+int(T_s), int(T_s)).astype(int):
            pars.append((plname, pdir, "%.1f" % Mdot, T, SEDname, fH, zdict, mu_conv, mu_maxit))

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
        '''
        Custom class for an argparse argument that adds to a dictionary.
        '''
        def __call__(self, parser, namespace, values, option_string=None):
            if not hasattr(namespace, self.dest) or getattr(namespace, self.dest) is None:
                setattr(namespace, self.dest, {})
            for value in values:
                key, val = value.split('=')
                getattr(namespace, self.dest)[key] = float(val)


    t0 = time.time()

    parser = argparse.ArgumentParser(description="Creates 1D Parker profile(s) using the p_winds code and Cloudy.")

    parser.add_argument("-plname", required=True, help="planet name (must be in planets.txt)")
    parser.add_argument("-pdir", required=True, help="directory where the profiles are saved. It is adviced to choose a name that " \
                                        "somehow represents the chosen parameters, e.g. 'fH_0.9' or 'z=10'. The path will be tools.projectpath/parker_profiles/pdir/")
    parser.add_argument("-Mdot", required=True, nargs='+', action=OneOrThreeAction, help="log10(mass-loss rate), or three values specifying a grid of mass-loss rates: lowest, highest, stepsize.")
    parser.add_argument("-T", required=True, type=int, nargs='+', action=OneOrThreeAction, help="temperature, or three values specifying a grid of temperatures: lowest, highest, stepsize.")
    parser.add_argument("-SEDname", type=str, default='real', help="name of SED to use. Must be in Cloudy's data/SED/ folder [default=SEDname set in planet.txt file]")
    composition_group = parser.add_mutually_exclusive_group(required=True)
    composition_group.add_argument("-fH", type=float, help="hydrogen fraction by number. Using this command results in running standalone p_winds without invoking Cloudy.")
    composition_group.add_argument("-z", type=float, help="metallicity (=scale factor relative to solar for all elements except H and He). Using this " \
                                        "command results in running p_winds in an an iterative scheme where Cloudy updates the mu parameter.")
    parser.add_argument("-zelem", action = AddDictAction, nargs='+', default = {}, help="abundance scale factor for specific elements, e.g. -zelem Fe=10 -zelem He=0.01. " \
                                        "Can also be used to toggle elements off, e.g. -zelem Ca=0. Combines with -z argument. Using this " \
                                        "command results in running p_winds in an an iterative scheme where Cloudy updates the mu parameter.")
    parser.add_argument("-cores", type=int, default=1, help="number of parallel runs [default=1]")
    parser.add_argument("-mu_conv", type=float, default=0.01, help="relative change in mu allowed for convergence, when using p_winds/Cloudy iterative scheme [default=0.01]")
    parser.add_argument("-mu_maxit", type=int, default=7, help="maximum number of iterations the p_winds/Cloudy iterative scheme is ran " \
                                        "if convergence is not reached [default =7]")
    args = parser.parse_args()

    if args.z != None:
        zdict = tools.get_zdict(z=args.z, zelem=args.zelem)
    else: #if z==None we should not pass that to the tools.get_zdict function
        zdict = tools.get_zdict(zelem=args.zelem)

    if args.fH != None and (args.zelem != {} or args.mu_conv != 0.01 or args.mu_maxit != 7):
        print("The -zelem, -mu_conv and -mu_maxit commands only combine with -z, not with -fH, so I will ignore their input.")

    #set up the folder structure if it doesn't exist yet
    if not os.path.isdir(tools.projectpath+'/parker_profiles/'):
        os.mkdir(tools.projectpath+'/parker_profiles')
    if not os.path.isdir(tools.projectpath+'/parker_profiles/'+args.plname+'/'):
        os.mkdir(tools.projectpath+'/parker_profiles/'+args.plname)
    if not os.path.isdir(tools.projectpath+'/parker_profiles/'+args.plname+'/'+args.pdir+'/'):
        os.mkdir(tools.projectpath+'/parker_profiles/'+args.plname+'/'+args.pdir+'/')
    if (args.fH == None) and (not os.path.isdir(tools.projectpath+'/parker_profiles/'+args.plname+'/'+args.pdir+'/temp/')):
        os.mkdir(tools.projectpath+'/parker_profiles/'+args.plname+'/'+args.pdir+'/temp')

    if (len(args.T) == 1 and len(args.Mdot) == 1): #then we run a single model
        run_s(args.plname, args.pdir, args.Mdot[0], args.T[0], args.SEDname, args.fH, zdict, args.mu_conv, args.mu_maxit)
    elif (len(args.T) == 3 and len(args.Mdot) == 3): #then we run a grid over both parameters
        run_g(args.plname, args.pdir, args.cores, args.Mdot[0], args.Mdot[1], args.Mdot[2], args.T[0], args.T[1], args.T[2], args.SEDname, args.fH, zdict, args.mu_conv, args.mu_maxit)
    elif (len(args.T) == 3 and len(args.Mdot) == 1): #then we run a grid over only T
        run_g(args.plname, args.pdir, args.cores, args.Mdot[0], args.Mdot[0], args.Mdot[0], args.T[0], args.T[1], args.T[2], args.SEDname, args.fH, zdict, args.mu_conv, args.mu_maxit)
    elif (len(args.T) == 1 and len(args.Mdot) == 3): #then we run a grid over only Mdot
        run_g(args.plname, args.pdir, args.cores, args.Mdot[0], args.Mdot[1], args.Mdot[2], args.T[0], args.T[0], args.T[0], args.SEDname, args.fH, zdict, args.mu_conv, args.mu_maxit)

    print("\nCalculations took", int(time.time()-t0) // 3600, "hours, ", (int(time.time()-t0)%3600) // 60, "minutes and ", (int(time.time()-t0)%60), "seconds.\n")
