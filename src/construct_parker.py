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
from scipy.integrate import simpson, trapz
from scipy.interpolate import interp1d
import argparse
import multiprocessing
import traceback
import warnings


def cloudy_spec_to_pwinds(SEDfilename, dist_SED, dist_planet):
    """
    Reads a spectrum file in the format that we give it to Cloudy, namely
    angstroms and monochromatic flux (i.e., nu*F_nu or lambda*F_lambda) units.
    and converts it to a spectrum dictionary that p-winds uses.
    This is basically an equivalent of the p_winds.parker.make_spectrum_from_file() function.

    Parameters
    ----------
    SEDfilename : str
        Full path + filename of the SED file. SED file must be in the sunbather/Cloudy
        standard units, namely wavelengths in Å and lambda*F_lambda flux units.
    dist_SED : numeric
        Distance from the source at which the SED is defined (typically 1 AU).
        Must have the same units as dist_planet.
    dist_planet : numeric
        Distance from the source to which the SED must be scaled
        (typically semi-major axis - total atmospheric height). Must have the
        same units as dist_SED.

    Returns
    -------
    spectrum : dict
        SED at the planet distance in the dictionary format that p-winds expects.
    """

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


def calc_neutral_mu(zdict):
    """Calculates the mean particle mass assuming a completely neutral (i.e., atomic)
    gas, for a given composition (specified through elemental scale factors that
    can be converted into abundances).

    Parameters
    ----------
    zdict : dict
        Dictionary with the scale factors of all elements relative
        to the default solar composition. Can be easily created with tools.get_zdict().

    Returns
    -------
    neutral_mu : float
        Mean particle mass in units of amu.
    """

    abundances = tools.get_abundances(zdict)
    neutral_mu = tools.calc_mu(1., 0., abundances=abundances) #set ne=0 so completely neutral

    return neutral_mu


def save_plain_parker_profile(planet, Mdot, T, spectrum, h_fraction=0.9, 
                              pdir='fH_0.9', overwrite=False, no_tidal=False, altmax=20):
    """
    Uses the p-winds code (dos Santos et al. 2022).
    Runs p-winds and saves a 'pprof' txt file with the r, rho, v, mu structure.
    This function uses p-winds standalone and can thus only calculate H/He atmospheres.
    Most of this code is taken from the p-winds tutorial found via the github:
    https://colab.research.google.com/drive/1mTh6_YEgCRl6DAKqnmRp2XMOW8CTCvm7?usp=sharing

    Sometimes when the solver cannot find a solution, you may want to change
    initial_f_ion to 0.5 or 1.0.

    Parameters
    ----------
    planet : tools.Planet
        Planet parameters.
    Mdot : str or numeric
        log of the mass-loss rate in units of g s-1.
    T : str or numeric
        Temperature in units of K.
    spectrum : dict
        SED at the planet distance in the dictionary format that p-winds expects.
        Can be made with cloudy_spec_to_pwinds().
    h_fraction : float, optional
        Hydrogen abundance expressed as a fraction of the total, by default 0.9
    pdir : str, optional
        Directory as $SUNBATHER_PROJECT_PATH/parker_profiles/planetname/*pdir*/
        where the isothermal parker wind density and velocity profiles are saved.
        Different folders may exist there for a given planet, to separate for example profiles
        with different assumptions such as stellar SED/semi-major axis/composition. By default 'fH_0.9'.
    overwrite : bool, optional
        Whether to overwrite existing models, by default False.
    notidal : bool, optional
        Whether to neglect tidal gravity - fourth term of Eq. 4 of Linssen et al. (2024).
        See also Appendix D of Vissapragada et al. (2022) for the p-winds implementation.
        Default is False, i.e. tidal gravity incluced.
    altmax : int, optional
        Maximum altitude of the profile in units of the planet radius. By default 20.
    """

    Mdot = float(Mdot)
    T = int(T)

    save_name = tools.projectpath+'/parker_profiles/'+planet.name+'/'+pdir+'/pprof_'+planet.name+'_T='+str(T)+'_M='+ \
                            "%.3f" %Mdot +".txt"
    if os.path.exists(save_name) and not overwrite:
        print("Parker profile already exists and overwrite = False:", planet.name, pdir, "%.3f" %Mdot, T)
        return #this quits the function but if we're running a grid, it doesn't quit the whole Python code

    R_pl = planet.R / tools.RJ #convert from cm to Rjup
    M_pl = planet.M / tools.MJ #convert from g to Mjup

    m_dot = 10 ** Mdot  # Total atmospheric escape rate in g / s
    r = np.logspace(0, np.log10(altmax), 1000)  # Radial distance profile in unit of planetary radii

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
                                return_mu=True, atol=1e-8, rtol=1e-5)

    vs = pw_parker.sound_speed(T, mu_bar)  # Speed of sound (km/s, assumed to be constant)
    if no_tidal:
        rs = pw_parker.radius_sonic_point(M_pl, vs)  # Radius at the sonic point (jupiterRad)
        rhos = pw_parker.density_sonic_point(m_dot, rs, vs)  # Density at the sonic point (g/cm^3)
        r_array = r * R_pl / rs
        v_array, rho_array = pw_parker.structure(r_array)
    else:
        Mstar = planet.Mstar / tools.Msun #convert from g to Msun
        a = planet.a / tools.AU #convert from cm to AU
        rs = pw_parker.radius_sonic_point_tidal(M_pl, vs, Mstar, a) #radius at the sonic point (jupiterRad)
        rhos = pw_parker.density_sonic_point(m_dot, rs, vs)  # Density at the sonic point (g/cm^3)
        r_array = r * R_pl / rs
        v_array, rho_array = pw_parker.structure_tidal(r_array, vs, rs, M_pl, Mstar, a)
    mu_array = ((1-h_fraction)*4.0 + h_fraction)/(h_fraction*(1+f_r)+(1-h_fraction)) #this assumes no Helium ionization

    save_array = np.column_stack((r*planet.R, rho_array*rhos, v_array*vs*1e5, mu_array))
    np.savetxt(save_name, save_array, delimiter='\t', header=f"hydrogen fraction: {h_fraction:.3f}\nalt rho v mu")
    print("Parker wind profile done:", save_name)

    launch_velocity = v_array[0] #velocity at Rp in units of sonic speed

    if launch_velocity > 1:
        warnings.warn(f"This Parker wind profile is supersonic already at Rp: {save_name}")


def save_temp_parker_profile(planet, Mdot, T, zdict, pdir, 
                             mu_bar, mu_struc=None, no_tidal=False, altmax=20):
    """
    Uses the p-winds code (dos Santos et al. 2022)
    Runs p_winds and saves a 'pprof' txt file with the r, rho, v, mu structure.
    The difference with save_plain_parker_profile() is that this function can
    be given a mu_bar value (e.g. from what Cloudy reports) and calculate a
    Parker wind profile based on that.
    Most of this code is taken from the tutorial found via the github:
    https://colab.research.google.com/drive/1mTh6_YEgCRl6DAKqnmRp2XMOW8CTCvm7?usp=sharing

    Parameters
    ----------
    planet : tools.Planet
        Object storing the planet parameters.
    Mdot : str or numeric
        log of the mass-loss rate in units of g s-1.
    T : str or numeric
        Temperature in units of K.
    zdict : dict
        Dictionary with the scale factors of all elements relative
        to the default solar composition. Can be easily created with tools.get_zdict().
    pdir : str
        Directory as $SUNBATHER_PROJECT_PATH/parker_profiles/planetname/*pdir*/
        where the isothermal parker wind density and velocity profiles are saved.
        Different folders may exist there for a given planet, to separate for example profiles
        with different assumptions such as stellar SED/semi-major axis/composition.
    mu_bar : float
        Weighted mean of the mean particle mass. Based on Eq. A.3 of Lampon et al. (2020).
    mu_struc : numpy.ndarray, optional
        Mean particle mass profile, must be provided if mu_bar is None.
        Typically, this is a mu(r)-profile as given by Cloudy. By default None.
    no_tidal : bool, optional
        Whether to neglect tidal gravity - fourth term of Eq. 4 of Linssen et al. (2024).
        See also Appendix D of Vissapragada et al. (2022) for the p-winds implementation.
        Default is False, i.e. tidal gravity included.
    altmax : int, optional
        Maximum altitude of the profile in units of the planet radius. By default 20.

    Returns
    -------
    save_name : str
        Full path + filename of the saved Parker wind profile file.
    launch_velocity : float
        Velocity at the planet radius in units of the sonic speed. If it is larger than 1,
        the wind is "launched" already supersonic, and hence the assumption of a transonic
        wind is not valid anymore.
    """

    Mdot = float(Mdot)
    T = int(T)

    R_pl = planet.R / tools.RJ #convert from cm to Rjup
    M_pl = planet.M / tools.MJ #convert from g to Mjup

    m_dot = 10 ** Mdot  # Total atmospheric escape rate in g / s
    r = np.logspace(0, np.log10(altmax), 1000)  # Radial distance profile in unit of planetary radii

    assert np.abs(mu_struc[0,0] - 1.) < 0.03 and np.abs(mu_struc[-1,0] - altmax) < 0.0001, "Looks like Cloudy didn't simulate to 1Rp: "+str(mu_struc[0,0]) #ensure safe extrapolation
    mu_array = interp1d(mu_struc[:,0], mu_struc[:,1], fill_value='extrapolate')(r)

    vs = pw_parker.sound_speed(T, mu_bar)  # Speed of sound (km/s, assumed to be constant)
    if no_tidal:
        rs = pw_parker.radius_sonic_point(M_pl, vs)  # Radius at the sonic point (jupiterRad)
        rhos = pw_parker.density_sonic_point(m_dot, rs, vs)  # Density at the sonic point (g/cm^3)
        r_array = r * R_pl / rs
        v_array, rho_array = pw_parker.structure(r_array)
    else:
        Mstar = planet.Mstar / tools.Msun #convert from g to Msun
        a = planet.a / tools.AU #convert from cm to AU
        rs = pw_parker.radius_sonic_point_tidal(M_pl, vs, Mstar, a) #radius at the sonic point (jupiterRad)
        rhos = pw_parker.density_sonic_point(m_dot, rs, vs)  # Density at the sonic point (g/cm^3)
        r_array = r * R_pl / rs
        v_array, rho_array = pw_parker.structure_tidal(r_array, vs, rs, M_pl, Mstar, a)

    save_array = np.column_stack((r*planet.R, rho_array*rhos, v_array*vs*1e5, mu_array))
    save_name = tools.projectpath+'/parker_profiles/'+planet.name+'/'+pdir+'/temp/pprof_'+planet.name+'_T='+str(T)+'_M='+"%.3f" %Mdot +".txt"
    zdictstr = "abundance scale factors relative to solar:"
    for sp in zdict.keys():
        zdictstr += " "+sp+"="+"%.1f" %zdict[sp]
    np.savetxt(save_name, save_array, delimiter='\t', header=zdictstr+"\nalt rho v mu")

    launch_velocity = v_array[0] #velocity at Rp in units of sonic speed

    return save_name, launch_velocity


def run_parker_with_cloudy(filename, T, planet, zdict):
    """Runs an isothermal Parker wind profile through Cloudy, using the isothermal temperature profile.

    Parameters
    ----------
    filename : str
        Full path + filename of the isothermal Parker wind profile.
        Typically $SUNBATHER_PROJECT_PATH/parker_profiles/*planetname*/*pdir*/*filename*
    T : numeric
        Isothermal temperature value.
    planet : tools.Planet
        Object storing the planet parameters.
    zdict : dict
        Dictionary with the scale factors of all elements relative
        to the default solar composition. Can be easily created with tools.get_zdict().

    Returns
    -------
    simname : str
        Full path + name of the Cloudy simulation file without file extension.
    pprof : pandas.DataFrame
        Radial density, velocity and mean particle mass profiles of the isothermal Parker wind profile.
    """

    pprof = tools.read_parker('', '', '', '', filename=filename)

    altmax = pprof.alt.iloc[-1] / planet.R #maximum altitude of the profile in units of Rp
    alt = pprof.alt.values
    hden = tools.rho_to_hden(pprof.rho.values, abundances=tools.get_abundances(zdict))
    dlaw = tools.alt_array_to_Cloudy(alt, hden, altmax, planet.R, 1000, log=True)

    nuFnu_1AU_linear, Ryd = tools.get_SED_norm_1AU(planet.SEDname)
    nuFnu_a_log = np.log10(nuFnu_1AU_linear / ((planet.a - altmax*planet.R)/tools.AU)**2)

    simname = filename.split('.txt')[0]
    tools.write_Cloudy_in(simname, title='Simulation of '+filename, overwrite=True,
                                flux_scaling=[nuFnu_a_log, Ryd], SED=planet.SEDname,
                                dlaw=dlaw, double_tau=True, cosmic_rays=True, zdict=zdict, constantT=T, outfiles=['.ovr'])

    tools.run_Cloudy(simname)

    return simname, pprof


def calc_mu_bar(sim):
    """
    Calculates the weighted mean of the radial mean particle mass profile,
    according to Eq. A.3 of Lampon et al. (2020). Code adapted from 
    p_winds.parker.average_molecular_weight().

    Parameters
    ----------
    sim : tools.Sim
        Cloudy simulation output object.

    Returns
    -------
    mu_bar : float
        Weighted mean of the mean particle mass.
    """

    # Converting units
    m_planet = sim.p.M / 1000. #planet mass in kg
    r = sim.ovr.alt.values[::-1] / 100.  # Radius profile in m
    v_r = sim.ovr.v.values[::-1] / 100.  # Velocity profile in unit of m / s
    temperature = sim.ovr.Te.values[0] # (Isothermal) temperature in units of K

    # Physical constants
    k_b = 1.380649e-23  # Boltzmann's constant in J / K
    grav = 6.6743e-11  # Gravitational constant in m ** 3 / kg / s ** 2

    # Mean molecular weight in function of radial distance r
    mu_r = sim.ovr.mu.values[::-1]

    # Eq. A.3 of Lampón et al. 2020 is a combination of several integrals, which
    # we calculate here
    int_1 = simpson(mu_r / r ** 2, r)
    int_2 = simpson(mu_r * v_r, v_r)
    int_3 = trapz(mu_r, 1 / mu_r)
    int_4 = simpson(1 / r ** 2, r)
    int_5 = simpson(v_r, v_r)
    int_6 = 1 / mu_r[-1] - 1 / mu_r[0]
    term_1 = grav * m_planet * int_1 + int_2 + k_b * temperature * int_3
    term_2 = grav * m_planet * int_4 + int_5 + k_b * temperature * int_6
    mu_bar = term_1 / term_2

    return mu_bar


def save_cloudy_parker_profile(planet, Mdot, T, zdict, pdir, 
                               convergence=0.01, maxit=7, cleantemp=False, 
                               overwrite=False, verbose=False,
                               no_tidal=False, altmax=20):
    """
    Calculates an isothermal Parker wind profile with any composition by iteratively
    running the p-winds code (dos Santos et al. 2022) and Cloudy (Ferland et al. 1998; 2017, 
    Chatziokos et al. 2023). This function works iteratively as follows:
    p_winds calculates a density profile, Cloudy calculates the mean particle mass profile,
    we calculate the associated mu_bar value, which is passed to p-winds to calculate a new
    density profile, until mu_bar has converged to a stable value.
    Saves a 'pprof' txt file with the r, rho, v, mu structure.

    Parameters
    ----------
    planet : tools.Planet
        Object storing the planet parameters.
    Mdot : str or numeric
        log of the mass-loss rate in units of g s-1.
    T : str or numeric
        Temperature in units of K.
    zdict : dict
        Dictionary with the scale factors of all elements relative
        to the default solar composition. Can be easily created with tools.get_zdict().
    pdir : str
        Directory as $SUNBATHER_PROJECT_PATH/parker_profiles/planetname/*pdir*/
        where the isothermal parker wind density and velocity profiles are saved.
        Different folders may exist there for a given planet, to separate for example profiles
        with different assumptions such as stellar SED/semi-major axis/composition.
    convergence : float, optional
        Convergence threshold expressed as the relative change in mu_bar between iterations, by default 0.01
    maxit : int, optional
        Maximum number of iterations, by default 7
    cleantemp : bool, optional
        Whether to remove the temporary files in the /temp folder. These files store
        the intermediate profiles during the iterative process to find mu_bar. By default False.
    overwrite : bool, optional
        Whether to overwrite existing models, by default False.
    verbose : bool, optional
        Whether to print diagnostics about the convergence of mu_bar, by default False
    no_tidal : bool, optional
        Whether to neglect tidal gravity - fourth term of Eq. 4 of Linssen et al. (2024).
        See also Appendix D of Vissapragada et al. (2022) for the p-winds implementation.
        Default is False, i.e. tidal gravity included.
    altmax : int, optional
        Maximum altitude of the profile in units of the planet radius. By default 20.
    """

    save_name = tools.projectpath+'/parker_profiles/'+planet.name+'/'+pdir+'/pprof_'+planet.name+'_T='+str(T)+'_M='+ \
                            "%.3f" %Mdot +".txt"
    if os.path.exists(save_name) and not overwrite:
        print("Parker profile already exists and overwrite = False:", planet.name, pdir, "%.3f" %Mdot, T)
        return #this quits the function but if we're running a grid, it doesn't quit the whole Python code

    tools.verbose_print("Making initial parker profile while assuming a completely neutral mu_bar...", verbose=verbose)
    neutral_mu_bar = calc_neutral_mu(zdict)
    neutral_mu_struc = np.array([[1., neutral_mu_bar], [altmax, neutral_mu_bar]]) #set up an array with constant mu(r) at the neutral value
    filename, launch_velocity = save_temp_parker_profile(planet, Mdot, T, zdict, pdir, 
                                                            neutral_mu_bar, mu_struc=neutral_mu_struc, no_tidal=no_tidal, altmax=altmax)
    tools.verbose_print(f"Saved temp parker profile with neutral mu_bar: {neutral_mu_bar}" , verbose=verbose)
    previous_mu_bar = neutral_mu_bar

    for itno in range(maxit):
        tools.verbose_print(f"Iteration number: {itno+1}", verbose=verbose)
        
        tools.verbose_print("Running parker profile through Cloudy...", verbose=verbose)
        simname, pprof = run_parker_with_cloudy(filename, T, planet, zdict)
        tools.verbose_print("Cloudy run done.", verbose=verbose)

        sim = tools.Sim(simname, altmax=altmax, planet=planet)
        sim.addv(pprof.alt, pprof.v) #add the velocity structure to the sim, so that calc_mu_bar() works.

        mu_bar = calc_mu_bar(sim)
        tools.verbose_print(f"Making new parker profile with p-winds based on Cloudy's reported mu_bar: {mu_bar}", verbose=verbose)
        mu_struc = np.column_stack((sim.ovr.alt.values[::-1]/planet.R, sim.ovr.mu[::-1].values)) #pass Cloudy's mu structure to save in the pprof
        filename, launch_velocity = save_temp_parker_profile(planet, Mdot, T, zdict, pdir, 
                                                    mu_bar, mu_struc=mu_struc, no_tidal=no_tidal, altmax=altmax)
        tools.verbose_print("Saved temp parker profile.", verbose=verbose)

        if np.abs(mu_bar - previous_mu_bar)/previous_mu_bar < convergence:
            print("mu_bar converged:", save_name)
            if launch_velocity > 1:
                warnings.warn(f"This Parker wind profile is supersonic already at Rp: {save_name}")
            break
        else:
            previous_mu_bar = mu_bar

    copyfile(filename, filename.split('temp/')[0] + filename.split('temp/')[1])
    tools.verbose_print("Copied final parker profile from temp to parent folder.", verbose=verbose)

    if cleantemp: #then we remove the temp files
        os.remove(simname+'.in')
        os.remove(simname+'.out')
        os.remove(simname+'.ovr')
        os.remove(filename)
        tools.verbose_print("Temporary files removed.", verbose=verbose)


def run_s(plname, pdir, Mdot, T, SEDname, fH, zdict, mu_conv, 
          mu_maxit, overwrite, verbose, no_tidal):
    """
    Calculates a single isothermal Parker wind profile.

    Parameters
    ----------
    plname : str
        Planet name (must have parameters stored in $SUNBATHER_PROJECT_PATH/planets.txt).
    pdir : str
        Directory as $SUNBATHER_PROJECT_PATH/parker_profiles/*plname*/*pdir*/
        where the isothermal parker wind density and velocity profiles are saved.
        Different folders may exist there for a given planet, to separate for example profiles
        with different assumptions such as stellar SED/semi-major axis/composition.
    Mdot : str or numeric
        log of the mass-loss rate in units of g s-1.
    T : str or numeric
        Temperature in units of K.
    SEDname : str
        Name of SED file to use. If SEDname is 'real', we use the name as
        given in the planets.txt file, but if SEDname is something else,
        we advice to use a separate pdir folder for this.
    fH : float or None
        Hydrogen abundance expressed as a fraction of the total. If a value is given,
        Parker wind profiles will be calculated using p-winds standalone with a H/He
        composition. If None is given, Parker wind profiles will be calculated using the
        p-winds/Cloudy iterative method and the composition is specified via the zdict argument.
    zdict : dict
        Dictionary with the scale factors of all elements relative
        to the default solar composition. Can be easily created with tools.get_zdict().
        Will only be used if fH is None, in which case the p-winds/Cloudy iterative method
        is applied.
    mu_conv : float
        Convergence threshold expressed as the relative change in mu_bar between iterations.
        Will only be used if fH is None, in which case the p-winds/Cloudy iterative method
        is applied.
    mu_maxit : int
        Maximum number of iterations for the p-winds/Cloudy iterative method. Will only
        be used if fH is None.
    overwrite : bool
        Whether to overwrite existing models.
    verbose : bool
        Whether to print diagnostics about the convergence of mu_bar.
    no_tidal : bool
        Whether to neglect tidal gravity - fourth term of Eq. 4 of Linssen et al. (2024).
        See also Appendix D of Vissapragada et al. (2022) for the p-winds implementation.
    """

    p = tools.Planet(plname)
    if SEDname != 'real':
        p.set_var(SEDname=SEDname)
    altmax = min(20, int((p.a - p.Rstar) / p.R)) #solve profile up to 20 Rp, unless the star is closer than that

    if fH != None: #then run p_winds standalone
        spectrum = cloudy_spec_to_pwinds(tools.cloudypath+'/data/SED/'+p.SEDname, 1., (p.a - altmax*p.R)/tools.AU) #assumes SED is at 1 AU
        save_plain_parker_profile(p, Mdot, T, spectrum, h_fraction=fH, pdir=pdir, overwrite=overwrite, no_tidal=no_tidal, altmax=altmax)
    else: #then run p_winds/Cloudy iterative scheme
        save_cloudy_parker_profile(p, Mdot, T, zdict, pdir, 
                                   convergence=mu_conv, maxit=mu_maxit, cleantemp=True, 
                                   overwrite=overwrite, verbose=verbose,
                                   no_tidal=no_tidal, altmax=altmax)


def catch_errors_run_s(*args):
    """
    Executes the run_s() function with provided arguments, while catching errors more gracefully.
    """

    try:
        run_s(*args)
    except Exception as e:
        traceback.print_exc()


def run_g(plname, pdir, cores, Mdot_l, Mdot_u, Mdot_s, 
          T_l, T_u, T_s, SEDname, fH, zdict, mu_conv, 
          mu_maxit, overwrite, verbose, no_tidal):
    """
    Calculates a grid of isothermal Parker wind models, by executing the run_s() function in parallel.

    Parameters
    ----------
    plname : str
        Planet name (must have parameters stored in $SUNBATHER_PROJECT_PATH/planets.txt).
    pdir : str
        Directory as $SUNBATHER_PROJECT_PATH/parker_profiles/*plname*/*pdir*/
        where the isothermal parker wind density and velocity profiles are saved.
        Different folders may exist there for a given planet, to separate for example profiles
        with different assumptions such as stellar SED/semi-major axis/composition.
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
    SEDname : str
        Name of SED file to use. If SEDname is 'real', we use the name as
        given in the planets.txt file, but if SEDname is something else,
        we advice to use a separate pdir folder for this.
    fH : float or None
        Hydrogen abundance expressed as a fraction of the total. If a value is given,
        Parker wind profiles will be calculated using p-winds standalone with a H/He
        composition. If None is given, Parker wind profiles will be calculated using the
        p-winds/Cloudy iterative method and the composition is specified via the zdict argument.
    zdict : dict
        Dictionary with the scale factors of all elements relative
        to the default solar composition. Can be easily created with tools.get_zdict().
        Will only be used if fH is None, in which case the p-winds/Cloudy iterative method
        is applied.
    mu_conv : float
        Convergence threshold expressed as the relative change in mu_bar between iterations.
        Will only be used if fH is None, in which case the p-winds/Cloudy iterative method
        is applied.
    mu_maxit : int
        Maximum number of iterations for the p-winds/Cloudy iterative method. Will only
        be used if fH is None.
    overwrite : bool
        Whether to overwrite existing models.
    verbose : bool
        Whether to print diagnostics about the convergence of mu_bar.
    no_tidal : bool
        Whether to neglect tidal gravity - fourth term of Eq. 4 of Linssen et al. (2024).
        See also Appendix D of Vissapragada et al. (2022) for the p-winds implementation.
    """

    p = multiprocessing.Pool(cores)

    pars = []
    for Mdot in np.arange(float(Mdot_l), float(Mdot_u)+1e-6, float(Mdot_s)): #1e-6 so that upper bound is inclusive
        for T in np.arange(int(T_l), int(T_u)+1e-6, int(T_s)).astype(int):
            pars.append((plname, pdir, Mdot, T, SEDname, fH, zdict, mu_conv, mu_maxit, overwrite, verbose, no_tidal))

    p.starmap(catch_errors_run_s, pars)
    p.close()
    p.join()




if __name__ == '__main__':

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
            if not hasattr(namespace, self.dest) or getattr(namespace, self.dest) is None:
                setattr(namespace, self.dest, {})
            for value in values:
                key, val = value.split('=')
                getattr(namespace, self.dest)[key] = float(val)


    t0 = time.time()

    parser = argparse.ArgumentParser(description="Creates 1D Parker profile(s) using the p_winds code and Cloudy.")

    parser.add_argument("-plname", required=True, help="planet name (must be in planets.txt)")
    parser.add_argument("-pdir", required=True, help="directory where the profiles are saved. It is adviced to choose a name that " \
                                        "somehow represents the chosen parameters, e.g. 'fH_0.9' or 'z=10'. The path will be $SUNBATHER_PROJECT_PATH/parker_profiles/pdir/")
    parser.add_argument("-Mdot", required=True, type=float, nargs='+', action=OneOrThreeAction, help="log10(mass-loss rate), or three values specifying a grid of " \
                                        "mass-loss rates: lowest, highest, stepsize. -Mdot will be rounded to three decimal places.")
    parser.add_argument("-T", required=True, type=int, nargs='+', action=OneOrThreeAction, help="temperature, or three values specifying a grid of temperatures: lowest, highest, stepsize.")
    parser.add_argument("-SEDname", type=str, default='real', help="name of SED to use. Must be in Cloudy's data/SED/ folder [default=SEDname set in planet.txt file]")
    parser.add_argument("-overwrite", action='store_true', help="overwrite existing profile if passed [default=False]")
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
    parser.add_argument("-verbose", action='store_true', help="print out mu-bar values of each iteration [default=False]")
    parser.add_argument("-no_tidal", action='store_true', help="neglect the stellar tidal gravity term [default=False, i.e. tidal term included]")
    args = parser.parse_args()

    if args.z != None:
        zdict = tools.get_zdict(z=args.z, zelem=args.zelem)
    else: #if z==None we should not pass that to the tools.get_zdict function
        zdict = tools.get_zdict(zelem=args.zelem)

    if args.fH != None and (args.zelem != {} or args.mu_conv != 0.01 or args.mu_maxit != 7):
        warnings.warn("The -zelem, -mu_conv, and -mu_maxit commands only combine with -z, not with -fH, so I will ignore their input.")

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
        run_s(args.plname, args.pdir, args.Mdot[0], args.T[0], args.SEDname, args.fH, zdict, args.mu_conv, args.mu_maxit, args.overwrite, args.verbose, args.no_tidal)
    elif (len(args.T) == 3 and len(args.Mdot) == 3): #then we run a grid over both parameters
        run_g(args.plname, args.pdir, args.cores, args.Mdot[0], args.Mdot[1], args.Mdot[2], args.T[0], args.T[1], args.T[2], args.SEDname, args.fH, zdict, args.mu_conv, args.mu_maxit, args.overwrite, args.verbose, args.no_tidal)
    elif (len(args.T) == 3 and len(args.Mdot) == 1): #then we run a grid over only T
        run_g(args.plname, args.pdir, args.cores, args.Mdot[0], args.Mdot[0], args.Mdot[0], args.T[0], args.T[1], args.T[2], args.SEDname, args.fH, zdict, args.mu_conv, args.mu_maxit, args.overwrite, args.verbose, args.no_tidal)
    elif (len(args.T) == 1 and len(args.Mdot) == 3): #then we run a grid over only Mdot
        run_g(args.plname, args.pdir, args.cores, args.Mdot[0], args.Mdot[1], args.Mdot[2], args.T[0], args.T[0], args.T[0], args.SEDname, args.fH, zdict, args.mu_conv, args.mu_maxit, args.overwrite, args.verbose, args.no_tidal)

    print("\nCalculations took", int(time.time()-t0) // 3600, "hours, ", (int(time.time()-t0)%3600) // 60, "minutes and ", (int(time.time()-t0)%60), "seconds.\n")
