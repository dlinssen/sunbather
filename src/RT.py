import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.integrate import trapezoid
import spectres
import tools


sigt0 = 2.654e-2 #cm2 s-1 = cm2 Hz, from Axner et al. 2004
ab_J = {'HD209458':[0.12448605, 0.30226135], 'WASP69':[0.26538000, 0.26402800],
        'WASP107':[0.28310570, 0.25304740], 'HATP18':[0.25805403, 0.26665960],
        'HATP11':[0.26726638, 0.26480215], 'WASP52':[0.23179000, 0.27731000],
        'HD189733':[0.22484019, 0.27953897], #limb darkening coefficients in J band
        'TOI1259Ab':[0.35, 0.14], 'TOI1268b':[0.34, 0.12], 'TOI1420b':[0.42, -0.03]} #for 1259, 1268 and 1420 in the He band as measured by Shreyas


def limbdark_quad(mu, ab):
    '''
    Quadratic limb darkening law from Claret & Bloemen 2011.
    Returns I(mu)/I(1). mu is cos(theta) with theta the angle between
    the normal direction and beam direction. The following holds:
    mu = sqrt(1 - (r/Rs)^2)     with r/Rs the fractional distance to the
    center of the star (i.e. =0 at stellar disk center and =1 at limb).
    '''

    a, b = ab #turn list into separate parameters
    return 1 - a*(1-mu) - b*(1-mu)**2


def avg_limbdark_quad(ab):
    '''
    Average of the quadratic limb darkening I(mu) over the stellar disk.

    ab: list of the two quadratic limb darkening coefficients
    '''

    a, b = ab #turn list into separate parameters
    rf = np.linspace(0, 1, num=1000) #sample the stellar disk in 1000 donuts
    rfm = (rf[:-1] + rf[1:])/2 #midpoints
    mu = np.sqrt(1 - rfm**2) #mu of each donut
    I = 1 - a*(1-mu) - b*(1-mu)**2 #I of each donut
    projsurf = np.pi*(rf[1:]**2 - rf[:-1]**2) #area of each donut

    I_avg = np.sum(I * projsurf) / np.pi

    return I_avg


def calc_tau(x, ndens, Te, vx, nu, nu0, m, sig0, gamma):
    '''
    Calculates optical depth using Eq. 19 from Oklopcic&Hirata 2018.
    Does this at once for all rays, lines and frequencies. When doing
    multiple lines at once, they must all be from the same species and
    same level so that m and ndens are the same for the different lines.
    So you can do e.g. helium triplet or Ly-series at once. The FinFout_1D()
    function does currently not make use of that (i.e. the helium triplet is
    calculated with three calls to this function).

    x:      depth values of the grid 1D
    ndens:  number density of the species 2D (ray, depth axes)
    Te:     temperature  2D (ray, depth axes)
    vx:     l.o.s. velocity  2D (ray, depth axes)
    nu:     frequencies to calculate 1D
    nu0:    central frequencies of the different lines 1D
    m:      mass of species in g
    sig0:   cross-section of the lines 1D, Eq. 20 from Oklopcic&Hirata 2018.
    gamma:  HWHM of Lorentzian line part, 1D

    The quantities are all treated as 4D here internally, where:
    axis 0: the frequency axis
    axis 1: the different spectral lines
    axis 2: the different rays
    axis 3: the x (depth) direction along each ray
    '''

    if not isinstance(nu0, np.ndarray):
        nu0 = np.array([nu0])
    if not isinstance(sig0, np.ndarray):
        sig0 = np.array([sig0])
    if not isinstance(gamma, np.ndarray):
        gamma = np.array([gamma])

    gaus_sigma = np.sqrt(tools.k * Te[None,None,:] / m) * nu0[None,:,None,None] / tools.c
    Delnu = (nu[:,None,None,None] - nu0[None,:,None,None]) - nu0[None,:,None,None] / tools.c * vx[None,None,:]
    tau_cube = trapezoid(ndens[None,None,:] * sig0[None,:,None,None] * voigt_profile(Delnu, gaus_sigma, gamma[None,:,None,None]), x=x)
    tau = np.sum(tau_cube, axis=1) #sum up the contributions of the different lines -> now tau has axis 0:freq, axis 1:rayno

    return tau


def calc_cum_tau(x, ndens, Te, vx, nu, nu0, m, sig0, gamma):
    '''
    Similar to the function 'calc_tau', except that this does not just give the
    total optical depth for each ray, but gives the cumulative optical depth at
    one specific frequency at each depth point into each ray. It can still
    calculate the contributions of multiple spectral lines at that frequency.

    x:      depth values of the grid 1D
    ndens:  number density of the species 2D (ray, depth axes)
    Te:     temperature  2D (ray, depth axes)
    vx:     l.o.s. velocity  2D (ray, depth axes)
    nu:     frequency to calculate (float!)
    nu0:    central frequencies of the different lines 1D
    m:      mass of species in g
    sig0:   cross-section of the lines 1D, Eq. 20 from Oklopcic&Hirata 2018.
    gamma:  HWHM of Lorentzian line part, 1D

    The quantities are all treated as 3D here internally, where:
    axis 0: the different spectral lines
    axis 1: the different rays
    axis 2: the x (depth) direction along each ray
    '''

    if not isinstance(nu0, np.ndarray):
        nu0 = np.array([nu0])
    if not isinstance(sig0, np.ndarray):
        sig0 = np.array([sig0])
    if not isinstance(gamma, np.ndarray):
        gamma = np.array([gamma])

    gaus_sigma = np.sqrt(tools.k * Te[None,:] / m) * nu0[:,None,None] / tools.c
    Delnu = (nu - nu0[:,None,None]) - nu0[:,None,None] / tools.c * vx[None,:]
    integrand = ndens[None,:] * sig0[:,None,None] * voigt_profile(Delnu, gaus_sigma, gamma[:,None,None])
    bin_tau = np.zeros_like(integrand)
    bin_tau[:,:,1:] = (integrand[:,:,1:] + np.roll(integrand, 1, axis=2)[:,:,1:])/2. * np.diff(x)[None,None,:]
    bin_tau = np.sum(bin_tau, axis=0) #sum up contribution of different lines, now tau_bins has same shape as Te
    cum_tau = np.cumsum(bin_tau, axis=1) #do cumulative sum over the x-direction

    return cum_tau, bin_tau


def tau_to_FinFout(b, tau, Rs, bp=0., ab=[0., 0.], a=0., phase=0.):
    '''
    Takes in tau values and calculates the Fin/Fout transit spectrum,
    using the stellar radius and optional limb darkening and transit phase
    parameters. If all set to 0 (default), uses planet at stellar disk center
    with no limb darkening.

    b:      impact parameters of the rays through the planet atmosphere (1D)
    tau:    optical depth values per frequency and ray (2D: freq, ray)
    Rs:     stellar radius in cm
    bp:     impact parameter of the planet w.r.t the star center 0<bp<1  [in units of Rs]
    ab:     quadratic limb darkening parameters (list of 2 values)
    a:      planet orbital semi-major axis in cm
    phase:  planetary orbital phase 0<phase<1 where 0 is mid-transit.
            my implementation of phase does not (yet) take into account the
            tidally-locked rotation of the planet. so you'll always see the
            exact same (mid-transit) projection of the planet, just against
            a different limb-darkened background.
    '''
    #add some impact parameters and tau=inf bins that make up the planet core:
    b = np.concatenate((np.linspace(0, b[0], num=50, endpoint=False), b))
    tau = np.concatenate((np.ones((np.shape(tau)[0], 50))*np.inf, tau), axis=1)

    projsurf = np.pi*(b[1:]**2 - b[:-1]**2) #ring surface of each ray
    phis = np.linspace(0, 2*np.pi, num=500, endpoint=False) #divide rings into different angles phi
    #rc = distance to stellar center. Axis 0: radial rings, axis 1: phi
    rc = np.sqrt((bp*Rs + b[:-1,None]*np.cos(phis[None,:]))**2 + (b[:-1,None]*np.sin(phis[None,:]) + a*np.sin(2*np.pi*phase))**2) #don't use last b value
    rc = ma.masked_where(rc > Rs, rc) #will ensure I is masked outside stellar projected disk
    mu = np.sqrt(1 - (rc/Rs)**2) #angle, see 'limbdark_quad' function
    I = limbdark_quad(mu, ab)
    Ir_avg = np.sum(I, axis=1) / len(phis) #average I per ray
    Is_avg = avg_limbdark_quad(ab) #average I of the full stellar disk

    FinFout = np.ones_like(tau[:,0]) - np.sum(((1 - np.exp(-tau[:,:-1])) * Ir_avg*projsurf/(Is_avg*np.pi*Rs**2)), axis=1)

    return FinFout


def read_NIST_lines(species, wavlower=None, wavupper=None):
    '''
    This function reads a table of lines form the NIST atomic database.

    To generate one such table for a given atom and wavlength range,
    go to NIST and request lines, output as CSV (tab-delimited) and copy
    into a file such as "C.txt" or "Fe.txt". Make sure that energies are in Ry,
    wavelengths in vacuum and add the fik values in advanced settings.
    '''

    spNIST = pd.read_table(tools.cloupy_path+'/RT_tables/'+species+'_lines_NIST.txt') #line info
    #remove lines with nan fik or Aik values. Note that lineno doesn't change (uses index instead of rowno.)
    spNIST = spNIST[spNIST.fik.notna()]
    spNIST = spNIST[spNIST['Aki(s^-1)'].notna()]
    if type(spNIST['Ei(Ry)'].iloc[0]) == str: #if there are no [](), the datatype will be float already
        #spNIST.loc[:,'Ei(Ry)'] = spNIST['Ei(Ry)'].str.replace('[', '').str.replace(']','').str.replace('(','').str.replace(')','').str.replace('+','').str.replace('x','').str.replace('?','').astype(float)
        spNIST['Ei(Ry)'] = spNIST['Ei(Ry)'].str.extract('(\d+)', expand=False).astype(float) #remove non-numeric characters such as [] and ()
    spNIST['sig0'] = sigt0 * spNIST.fik
    spNIST['nu0'] = tools.c*1e8 / (spNIST['ritz_wl_vac(A)']) #speed of light to AA/s
    spNIST['lorgamma'] = spNIST['Aki(s^-1)'] / (4*np.pi) #lorentzian gamma is not function of depth or nu. Value in Hz

    if wavlower != None:
        spNIST.drop(labels=spNIST.index[spNIST['ritz_wl_vac(A)'] <= wavlower], inplace=True)
    if wavupper != None:
        spNIST.drop(labels=spNIST.index[spNIST['ritz_wl_vac(A)'] >= wavupper], inplace=True)

    return spNIST


def FinFout_1D(sim, wavsAA, species, numrays=100, broad_voigt_width=1., bp=0., ab=[0., 0.], phase=0., a=0., **kwargs):
    '''
    Calculates Fin/Fout transit spectrum for a given wavelength range, and a given
    (list of) species. Includes limb darkening.

    sim:        'Sim' class object of a Cloudy simulation. Needs to have
                Planet and Parker objects as attributes.
    wavsAA:     wavelengths to calculate spectrum on, in Angstroms (1D)
    species:    string or list of species name(s) to calculate, e.g. H, Fe+, C, Mg2+
                this species must be present in Cloudy's .en and .den files
    numrays:    number of rays with different impact parameters we project the 1D structure to (int)
    broad_voigt_width:  a multiplication factor for the 'max_voigt_width'
                        parameter, which sets how far to either side of the line core
                        we still calculate optical depths for every line.
                        Standard value is 5 Gaussian standard deviations + 10 Lorentzian gammas.
                        For e.g. Lyman alpha, you probably need a >1 factor here,
                        since the far Lorentzian wings are probed.
    bp:         impact parameter of the planet w.r.t the star center 0<bp<1  [in units of Rs]
    ab:         quadratic limb darkening parameters (list of 2 values)
    a:          planet orbital semi-major axis in cm
    phase:      planetary orbital phase 0<phase<1 where 0 is mid-transit.
                my implementation of phase does not (yet) take into account the
                tidally-locked rotation of the planet. so you'll always see the
                exact same (mid-transit) projection of the planet, just against
                a different limb-darkened background.
    '''

    assert hasattr(sim, 'p'), "The sim must have an attributed Planet object"
    assert 'v' in sim.ovr.columns, "We need a velocity structure, such as that from adding a Parker object to the sim."

    Rroche, Rs, Rp = sim.p.Rroche, sim.p.Rstar, sim.p.R
    nus = tools.c*1e8 / wavsAA #Hz, converted c to AA/s

    r1 = sim.ovr.alt.values[::-1]
    Te1 = sim.ovr.Te.values[::-1]
    v1 = sim.ovr.v.values[::-1]

    #v1 *= 10.
    #print("Scaled all velocities by a factor 10 in RT.FinFout_1D()")

    b, x, Te = tools.project_1D_to_2D(r1, Te1, Rp, numb=numrays)
    b, x, vx = tools.project_1D_to_2D(r1, v1, Rp, numb=numrays, directional=True)

    state_ndens = {}
    tau = np.zeros((len(wavsAA), numrays))

    if isinstance(species, str):
        species = [species]

    found_lines = [] #will store nu0 of all lines that were used (might be nice to make it a dict per species in future!)
    notfound_lines = [] #will store nu0 of all lines that were not found

    for spec in species:
        spNIST = read_NIST_lines(spec, wavlower=wavsAA[0], wavupper=wavsAA[-1])

        for lineno in spNIST.index.values: #loop over all lines in the spNIST table.
            gaus_sigma_max = 0.5*np.sqrt(tools.k * np.nanmax(Te) / tools.get_mass(spec)) * spNIST.nu0.loc[lineno] / tools.c #maximum stddev of Gaussian part
            max_voigt_width = 5*(gaus_sigma_max+spNIST['lorgamma'].loc[lineno]) * broad_voigt_width #the max offset of Voigt components (=natural+thermal broad.)
            linenu_low = (1-np.max(vx)/tools.c) * spNIST.nu0.loc[lineno] - max_voigt_width
            linenu_hi = (1-np.min(vx)/tools.c) * spNIST.nu0.loc[lineno] + max_voigt_width

            nus_line = nus[(nus > linenu_low) & (nus < linenu_hi)] #the frequency values that make sense to calculate for this line
            if nus_line.size == 0: #then this line is not in our wav range and we skip it
                continue #to next spectral line

            #get all columns in .den file which energy corresponds to this Ei
            colname, lineweight = tools.find_line_lowerstate_in_en_df(spec, spNIST.loc[lineno], sim.en, printmessage=False)
            if colname == None: #we skip this line if the line energy is not found.
                notfound_lines.append(spNIST['ritz_wl_vac(A)'][lineno])
                continue #to next spectral line

            found_lines.append((spNIST['ritz_wl_vac(A)'].loc[lineno], colname)) #if we got to here, we did find the spectral line

            if colname in state_ndens.keys():
                ndens = state_ndens[colname]
            else:
                ndens1 = sim.den[colname].values[::-1]
                b, x, ndens = tools.project_1D_to_2D(r1, ndens1, Rp, numb=numrays, cut_at=Rroche, **kwargs) #in kwargs, there can be keywords to exlude regions (see tools.project_1D_to_2D)
                state_ndens[colname] = ndens #add to dictionary for future reference

            ndens_lw = ndens*lineweight #important that we make this a new variable as otherwise state_ndens would change as well!

            tau_line = calc_tau(x, ndens_lw, Te, vx, nus_line, spNIST.nu0.loc[lineno], tools.get_mass(spec), spNIST.sig0.loc[lineno], spNIST['lorgamma'].loc[lineno])
            tau[(nus > linenu_low) & (nus < linenu_hi), :] += tau_line #add the tau values to the correct nu bins

    FinFout = tau_to_FinFout(b, tau, Rs, bp=bp, ab=ab, phase=phase, a=a)

    return FinFout, found_lines, notfound_lines


def tau_1D(sim, wavAA, species, broad_voigt_width=1., **kwargs):
    '''
    This function maps out the optical depth at one specific wavelength.
    The running integral of the optical deph is calculated at each depth of the ray.
    Useful for plotting purposes (i.e. where does a line form).
    Keep in mind that this function maps out the optical depth along the direction
    of the Cloudy simulation (i.e. the substellar ray). To do proper RT calculations,
    you need to calculate the optical depth in the 2D plane, and the tau_12D()
    function can be used for that.

    sim:        'Sim' class object of a Cloudy simulation. Needs to have
                Planet and Parker objects as attributes.
    wavsAA:     wavelengths to calculate spectrum on, in Angstroms (1D)
    species:    string or list of species name(s) to calculate, e.g. H, Fe+, C, Mg2+
                this species must be present in Cloudy's .en and .den files
    numrays:    number of rays with different impact parameters we project the 1D structure to (int)
    broad_voigt_width:  a multiplication factor for the 'max_voigt_width'
                        parameter, which sets how far to either side of the line core
                        we still calculate optical depths for every line.
                        Standard value is 5 Gaussian standard deviations + 10 Lorentzian gammas.
                        For e.g. Lyman alpha, you probably need a >1 factor here,
                        since the far Lorentzian wings are probed.
    '''

    assert hasattr(sim, 'p'), "The sim must have an attributed Planet object"
    assert 'v' in sim.ovr.columns, "We need a velocity structure, such as that from adding a Parker object to the sim."

    Rroche, Rs, Rp = sim.p.Rroche, sim.p.Rstar, sim.p.R
    nu = tools.c*1e8 / wavAA #Hz, converted c to AA/s

    d = sim.ovr.depth.values
    Te = sim.ovr.Te.values
    v = sim.ovr.v.values[::-1]

    tot_cum_tau, tot_bin_tau = np.zeros_like(d), np.zeros_like(d)

    if isinstance(species, str):
        species = [species]

    found_lines = [] #will store nu0 of all lines that were used (might be nice to make it a dict per species in future!)
    notfound_lines = [] #will store nu0 of all lines that were not found

    for spec in species:
        spNIST = read_NIST_lines(spec)

        for lineno in spNIST.index.values: #loop over all lines in the spNIST table.
            gaus_sigma_max = 0.5*np.sqrt(tools.k * np.nanmax(Te) / tools.get_mass(spec)) * spNIST.nu0.loc[lineno] / tools.c #maximum stddev of Gaussian part
            max_voigt_width = 5*(gaus_sigma_max+spNIST['lorgamma'].loc[lineno]) * broad_voigt_width #the max offset of Voigt components (=natural+thermal broad.)
            linenu_low = (1-np.max(v)/tools.c) * spNIST.nu0.loc[lineno] - max_voigt_width
            linenu_hi = (1-np.min(v)/tools.c) * spNIST.nu0.loc[lineno] + max_voigt_width

            if (nu < linenu_low) | (nu > linenu_hi): #then this line does not probe our requested wav and we skip it
                continue #to next spectral line

            #get all columns in .den file which energy corresponds to this Ei
            colname, lineweight = tools.find_line_lowerstate_in_en_df(spec, spNIST.loc[lineno], sim.en, printmessage=False)
            if colname == None: #we skip this line if the line energy is not found.
                notfound_lines.append(spNIST['ritz_wl_vac(A)'][lineno])
                continue #to next spectral line

            found_lines.append((spNIST['ritz_wl_vac(A)'].loc[lineno], colname)) #if we got to here, we did find the spectral line

            ndens = sim.den[colname].values * lineweight #see explanation in FinFout_2D function

            cum_tau, bin_tau = calc_cum_tau(d, ndens, Te, v, nu, spNIST.nu0.loc[lineno], tools.get_mass(spec), spNIST.sig0.loc[lineno], spNIST['lorgamma'].loc[lineno])
            tot_cum_tau += cum_tau[0] #add the tau values to the total (of all species & lines together)
            tot_bin_tau += bin_tau[0]

    return tot_cum_tau, tot_bin_tau, found_lines, notfound_lines


def tau_12D(sim, wavAA, species, broad_voigt_width=1.):
    '''
    For a 1D simulation, still maps out the optical depth in 2D.
    '''

    assert hasattr(sim, 'p')
    assert 'v' in sim.ovr.columns, "We need a velocity structure, such as that from adding a Parker object to the sim."

    nu = tools.c*1e8 / wavAA #Hz, converted c to AA/s

    b, x, Te = tools.project_1D_to_2D(sim.ovr.alt.values[::-1], sim.ovr.Te.values[::-1], sim.p.R)
    b, x, vx = tools.project_1D_to_2D(sim.ovr.alt.values[::-1], sim.ovr.v.values[::-1], sim.p.R, directional=True)

    tot_cum_tau, tot_bin_tau = np.zeros_like(vx), np.zeros_like(vx)

    if isinstance(species, str):
        species = [species]

    found_lines = [] #will store nu0 of all lines that were used (might be nice to make it a dict per species in future!)
    notfound_lines = [] #will store nu0 of all lines that were not found

    for spec in species:
        spNIST = read_NIST_lines(spec)

        for lineno in spNIST.index.values: #loop over all lines in the spNIST table.
            gaus_sigma_max = 0.5*np.sqrt(tools.k * np.nanmax(Te) / tools.get_mass(spec)) * spNIST.nu0.loc[lineno] / tools.c #maximum stddev of Gaussian part
            max_voigt_width = 5*(gaus_sigma_max+spNIST['lorgamma'].loc[lineno]) * broad_voigt_width #the max offset of Voigt components (=natural+thermal broad.)
            linenu_low = (1-np.max(vx)/tools.c) * spNIST.nu0.loc[lineno] - max_voigt_width
            linenu_hi = (1-np.min(vx)/tools.c) * spNIST.nu0.loc[lineno] + max_voigt_width

            if (nu < linenu_low) | (nu > linenu_hi): #then this line does not probe our requested wav and we skip it
                continue #to next spectral line

            #get all columns in .den file which energy corresponds to this Ei
            colname, lineweight = tools.find_line_lowerstate_in_en_df(spec, spNIST.loc[lineno], sim.en, printmessage=False)
            if colname == None: #we skip this line if the line energy is not found.
                notfound_lines.append(spNIST['ritz_wl_vac(A)'][lineno])
                continue #to next spectral line

            found_lines.append((spNIST['ritz_wl_vac(A)'].loc[lineno], colname)) #if we got to here, we did find the spectral line

            #multiply with the lineweight! Such that for unresolved J, a line originating from J=1/2 does not also get density of J=3/2 state
            _, _, ndens = tools.project_1D_to_2D(sim.ovr.alt.values[::-1], sim.den[colname].values[::-1], sim.p.R, cut_at=sim.p.Rroche)
            ndens *= lineweight

            cum_tau, bin_tau = calc_cum_tau(x, ndens, Te, vx, nu, spNIST.nu0.loc[lineno], tools.get_mass(spec), spNIST.sig0.loc[lineno], spNIST['lorgamma'].loc[lineno])
            tot_cum_tau += cum_tau #add the tau values to the total (of all species & lines together)
            tot_bin_tau += bin_tau

    return tot_cum_tau, tot_bin_tau, found_lines, notfound_lines



def rebin_spec(wavein, specin, wavnew):
    '''
    Rebins a spectrum with flux conservation.
    '''

    return spectres.spectres(wavnew, wavein, specin)


def FinFout2RpRs(FinFout):
    '''
    Converts the Fin/Fout (i.e. flux in-transit / flux out-of-transit) to
    Rp/Rs (i.e. the apparent size of the planet relative to the star).
    The continuum should be roughly Rp/Rs, but not necessarily exactly if
    limb-darkening was used.
    '''

    return np.sqrt(1-FinFout)


def RpRs2FinFout(RpRs):
    '''
    Reverse function of FinFout2RpRs().
    '''

    return 1-RpRs**2


def vac2air(wavs_vac):
    '''
    Converts vacuum wavelengths to air. Wavelengths MUST be in Angstroms.
    from: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    '''

    s = 1e4 / wavs_vac
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    wavs_air = wavs_vac / n

    return wavs_air


def air2vac(wavs_air):
    '''
    Converts air wavelengths to vacuum. Wavelengths MUST be in Angstroms.
    from: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    '''

    s = 1e4 / wavs_air
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    wavs_vac = wavs_air * n

    return wavs_vac
