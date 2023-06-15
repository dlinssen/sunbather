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



def Cl_FinFout_1D(sim):
    '''
    Calculates FinFout with Cloudy's own RT.
    Since we cannot project the flux values to a 2D grid, this function
    does not really consider a planetary transit, but rather just returns
    the ratio of incoming to outgoing flux, both for the case where we
    only consider absorption of the gas, as well as the case where
    emission is taken into account.
    '''
    FinFout = (sim.con.trans-sim.con.incident)/sim.con.incident
    FinFout_em = (sim.con.nettrans-sim.con.incident)/sim.con.incident

    return FinFout, FinFout_em


def Cl_FinFout_2D(sim_obj):
    '''
    Calculates Fin/Fout for a 2D atmosphere with Cloudy's own RT.

    For both the case where we don't include the emission from the gas,
    as well as the case where emission is taken into account.

    Currently does not include limb darkening or a transit impact parameter yet.
    And we do not cut at the roche lobe yet, because that's hard to implement,
    we cannot do that as a spherical surface, but we could cut out rays with
    impact parameters higher than the roche radius.
    '''
    Rs = tools.Planet(sim_obj.plname).Rstar
    wavsAA = sim_obj.sims[0].con.wav[sim_obj.sims[0].con.incident != 0] #take wavelength grid of first ray (where incident is nonzero)
    FinFout = np.ones(len(wavsAA))
    FinFout -= (sim_obj.Rp/Rs)**2 #core transit depth
    FinFout_em = np.copy(FinFout)

    for rayno in range(sim_obj.nsteps-1): #not including last 'ray' which is a point
        ray = sim_obj.sims[rayno]
        projsurf = np.pi*(sim_obj.bs[rayno+1])**2 - np.pi*(sim_obj.bs[rayno])**2

        ray.con = ray.con[ray.con.incident != 0] #otherwise division by zero
        assert np.all(ray.con.wav == wavsAA), "Trying to add .con spectra of rays with different grids."

        ext = 1. - ray.con.trans / ray.con.incident #extinction fraction
        FinFout -= ext * projsurf/(np.pi*Rs**2)

        ext_em = 1. - ray.con.nettrans / ray.con.incident #also including the emission from the gas
        FinFout_em -= ext_em * projsurf/(np.pi*Rs**2)

    FinFout = np.clip(FinFout, 0., 1.) #to prevent negative FinFout if the cloud remains optically thick beyond the stellar disk
    FinFout_em = np.clip(FinFout_em, 0., np.inf)
    #------>
    #------> THIS SHOULD BE CHANGED TO USING LIMB DARKENING AND IMPACT PARAMETER
    #------>

    return wavsAA.values[::-1], FinFout.values[::-1], FinFout_em.values[::-1]


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
    Average of the quadratic limb darkening I over the stellar disk.

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
    So you can do e.g. Helium triplet or Ly-series at once.

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
    This reads a table of lines form the NIST atomic database.

    To generate one such table for a given atom and wavlength range,
    go to NIST and request lines, output as CSV (tab-delimited) and copy
    into a file such as "C.txt" or "Fe.txt". Make sure that energies are in Ry,
    take note of the vacuum/air distinction and add the fik values in advanced
    settings.
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


def FinFout_2D(sim_obj, wavsAA, species, broad_voigt_width=1., bp=0., ab=[0., 0.], phase=0., a=0., Rs=None):
    '''
    Calculates Fin/Fout transit spectrum for a given wavelength range, and a given
    (list of) species. Includes limb darkening.

    sim_obj:    'Sim2D' class object of a 2D Cloudy simulation
    wavsAA:     wavelengths to calculate spectrum on, in Angstroms (1D)
    species:    string or list of species name(s) to calculate, e.g. H, Fe+, C, Mg2+
                this species must be present in Cloudy's .en and .den files
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

    if sim_obj.plname == '': #then Rs needs to be given as argument! Otherwise error will follow.
        Rroche = np.inf
    else:
        Rroche = sim_obj.p.Rroche
        Rs = sim_obj.p.Rstar

    Rp = sim_obj.Rp
    nus = tools.c*1e8 / wavsAA #Hz, converted c to AA/s
    b = sim_obj.bs
    x = sim_obj.vxstruc[:,0]

    xx, bb = np.meshgrid(x, b) #don't use the last ray, that's only to get the before-last 'ring' size
    rr = np.sqrt(bb**2 + xx**2) #radii from planet core in 2D, needed to cut ndens to 0 outside Rroche

    vx = sim_obj.vxstruc[:,1:].T #because first column is the x values
    Te = np.genfromtxt(sim_obj.path+'grid/Te.txt').T
    tau = np.zeros((len(wavsAA), sim_obj.nsteps))

    Te[np.isnan(Te)] = 0. #ndens will also be 0 there so it won't contribute to optical depth

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
            colname, lineweight = tools.find_line_lowerstate_in_en_df(spec, spNIST.loc[lineno], sim_obj.sims[0].en, printmessage=False)
            if colname == None: #we skip this line if the line energy is not found.
                notfound_lines.append(spNIST['ritz_wl_vac(A)'][lineno])
                continue #to next spectral line

            found_lines.append((spNIST['ritz_wl_vac(A)'].loc[lineno], colname)) #if we got to here, we did find the spectral line

            #multiply with the lineweight! Such that for unresolved J, a line originating from J=1/2 does not also get density of J=3/2 state
            ndens = np.genfromtxt(sim_obj.path+'/grid/'+colname+'.txt').T * lineweight #number density of the species that is in the lower level
            ndens[rr > Rroche] = 0.

            tau_line = calc_tau(x, ndens, Te, vx, nus_line, spNIST.nu0.loc[lineno], tools.get_mass(spec), spNIST.sig0.loc[lineno], spNIST['lorgamma'].loc[lineno])
            tau[(nus > linenu_low) & (nus < linenu_hi), :] += tau_line #add the tau values to the correct nu bins

    FinFout = tau_to_FinFout(b, tau, Rs, bp=bp, ab=ab, phase=phase, a=a)

    return FinFout, found_lines, notfound_lines


def tau_2D(sim_obj, wavAA, species, broad_voigt_width=1.):
    '''
    This function maps out the optical depth at one specific wavelength.
    The running integral of the optical deph is calculated at each depth of each ray.
    Useful for plotting purposes (i.e. where does a line form).

    sim_obj:    'Sim2D' class object of a 2D Cloudy simulation
    wavAA:      wavelength to calculate the optical depth, in Angstroms (float)
    species:    string or list of species name(s) to calculate, e.g. H, Fe+, C, Mg2+
                this species must be present in Cloudy's .en and .den files
    broad_voigt_width:  a multiplication factor for the 'max_voigt_width'
                        parameter, which sets how far to either side of the line core
                        we still calculate optical depths for every line.
                        Standard value is 5 Gaussian standard deviations + 10 Lorentzian gammas.
                        For e.g. Lyman alpha, you probably need a >1 factor here,
                        since the far Lorentzian wings are probed.
    '''
    if sim_obj.plname == '':
        Rroche = np.inf
    else:
        pl = sim_obj.p
        Rroche = tools.roche_radius(pl.a, pl.M*tools.MJ, pl.Mstar*tools.Msun)

    Rp = sim_obj.Rp
    nu = tools.c*1e8 / wavAA #Hz, converted c to AA/s
    b = sim_obj.bs
    x = sim_obj.vxstruc[:,0]

    xx, bb = np.meshgrid(x, b) #here we do use the last ray
    rr = np.sqrt(bb**2 + xx**2) #radii from planet core in 2D, needed to cut ndens to 0 outside Rroche

    vx = sim_obj.vxstruc[:,1:].T
    Te = np.genfromtxt(sim_obj.path+'/grid/Te.txt').T
    tot_cum_tau, tot_bin_tau = np.zeros_like(vx), np.zeros_like(vx)

    Te[np.isnan(Te)] = 0. #ndens will also be 0 there so it won't contribute to optical depth

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
            colname, lineweight = tools.find_line_lowerstate_in_en_df(spec, spNIST.loc[lineno], sim_obj.sims[0].en, printmessage=False)
            if colname == None: #we skip this line if the line energy is not found.
                notfound_lines.append(spNIST['ritz_wl_vac(A)'][lineno])
                continue #to next spectral line

            found_lines.append((spNIST['ritz_wl_vac(A)'].loc[lineno], colname)) #if we got to here, we did find the spectral line

            #multiply with the lineweight! Such that for unresolved J, a line originating from J=1/2 does not also get density of J=3/2 state
            ndens = np.genfromtxt(sim_obj.path+'/grid/'+colname+'.txt').T * lineweight #number density of the species that is in the lower level
            ndens[rr > Rroche] = 0.

            cum_tau, bin_tau = calc_cum_tau(x, ndens, Te, vx, nu, spNIST.nu0.loc[lineno], tools.get_mass(spec), spNIST.sig0.loc[lineno], spNIST['lorgamma'].loc[lineno])
            tot_cum_tau += cum_tau #add the tau values to the total (of all species & lines together)
            tot_bin_tau += bin_tau

    return tot_cum_tau, tot_bin_tau, found_lines, notfound_lines


def FinFout_1D(sim, wavsAA, species, numrays=100, broad_voigt_width=1., bp=0., ab=[0., 0.], phase=0., a=0., **kwargs):
    '''
    Calculates Fin/Fout transit spectrum for a given wavelength range, and a given
    (list of) species. Includes limb darkening.

    sim:        'Sim' class object of a Cloudy simulation. Needs to have
                Planet object as attribute, and either Parker object as attribute
                or vstruc given as kwarg.
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
    vstruc:    two-column array of altitude and velocity values (optional kwarg, 2D). if not
                given, sim needs to have Parker object set.
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

    sim:        'Sim' class object of a Cloudy simulation
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
    **kwargs=
    vstruc:    two-column array of depth and velocity values (optional, 2D).
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
    return spectres.spectres(wavnew, wavein, specin)


def convolve_HST(FinFout, grating, aperture):
    '''
    Works only in AA. Is interpolation to linear grid the right thing to do?
    Or does that not preserve flux?
    '''
    if grating == 'G230M': #constant Delta lambda
        LSF = pd.read_table(tools.projectpath+'/STIS_LSF/LSF_'+grating+'_2400.txt', header=1,
                                names=['relpix', '52x0.1', '52x0.2', '52x0.5', '52x2.0'],
                                delim_whitespace=True, index_col=False)

        AAperpixel = {'G230M':0.09}

        LSF_area = trapezoid(LSF[aperture], x=range(len(LSF))) #needed for normalization. Cst Delta lambda spacing

        ifunc = interp1d(FinFout[:,0], FinFout[:,1], kind='linear')
        grid = np.arange(FinFout[0,0], FinFout[-1,0], AAperpixel[grating])
        td = ifunc(grid)

        tdc = np.convolve(td, LSF[aperture], mode='same') / LSF_area

    elif grating == 'E230M': #not constant Delta lambda but constant resolution
        LSF = pd.read_table(tools.projectpath+'/STIS_LSF/LSF_'+grating+'_2400.txt', header=1,
                                names=['relpix', '0.1x0.03', '0.2x0.06', '0.2x0.2', '6x0.2'],
                                delim_whitespace=True, index_col=False)

        Dellamb = {'E230M':1./60000.}

        LSF_area = trapezoid(LSF[aperture], x=LSF['relpix']) #in units of pixels

        ifunc = interp1d(FinFout[:,0], FinFout[:,1], kind='linear')
        grid = [FinFout[0,0]] #start making a grid at constant lamb/60000 resolution. This will be 'where the pixels are spaced'
        while grid[-1] < FinFout[-1,0]:
            grid.append(grid[-1] * (1. + Dellamb[grating]))
        td = ifunc(grid)

        tdc = np.convolve(td, LSF[aperture], mode='same') / LSF_area

    return np.column_stack((grid, tdc))


def FinFout2RpRs(FinFout):
    return np.sqrt(1-FinFout)


def RpRs2FinFout(RpRs):
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


def calc_sigEW_Vollmann2006(Fl, Fc, dlamb, EW, SN):
    '''
    For a given EW, calculates the error on it.
    From Vollmann K, Eversberg T (2006)
    '''
    return np.sqrt(1 + Fc/Fl) * (dlamb - EW)/SN


def calc_sigEW_Cayrel1988(line_std, dx, rms_er):
    '''
    Calculates the error on EW. Assumes a Gaussian line profile.
    From Cayrel R (2006)
    '''
    return np.sqrt(3*np.pi) / np.pi**(0.25) * np.sqrt(line_std * dx) * rms_er


def bootstrap_EW(wav, F, sigF, n=5000, correlated=False):
    '''
    Bootstraps a spectral line dataset to get the error on the EW.
    sigF is a constant or a array with same length as F.

    If correlated=True, the error on every two neighbouring points
    will be the same, so if one point falls 2sigma above the observed value,
    the next one will as well, etc.
    '''

    EWs = np.zeros(n)
    excessabs = np.zeros(n)

    for it in range(n):
        dF = np.random.normal(loc=np.repeat(0., len(wav)), scale=sigF)
        if correlated:
            if len(dF)%2 == 0: #even array length
                dF[1::2] = dF[0::2] #change every other element to the previous one
            if len(dF)%2 == 1: #uneven array length
                dF[1:-1:2] = dF[0:-1:2] #change every other element to the previous one
        mockF = F + dF
        mockEW = trapezoid(mockF, x=wav)
        EWs[it] = mockEW
        excessabs[it] = np.mean(mockF)

    return np.mean(EWs), np.std(EWs), np.mean(excessabs), np.std(excessabs)


def save_strong_lines2D(sim2d, species=['H', 'He', 'C', 'C+', 'Na', 'Mg', 'Mg+', 'Si+', 'Si+2', 'Fe+']):
    wavs = np.logspace(np.log10(911), np.log10(20000), num=int(1e6))
    tds = pd.DataFrame(columns=species, index=wavs)
    for sp in species:
        print("Species:", sp)
        w = 1.
        if sp == 'H' or sp == 'Fe+' or sp == 'Mg+' or sp=='Si+':
            w = 10.
        td, fl, nfl = FinFout_2D(sim2d, wavs, sp, broad_voigt_width=w)

        tds[sp] = td
        np.savetxt(sim2d.path+'td/not_found_lines_'+sp+'.txt', np.array(nfl), fmt='%1.10e')
        with open(sim2d.path+'td/found_lines_'+sp+'.txt', 'w') as f:
            for line in fl:
                f.write("%1.10e" %line[0] +","+line[1]+"\n")

    continuum = np.max(tds.values)
    tds[tds == continuum] = np.nan #saves a lot in the file size
    with open(sim2d.path+'td/tdspecies.txt', 'w') as f:
        f.write('# empty/nan values are the continuum Fin/Fout='+"%1.10e" % continuum +"\n")
        tds.to_csv(f, float_format='%1.10e')

    print("Species: all")
    td, fl, nfl = FinFout_2D(sim2d, wavs, species, broad_voigt_width=10.)
    td[td == continuum] = np.nan
    np.savetxt(sim2d.path+'td/td.txt', np.column_stack((wavs, td)), fmt='%1.10e', header='empty/nan values are the continuum Fin/Fout='+"%1.10e" % continuum)


def save_strong_lines1D(sim, tdpath, species=tools.get_specieslist(), together=False, separate=False, **kwargs):
    if 'wavs' in kwargs:
        wavs = kwargs['wavs']
    else:
        wavs = np.logspace(np.log10(911), np.log10(20000), num=int(1e6))

    if separate:
        tds = pd.DataFrame(columns=species, index=wavs)
        for sp in species:
            print("Species:", sp)
            td, fl, nfl = FinFout_1D(sim, wavs, sp, numrays=100, **kwargs)

            tds[sp] = td
            np.savetxt(tdpath+'not_found_lines_'+sp+'.txt', np.array(nfl), fmt='%1.10e')
            with open(tdpath+'found_lines_'+sp+'.txt', 'w') as f:
                for line in fl:
                    f.write("%1.10e" %line[0] +","+line[1]+"\n")

        continuum = np.max(tds.values)
        tds[tds == continuum] = np.nan #saves a lot in the file size
        #with open(tdpath+'tdspecies.txt', 'w') as f:
        with open(tdpath+'tdspecies.txt', 'w') as f:
            f.write('# empty/nan values are the continuum Fin/Fout='+"%1.10e" % continuum +"\n")
            tds.to_csv(f, float_format='%1.10e')

    if together:
        print("Species: all")
        td, fl, nfl = FinFout_1D(sim, wavs, species, numrays=100, broad_voigt_width=10., **kwargs)
        continuum = np.max(td)
        td[td == continuum] = np.nan
        np.savetxt(tdpath+'td.txt', np.column_stack((wavs, td)), fmt='%1.10e', header='empty/nan values are the continuum Fin/Fout='+"%1.10e" % continuum)
        np.savetxt(tdpath+'not_found_lines_all.txt', np.array(nfl), fmt='%1.10e')
        with open(tdpath+'found_lines_all.txt', 'w') as f:
            for line in fl:
                f.write("%1.10e" %line[0] +","+line[1]+"\n")


def read_td(filename):
    '''
    Reads in a (big) transit depths file. To save file size, those files do not print the continuum
    value for every wavelength, but save it in the header. This function reads the td file
    and puts the continuum values back in.
    '''

    if 'species' in filename:
        with open(filename, 'r') as f:
            headerline = f.readline() #extract the continuum level from the first (header) line
            continuum = float(headerline.split('=')[-1])
            df = pd.read_csv(f, index_col=0) #then read the rest of the file
        df[np.isnan(df)] = continuum
        return df

    else:
        with open(filename, 'r') as f:
            headerline = f.readline() #extract the continuum level from the first (header) line
            continuum = float(headerline.split('=')[-1])
            ar = np.genfromtxt(f) #then read the rest of the file
        ar[np.isnan(ar[:,1]),1] = continuum
        return ar


def combine_speciestd_files(filenames, outfilename):
    '''
    Function to combine different 'tdspecies' files, assuming they have the same index,
    i.e. wavelength grid. This is mainly useful when first making speciestd.txt files with
    e.g. excluding calcium, then later adding calcium and then this function combines the td files.
    '''
    raise Exception("This function adds duplicate columns if they are already present in the dataframe. Fix that issue first.")
    assert len(filenames) > 1

    #first open the first one
    with open(filenames[0], 'r') as f:
        headerline = f.readline() #extract the continuum level from the first (header) line
        continuum = float(headerline.split('=')[-1])
        df = pd.read_csv(f, index_col=0) #then read the rest of the file

    #then read the rest and add them
    for filename in filenames[1:]:
        with open(filename, 'r') as f:
            headerline = f.readline() #extract the continuum level from the first (header) line
            continuum2 = float(headerline.split('=')[-1])
            df2 = pd.read_csv(f, index_col=0) #then read the rest of the file

            assert continuum2 == continuum
            df = pd.concat([df, df2[df2.columns.difference(df.columns)]], axis='columns') #assumes that they have the same indexes (which they should)
            print(df)

    #save the combined df
    with open(outfilename, 'w') as f:
        f.write('# empty/nan values are the continuum Fin/Fout='+"%1.10e" % continuum +"\n")
        df.to_csv(f, float_format='%1.10e')




if __name__ == '__main__':
    pass
