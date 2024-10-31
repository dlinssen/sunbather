import warnings
import pandas as pd
import numpy as np
from numpy import ma
from scipy.interpolate import interp1d
from scipy.special import voigt_profile
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d

from sunbather import tools

sigt0 = 2.654e-2  # cm2 s-1 = cm2 Hz, from Axner et al. 2004


def project_1D_to_2D(
    r1,
    q1,
    Rp,
    numb=101,
    x_projection=False,
    cut_at=None,
    skip_alt_range=None,
    skip_alt_range_dayside=None,
    skip_alt_range_nightside=None,
):
    """
    Projects a 1D sub-stellar solution onto a 2D grid. This function preserves
    the maximum altitude of the 1D ray, so that the 2D output looks like a half
    circle. Values in the numpy 2D array outside of the circle radius are set to
    0. This will also ensure 0 density and no optical depth.

    Parameters
    ----------
    r1 : array_like
        Altitude values from planet core in cm (ascending!).
    q1 : array_like
        1D quantity to project.
    Rp : numeric
        Planet core radius in cm. Needed because we start there, and not
        necessarily at the lowest r-value (which may be slightly r[0] != Rp).
    numb : int, optional
        The number of bins in the y-directtion (impact parameters).
        Twice this number is used in the x-direction (l.o.s.). Default is 101.
    x_projection : bool, optional
        Whether to return the projection of q1(r1) in the x direction.
        For example for radial outflow velocities, to convert it to a velocity in
        the x-direction, set this to True so that you get v_x, where positive v_x
        are in the x-direction, i.e. from the star towards the observer. Default
        is False.
    cut_at : numeric, optional
        Radius at which we 'cut' the 2D structure and set values to 0.
        For example, cut_at=sim.p.Rroche to set density 0 outside Roche radius.
        Default is None.
    skip_alt_range : tuple, optional
        Altitude range to skip for the whole 2D projection. Values within this
        range will be set to 0. Must be specified as a tuple (min_alt, max_alt).
        Default is None.
    skip_alt_range_dayside : tuple, optional
        Altitude range to skip for the dayside of the 2D projection. Values
        within this range will be set to 0. Must be specified as a tuple
        (min_alt, max_alt). Default is None.
    skip_alt_range_nightside : tuple, optional
        Altitude range to skip for the nightside of the 2D projection. Values
        within this range will be set to 0. Must be specified as a tuple
        (min_alt, max_alt). Default is None.

    Returns
    -------
    b_edges : array_like
        Impact parameters for 2D rays, the boundaries of the 'rays'.
    b_centers : array_like
        The actual positions of the rays and this is where the quantity is
        calculated at.
    x : array_like
        Total x grid with both negative and positive values (for day- and
        nightside).
    q2 : array_like
        Projected quantity onto the 2D grid.

    Raises
    ------
    AssertionError
        If arrays are not in order of ascending altitude or if skip_alt_range,
        skip_alt_range_dayside, or skip_alt_range_nightside are not specified
        correctly.
    """

    assert r1[1] > r1[0], "arrays must be in order of ascending altitude"

    b_edges = (
        np.logspace(np.log10(0.1 * Rp), np.log10(r1[-1] - 0.9 * Rp), num=numb)
        + 0.9 * Rp
    )  # impact parameters for 2D rays - these are the boundaries of the 'rays'
    b_centers = (
        b_edges[1:] + b_edges[:-1]
    ) / 2.0  # these are the actual positions of the rays and this is where the quantity is calculated at
    xhalf = (
        np.logspace(np.log10(0.101 * Rp), np.log10(r1[-1] + 0.1 * Rp), num=numb)
        - 0.1 * Rp
    )  # positive x grid
    x = np.concatenate(
        (-xhalf[::-1], xhalf)
    )  # total x grid with both negative and positive values (for day- and nightside)
    xx, bb = np.meshgrid(x, b_centers)
    rr = np.sqrt(bb**2 + xx**2)  # radii from planet core in 2D

    q2 = interp1d(r1, q1, fill_value=0.0, bounds_error=False)(rr)
    if x_projection:
        q2 = q2 * xx / rr  # now q2 is the projection in the x-direction

    if cut_at is not None:  # set values to zero outside the cut_at boundary
        q2[rr > cut_at] = 0.0

    # some options that were used in Linssen&Oklopcic (2023) to find where the line contribution comes from:
    if skip_alt_range is not None:
        assert skip_alt_range[0] < skip_alt_range[1]
        q2[(rr > skip_alt_range[0]) & (rr < skip_alt_range[1])] = 0.0
    if skip_alt_range_dayside is not None:
        assert skip_alt_range_dayside[0] < skip_alt_range_dayside[1]
        q2[
            (rr > skip_alt_range_dayside[0])
            & (rr < skip_alt_range_dayside[1])
            & (xx < 0.0)
        ] = 0.0
    if skip_alt_range_nightside is not None:
        assert skip_alt_range_nightside[0] < skip_alt_range_nightside[1]
        q2[
            (rr > skip_alt_range_nightside[0])
            & (rr < skip_alt_range_nightside[1])
            & (xx > 0.0)
        ] = 0.0

    return b_edges, b_centers, x, q2


def limbdark_quad(mu, ab):
    """
    Quadratic limb darkening law from Claret & Bloemen (2011).
    Returns I(mu)/I(1). mu is cos(theta) with theta the angle between
    the normal direction and beam direction. The following holds:
    mu = sqrt(1 - (r/Rs)^2)     with r/Rs the fractional distance to the
    center of the star (i.e. =0 at stellar disk center and =1 at limb).

    The quantities are all treated as 3D here internally, where:
    axis 0: the frequency axis
    axis 1: radial direction (rings) from planet core
    axis 2: angle phi within each radial ring

    Parameters
    ----------
    mu : array_like
        Cosine of the angle between the normal direction and beam direction.
    ab : array_like
        Coefficients of the quadratic limb darkening law, where `ab[:,0]`
        represents the linear coefficient and `ab[:,1]` represents the quadratic
        coefficient.

    Returns
    -------
    I : array_like
        Normalized intensity profile I(mu)/I(1) according to the quadratic
        limb darkening law.
    """

    a, b = ab[:, 0], ab[:, 1]
    I = (
        1
        - a[:, None, None] * (1 - mu[None, :, :])
        - b[:, None, None] * (1 - mu[None, :, :]) ** 2
    )

    return I


def avg_limbdark_quad(ab):
    """
    Average of the quadratic limb darkening I(mu) over the stellar disk.

    In the calculation of I, axis 0 is the frequency axis and axis 1 is the radial
    axis. The returned I_avg will then have only the frequency axis left.

    Parameters
    ----------
    ab : array_like
        Coefficients of the quadratic limb darkening law, where `ab[:,0]`
        represents the linear coefficient and `ab[:,1]` represents the quadratic
        coefficient.

    Returns
    -------
    I_avg : array_like
        Average intensity profile I_avg(mu) over the stellar disk according to
        the quadratic limb darkening law.
    """

    a, b = ab[:, 0], ab[:, 1]
    rf = np.linspace(0, 1, num=1000)  # sample the stellar disk in 1000 rings
    rfm = (rf[:-1] + rf[1:]) / 2  # midpoints
    mu = np.sqrt(1 - rfm**2)  # mu of each ring
    I = (
        1 - a[:, None] * (1 - mu[None, :]) - b[:, None] * (1 - mu[None, :]) ** 2
    )  # I of each ring
    projsurf = np.pi * (rf[1:] ** 2 - rf[:-1] ** 2)  # area of each ring

    I_avg = np.sum(I * projsurf, axis=1) / np.pi  # sum over the radial axis

    return I_avg


def calc_tau(x, ndens, Te, vx, nu, nu0, m, sig0, gamma, v_turb=0.0):
    """
    Calculates optical depth using Eq. 19 from Oklopcic&Hirata 2018.
    Does this at once for all rays, lines and frequencies. When doing
    multiple lines at once, they must all be from the same species and
    same level so that m and ndens are the same for the different lines.
    So you can do e.g. helium triplet or Ly-series at once. The FinFout()
    function does currently not make use of that (i.e. the helium triplet is
    calculated with three calls to this function).

    The quantities are all treated as 4D here internally, where:
    axis 0: the frequency axis
    axis 1: the different spectral lines
    axis 2: the different rays
    axis 3: the x (depth) direction along each ray

    Parameters
    ----------
    x : numpy.ndarray
        Depth values of the grid (1D array)
    ndens : numpy.ndarray
        Number density of the species (2D array with axes: (ray, depth)).
    Te : numpy.ndarray
        Temperature values (2D array with axes: (ray, depth)).
    vx : numpy.ndarray
        Line-of-sight velocity (2D array with axes: (ray, depth)).
    nu : numpy.ndarray
        Frequency values (1D array).
    nu0 : numeric or array-like
        Central frequencies of the different lines (1D array).
    m : numeric
        Mass of the chemical species in units of g.
    sig0 : numeric or array-like
        Cross-sections of the lines, Eq. 20 from Oklopcic&Hirata (2018) (1D array).
    gamma : numeric or array-like
        Half-width at half-maximum of the Lorentzian part of the line (1D array)
    v_turb : float, optional
        Root mean-squared of turbulent velocities in units of cm s-1. Turbulent
        motion will lead to extra spectral line broadening. By default 0.

    Returns
    -------
    tau : numpy.ndarray
        Optical depth values (2D array with axes: (frequency, ray)).
    """

    if not isinstance(nu0, np.ndarray):
        nu0 = np.array([nu0])
    if not isinstance(sig0, np.ndarray):
        sig0 = np.array([sig0])
    if not isinstance(gamma, np.ndarray):
        gamma = np.array([gamma])

    gaus_sigma = (
        np.sqrt(tools.k * Te[None, None, :] / m + 0.5 * v_turb**2)
        * nu0[None, :, None, None]
        / tools.c
    )
    # the following has a minus sign like in Eq. 21 of Oklopcic&Hirata (2018) because their formula is only correct if you take v_LOS from star->planet i.e. vx
    Delnu = (nu[:, None, None, None] - nu0[None, :, None, None]) - nu0[
        None, :, None, None
    ] / tools.c * vx[None, None, :]
    tau_cube = trapezoid(
        ndens[None, None, :]
        * sig0[None, :, None, None]
        * voigt_profile(Delnu, gaus_sigma, gamma[None, :, None, None]),
        x=x,
    )
    tau = np.sum(
        tau_cube, axis=1
    )  # sum up the contributions of the different lines -> now tau has axis 0:freq, axis 1:rayno

    return tau


def calc_cum_tau(x, ndens, Te, vx, nu, nu0, m, sig0, gamma, v_turb=0.0):
    """
    Calculates cumulative optical depth using Eq. 19 from Oklopcic&Hirata 2018,
    at one particular frequency. Does this at once for all rays and lines.
    When doing multiple lines at once, they must all be from the same species and
    same level so that m and ndens are the same for the different lines.
    So you can do e.g. helium triplet or Ly-series at once.

    The quantities are all treated as 3D here internally, where:
    axis 0: the different spectral lines
    axis 1: the different rays
    axis 2: the x (depth) direction along each ray

    Parameters
    ----------
    x : numpy.ndarray
        Depth values of the grid (1D array)
    ndens : numpy.ndarray
        Number density of the species (2D array with axes: (ray, depth)).
    Te : numpy.ndarray
        Temperature values (2D array with axes: (ray, depth)).
    vx : numpy.ndarray
        Line-of-sight velocity (2D array with axes: (ray, depth)).
    nu : numeric
        Frequency value.
    nu0 : numeric or array-like
        Central frequencies of the different lines (1D array).
    m : numeric
        Mass of the chemical species in units of g.
    sig0 : numeric or array-like
        Cross-sections of the lines, Eq. 20 from Oklopcic&Hirata (2018) (1D array).
    gamma : numeric or array-like
        Half-width at half-maximum of the Lorentzian part of the line (1D array)
    v_turb : float, optional
        Root mean-squared of turbulent velocities in units of cm s-1. Turbulent
        motion will lead to extra spectral line broadening. By default 0.

    Returns
    -------
    cum_tau : numpy.ndarray
        Cumulative (running sum) of the optical depth along the depth-direction.
    bin_tau : numpy.ndarray
        Optical depth contribution of each cell along the depth-direction.
    """

    if not isinstance(nu0, np.ndarray):
        nu0 = np.array([nu0])
    if not isinstance(sig0, np.ndarray):
        sig0 = np.array([sig0])
    if not isinstance(gamma, np.ndarray):
        gamma = np.array([gamma])

    gaus_sigma = (
        np.sqrt(tools.k * Te[None, None, :] / m + 0.5 * v_turb**2)
        * nu0[None, :, None, None]
        / tools.c
    )
    # the following has a minus sign like in Eq. 21 of Oklopcic&Hirata (2018) because their formula is only correct if you take v_LOS from star->planet i.e. vx
    Delnu = (nu - nu0[:, None, None]) - nu0[:, None, None] / tools.c * vx[None, :]
    integrand = (
        ndens[None, :]
        * sig0[:, None, None]
        * voigt_profile(Delnu, gaus_sigma, gamma[:, None, None])
    )
    bin_tau = np.zeros_like(integrand)
    bin_tau[:, :, 1:] = (
        (integrand[:, :, 1:] + np.roll(integrand, 1, axis=2)[:, :, 1:])
        / 2.0
        * np.diff(x)[None, None, :]
    )
    bin_tau = np.sum(
        bin_tau, axis=0
    )  # sum up contribution of different lines, now bin_tau has same shape as Te
    cum_tau = np.cumsum(bin_tau, axis=1)  # do cumulative sum over the x-direction

    return cum_tau, bin_tau


def tau_to_FinFout(b_edges, tau, Rs, bp=0.0, ab=np.zeros(2), a=0.0, phase=0.0):
    """
    Takes in optical depth values and calculates the Fin/Fout transit spectrum,
    using the stellar radius and optional limb darkening and transit phase
    parameters. If all set to 0 (default), uses planet at stellar disk center
    with no limb darkening.

    Parameters
    ----------
    b_edges : array-like
        Impact parameters of the rays through the planet atmosphere (1D array).
    tau : array-like
        Optical depth values (2D array with axes: (freq, ray))
    Rs : numeric
        Stellar radius in units of cm.
    bp : numeric, optional
        Transit impact parameter of the planet in units of stellar radius, by default 0.
    ab : array-like, optional
        Quadratic limb darkening parameters. Either a list/array of two values if the
        limb-darkening is wavelength-independent, or an array with shape (len(wavs),2)
        if the limb-darkening is wavelength-dependent. By default np.zeros(2).
    a : numeric, optional
        Planet orbital semi-major axis in units of cm, by default 0.
    phase : numeric, optional
        Planetary orbital phase defined as 0<phase<1 where 0 is mid-transit.
        The current implementation of phase does not take into account the
        tidally-locked rotation of the planet. So you'll always see the
        exact same projection (terminator) of the planet, just against
        a different limb-darkened stellar background. As long as the atmosphere is 1D
        symmetric, which we are assuming, this is exactly the same. But if in the
        future e.g. day-to-nightside winds are added on top, it will matter. By default 0.

    Returns
    -------
    FinFout : numpy.ndarray
        Transit spectrum in units of in-transit flux / out-of-transit flux (i.e., Fin/Fout).
    """

    if ab.ndim == 1:
        ab = ab[None, :]

    # add some impact parameters and tau=inf bins that make up the planet core:
    b_edges = np.concatenate(
        (np.linspace(0, b_edges[0], num=50, endpoint=False), b_edges)
    )
    b_centers = (
        b_edges[1:] + b_edges[:-1]
    ) / 2  # calculate bin centers with the added planet core rays included
    tau = np.concatenate((np.ones((np.shape(tau)[0], 50)) * np.inf, tau), axis=1)

    projsurf = np.pi * (
        b_edges[1:] ** 2 - b_edges[:-1] ** 2
    )  # ring surface of each ray (now has same length as b_centers)
    phis = np.linspace(
        0, 2 * np.pi, num=500, endpoint=False
    )  # divide rings into different angles phi
    # rc is the distance to stellar center. Axis 0: radial rings, axis 1: phi
    rc = np.sqrt(
        (bp * Rs + b_centers[:, None] * np.cos(phis[None, :])) ** 2
        + (b_centers[:, None] * np.sin(phis[None, :]) + a * np.sin(2 * np.pi * phase))
        ** 2
    )
    rc = ma.masked_where(
        rc > Rs, rc
    )  # will ensure I is masked (and later set to 0) outside stellar projected disk
    mu = np.sqrt(1 - (rc / Rs) ** 2)  # angle, see 'limbdark_quad' function
    I = limbdark_quad(mu, ab)
    Ir_avg = np.sum(I, axis=2) / len(phis)  # average I per ray
    Ir_avg = Ir_avg.filled(fill_value=0.0)  # convert back to regular numpy array
    Is_avg = avg_limbdark_quad(ab)  # average I of the full stellar disk

    FinFout = np.ones_like(tau[:, 0]) - np.sum(
        (
            (1 - np.exp(-tau))
            * Ir_avg
            * projsurf[None, :]
            / (Is_avg[:, None] * np.pi * Rs**2)
        ),
        axis=1,
    )

    return FinFout


def read_NIST_lines(species, wavlower=None, wavupper=None):
    """Reads a tabular file of spectral line coefficients from the NIST database.

    Parameters
    ----------
    species : str
        Atomic or ionic species, for example 'He' for atomic helium, or 'C+2' for doubly ionized carbon.
    wavlower : numeric, optional
        Lower boundary on the wavelengths to read in units of Å, by default None
    wavupper : numeric, optional
        Upper boundary on the wavelengths to read in units of Å, by default None

    Returns
    -------
    spNIST : pandas.DataFrame
        Line coefficients needed for radiative transfer calculations.
    """

    spNIST = pd.read_table(
        tools.sunbatherpath + "/RT_tables/" + species + "_lines_NIST.txt"
    )  # line info
    # remove lines with nan fik or Aik values. Note that lineno doesn't change (uses index instead of rowno.)
    spNIST = spNIST[spNIST.fik.notna()]
    spNIST = spNIST[spNIST["Aki(s^-1)"].notna()]
    if spNIST.empty:
        warnings.warn(f"No lines with necessary coefficients found for {species}")
        return spNIST
    if isinstance(spNIST["Ei(Ry)"].iloc[0], str):  # if there are no [](), the datatype will be float already
        spNIST["Ei(Ry)"] = (
            spNIST["Ei(Ry)"].str.extract(r"(\d+)", expand=False).astype(float)
        )  # remove non-numeric characters such as [] and ()
    spNIST["sig0"] = sigt0 * spNIST.fik
    spNIST["nu0"] = tools.c * 1e8 / (spNIST["ritz_wl_vac(A)"])  # speed of light to AA/s
    spNIST["lorgamma"] = spNIST["Aki(s^-1)"] / (
        4 * np.pi
    )  # lorentzian gamma is not function of depth or nu. Value in Hz

    if wavlower is not None:
        spNIST.drop(
            labels=spNIST.index[spNIST["ritz_wl_vac(A)"] <= wavlower], inplace=True
        )
    if wavupper is not None:
        spNIST.drop(
            labels=spNIST.index[spNIST["ritz_wl_vac(A)"] >= wavupper], inplace=True
        )

    return spNIST


def FinFout(
    sim,
    wavsAA,
    species,
    numrays=100,
    width_fac=1.0,
    ab=np.zeros(2),
    phase=0.0,
    phase_bulkshift=False,
    v_turb=0.0,
    cut_at=None,
):
    """
    Calculates a transit spectrum in units of in-transit flux / out-of-transit flux (i.e., Fin/Fout).
    Only spectral lines originating from provided species will be calculated.

    Parameters
    ----------
    sim : tools.Sim
        Cloudy simulation output of an upper atmosphere. Needs to have tools.Planet and
        tools.Parker class attributes.
    wavsAA : array-like
        Wavelengths to calculate transit spectrum on, in units of Å (1D array).
    species : str or array-like
        Chemical species to include in the calculations. Molecules are not supported.
        This argument distinguishes between atoms and ions, so for example 'Fe' will
        only calculate lines originating from atomic iron. To calculate lines from
        singly or doubly ionized iron, you must include 'Fe+' and 'Fe+2', respectively.
    numrays : int, optional
        Number of rays with different distance from the planet we project the 1D profiles onto.
        Higher number leads to slower computation times but higher accuracy, by default 100.
    width_fac : numeric, optional
        A multiplication factor for the 'max_voigt_width' variable within this function,
        which sets how far to either side of the rest-frame line centroid
        we calculate optical depths for every line.
        Standard value is 5 Gaussian standard deviations + 5 Lorentzian gammas.
        For very strong lines such as Ly-alpha, you may need a value >1 to properly calculate
        the far wings of the line. By default 1.
    ab : array-like, optional
        Quadratic limb darkening parameters. Either a list/array of two values if the
        limb-darkening is wavelength-independent, or an array with shape (len(wavs),2)
        if the limb-darkening is wavelength-dependent. By default np.zeros(2).
    phase : numeric, optional
        Planetary orbital phase defined as 0<phase<1 where 0 is mid-transit.
        The current implementation of phase does not take into account the
        tidally-locked rotation of the planet. So you'll always see the
        exact same projection (terminator) of the planet, just against
        a different limb-darkened stellar background. As long as the atmosphere is 1D
        symmetric, which we are assuming, this is exactly the same. But if in the
        future e.g. day-to-nightside winds are added on top, it will matter. By default 0.
    phase_bulkshift : bool, optional
        If phase != 0, the planet will have a nonzero bulk radial velocity in the stellar rest-frame.
        If this parameter is set to True, that velocity shift will be imposed on the transit spectrum as well.
        If this parameter is set to False, the spectral lines will still be at their rest-frame wavelengths. By default False.
    v_turb : float, optional
        Root mean-squared of turbulent velocities in units of cm s-1. Turbulent
        motion will lead to extra spectral line broadening. By default 0.
    cut_at : numeric, optional
        Radius at which we 'cut' the atmospheric profile and set values to 0.
        For example, use cut_at=sim.p.Rroche to set density 0 outside the Roche radius.
        Default is None (i.e., entire atmosphere included).

    Returns
    -------
    FinFout : numpy.ndarray
        Transit spectrum in units of in-transit flux / out-of-transit flux (i.e., Fin/Fout).
    found_lines : tuple
        Wavelengths and responsible species of the spectral lines included in this transit spectrum.
    notfound_lines : tuple
        Wavelengths and responsible species of spectral lines that are listed in the NIST database,
        but which could not be calculated due to their excitation state not being reported by Cloudy.
    """

    assert hasattr(sim, "p"), "The sim must have an attributed Planet object"
    assert (
        "v" in sim.ovr.columns
    ), "We need a velocity structure, such as that from adding a Parker object to the sim"
    assert hasattr(sim, "den"), (
        "The sim must have a .den file that stores the densities of the atomic/ionic excitation states. "
        "Please re-run your Cloudy simulation while saving these. Either re-run sunbather.convergeT_parker.py "
        "with the -save_sp flag, or use the tools.insertden_Cloudy_in() function with rerun=True."
    )

    ab = np.array(ab)  # turn possible list into array
    if ab.ndim == 1:
        ab = ab[None, :]  # add frequency axis
    assert (
        ab.ndim == 2
        and np.shape(ab)[1] == 2
        and (np.shape(ab)[0] == 1 or np.shape(ab)[0] == len(wavsAA))
    ), "Give ab as shape (1,2) or (2,) or (len(wavsAA),2)"

    Rs, Rp = sim.p.Rstar, sim.p.R
    nus = tools.c * 1e8 / wavsAA  # Hz, converted c to AA/s

    r1 = sim.ovr.alt.values[::-1]
    Te1 = sim.ovr.Te.values[::-1]
    v1 = sim.ovr.v.values[::-1]

    be, _, x, Te = project_1D_to_2D(r1, Te1, Rp, numb=numrays)
    be, _, x, vx = project_1D_to_2D(r1, v1, Rp, numb=numrays, x_projection=True)

    if phase_bulkshift:
        assert hasattr(
            sim.p, "Kp"
        ), "The Planet object does not have a Kp attribute, likely because either a, Mp or Mstar is unknown"
        vx = vx - sim.p.Kp * np.sin(
            phase * 2 * np.pi
        )  # negative sign because x is defined as positive towards the observer.

    state_ndens = {}
    tau = np.zeros((len(wavsAA), len(be) - 1))

    if isinstance(species, str):
        species = [species]

    found_lines = (
        []
    )  # will store nu0 of all lines that were used (might be nice to make it a dict per species in future!)
    notfound_lines = []  # will store nu0 of all lines that were not found

    for spec in species:
        if spec in sim.den.columns:
            warnings.warn(
                f"Your requested species {spec} is not resolved into multiple energy levels by Cloudy. "
                + f"I will make the spectrum assuming all {spec} is in the ground-state."
            )
        elif not any(spec + "[" in col for col in sim.den.columns):
            warnings.warn(
                f"Your requested species {spec} is not present in Cloudy's output, so the spectrum will be flat. "
                + "Please re-do your Cloudy simulation while saving this species. Either use the tools.insertden_Cloudy_in() "
                + "function, or run convergeT_parker.py again with the correct -save_sp arguments."
            )
            continue

        spNIST = read_NIST_lines(spec, wavlower=wavsAA[0], wavupper=wavsAA[-1])

        if len(species) == 1 and len(spNIST) == 0:
            warnings.warn(
                f"Your requested species {spec} does not have any lines in this wavelength range (according to the NIST database), "
                "so the spectrum will be flat."
            )

        for lineno in spNIST.index.values:  # loop over all lines in the spNIST table.
            gaus_sigma_max = (
                np.sqrt(
                    tools.k * np.nanmax(Te) / tools.get_mass(spec) + 0.5 * v_turb**2
                )
                * spNIST.nu0.loc[lineno]
                / tools.c
            )  # maximum stddev of Gaussian part
            max_voigt_width = (
                5 * (gaus_sigma_max + spNIST["lorgamma"].loc[lineno]) * width_fac
            )  # the max offset of Voigt components (=natural+thermal broad.)
            linenu_low = (1 + np.min(vx) / tools.c) * spNIST.nu0.loc[
                lineno
            ] - max_voigt_width
            linenu_hi = (1 + np.max(vx) / tools.c) * spNIST.nu0.loc[
                lineno
            ] + max_voigt_width

            nus_line = nus[
                (nus > linenu_low) & (nus < linenu_hi)
            ]  # the frequency values that make sense to calculate for this line
            if (
                nus_line.size == 0
            ):  # then this line is not in our wav range and we skip it
                continue  # to next spectral line

            # get all columns in .den file which energy corresponds to this Ei
            colname, lineweight = tools.find_line_lowerstate_in_en_df(
                spec, spNIST.loc[lineno], sim.en
            )
            if colname is None:  # we skip this line if the line energy is not found.
                notfound_lines.append(spNIST["ritz_wl_vac(A)"][lineno])
                continue  # to next spectral line

            found_lines.append(
                (spNIST["ritz_wl_vac(A)"].loc[lineno], colname)
            )  # if we got to here, we did find the spectral line

            if colname in state_ndens.keys():
                ndens = state_ndens[colname]
            else:
                ndens1 = sim.den[colname].values[::-1]
                be, _, x, ndens = project_1D_to_2D(
                    r1, ndens1, Rp, numb=numrays, cut_at=cut_at
                )
                state_ndens[colname] = ndens  # add to dictionary for future reference

            ndens_lw = (
                ndens * lineweight
            )  # important that we make this a new variable as otherwise state_ndens would change as well!

            tau_line = calc_tau(
                x,
                ndens_lw,
                Te,
                vx,
                nus_line,
                spNIST.nu0.loc[lineno],
                tools.get_mass(spec),
                spNIST.sig0.loc[lineno],
                spNIST["lorgamma"].loc[lineno],
                v_turb=v_turb,
            )
            tau[
                (nus > linenu_low) & (nus < linenu_hi), :
            ] += tau_line  # add the tau values to the correct nu bins

    FinFout = tau_to_FinFout(be, tau, Rs, bp=sim.p.bp, ab=ab, phase=phase, a=sim.p.a)

    return FinFout, found_lines, notfound_lines


def tau_1D(sim, wavAA, species, width_fac=1.0, v_turb=0.0):
    """
    Maps out the optical depth at one specific wavelength.
    The running integral of the optical deph is calculated at each depth of the ray.
    Useful for identifying where a spectral line forms.
    This function maps out the optical depth along the direction
    of the Cloudy simulation (i.e., the substellar ray). To do proper radiative
    transfer calculations, one needs to calculate the optical depth in a 2D-projected
    plane. The tau_12D() function can be used for that.

    Parameters
    ----------
    sim : tools.Sim
        Cloudy simulation output of an upper atmosphere. Needs to have tools.Planet and
        tools.Parker class attributes.
    wavAA : numeric
        Wavelength to calculate the optical depths at, in units of Å.
    species : str or array-like
        Chemical species to include in the calculations. Molecules are not supported.
        This argument distinguishes between atoms and ions, so for example 'Fe' will
        only calculate lines originating from atomic iron. To calculate lines from
        singly or doubly ionized iron, you must include 'Fe+' and 'Fe+2', respectively.
    width_fac : numeric, optional
        A multiplication factor for the 'max_voigt_width' variable within this function,
        which sets how far to either side of the rest-frame line centroid
        we calculate optical depths for every line.
        Standard value is 5 Gaussian standard deviations + 5 Lorentzian gammas.
        For very strong lines such as Ly-alpha, you may need a value >1 to properly calculate
        the far wings of the line. By default 1.
    v_turb : float, optional
        Root mean-squared of turbulent velocities in units of cm s-1. Turbulent
        motion will lead to extra spectral line broadening. By default 0.

    Returns
    -------
    tot_cum_tau : numpy.ndarray
        Cumulative (running sum) of the optical depth along the depth-direction (1D array).
    tot_bin_tau : numpy.ndarray
        Optical depth contribution of each cell along the depth-direction (1D array).
    found_lines : tuple
        Wavelengths and responsible species of the spectral lines included in this transit spectrum.
    notfound_lines : tuple
        Wavelengths and responsible species of spectral lines that are listed in the NIST database,
        but which could not be calculated due to their excitation state not being reported by Cloudy.
    """

    assert isinstance(wavAA, (float, int)), "Pass one wavelength in Å as a float or int"
    assert hasattr(sim, "p"), "The sim must have an attributed Planet object"
    assert (
        "v" in sim.ovr.columns
    ), "We need a velocity structure, such as that from adding a Parker object to the sim."

    Rs, Rp = sim.p.Rstar, sim.p.R
    nu = tools.c * 1e8 / wavAA  # Hz, converted c to AA/s

    d = sim.ovr.depth.values
    Te = sim.ovr.Te.values
    v = sim.ovr.v.values  # radial velocity
    vx = -v  # because we do the substellar ray which is towards the -x direction

    tot_cum_tau, tot_bin_tau = np.zeros_like(d), np.zeros_like(d)

    if isinstance(species, str):
        species = [species]

    found_lines = (
        []
    )  # will store nu0 of all lines that were used (might be nice to make it a dict per species in future!)
    notfound_lines = []  # will store nu0 of all lines that were not found

    for spec in species:
        spNIST = read_NIST_lines(spec)

        for lineno in spNIST.index.values:  # loop over all lines in the spNIST table.
            gaus_sigma_max = (
                np.sqrt(
                    tools.k * np.nanmax(Te) / tools.get_mass(spec) + 0.5 * v_turb**2
                )
                * spNIST.nu0.loc[lineno]
                / tools.c
            )  # maximum stddev of Gaussian part
            max_voigt_width = (
                5 * (gaus_sigma_max + spNIST["lorgamma"].loc[lineno]) * width_fac
            )  # the max offset of Voigt components (=natural+thermal broad.)
            linenu_low = (1 + np.min(vx) / tools.c) * spNIST.nu0.loc[
                lineno
            ] - max_voigt_width
            linenu_hi = (1 + np.max(vx) / tools.c) * spNIST.nu0.loc[
                lineno
            ] + max_voigt_width

            if (nu < linenu_low) | (
                nu > linenu_hi
            ):  # then this line does not probe our requested wav and we skip it
                continue  # to next spectral line

            # get all columns in .den file which energy corresponds to this Ei
            colname, lineweight = tools.find_line_lowerstate_in_en_df(
                spec, spNIST.loc[lineno], sim.en
            )
            if colname is None:  # we skip this line if the line energy is not found.
                notfound_lines.append(spNIST["ritz_wl_vac(A)"][lineno])
                continue  # to next spectral line

            found_lines.append(
                (spNIST["ritz_wl_vac(A)"].loc[lineno], colname)
            )  # if we got to here, we did find the spectral line

            ndens = (
                sim.den[colname].values * lineweight
            )  # see explanation in FinFout_2D function

            cum_tau, bin_tau = calc_cum_tau(
                d,
                ndens,
                Te,
                vx,
                nu,
                spNIST.nu0.loc[lineno],
                tools.get_mass(spec),
                spNIST.sig0.loc[lineno],
                spNIST["lorgamma"].loc[lineno],
                v_turb=v_turb,
            )
            tot_cum_tau += cum_tau[
                0
            ]  # add the tau values to the total (of all species & lines together)
            tot_bin_tau += bin_tau[0]

    return tot_cum_tau, tot_bin_tau, found_lines, notfound_lines


def tau_12D(sim, wavAA, species, width_fac=1.0, v_turb=0.0, cut_at=None):
    """
    Maps out the optical depth at one specific wavelength.
    The running integral of the optical deph is calculated at each stellar light ray
    with different impact parameter from the planet, and at each depth into those rays.
    Useful for identifying where a spectral line forms.

    Parameters
    ----------
    sim : tools.Sim
        Cloudy simulation output of an upper atmosphere. Needs to have tools.Planet and
        tools.Parker class attributes.
    wavAA : numeric
        Wavelength to calculate the optical depths at, in units of Å.
    species : str or array-like
        Chemical species to include in the calculations. Molecules are not supported.
        This argument distinguishes between atoms and ions, so for example 'Fe' will
        only calculate lines originating from atomic iron. To calculate lines from
        singly or doubly ionized iron, you must include 'Fe+' and 'Fe+2', respectively.
    width_fac : numeric, optional
        A multiplication factor for the 'max_voigt_width' variable within this function,
        which sets how far to either side of the rest-frame line centroid
        we calculate optical depths for every line.
        Standard value is 5 Gaussian standard deviations + 5 Lorentzian gammas.
        For very strong lines such as Ly-alpha, you may need a value >1 to properly calculate
        the far wings of the line. By default 1.
    v_turb : float, optional
        Root mean-squared of turbulent velocities in units of cm s-1. Turbulent
        motion will lead to extra spectral line broadening. By default 0.
    cut_at : numeric, optional
        Radius at which we 'cut' the atmospheric profile and set values to 0.
        For example, use cut_at=sim.p.Rroche to set density 0 outside the Roche radius.
        Default is None (i.e., entire atmosphere included).

    Returns
    -------
    tot_cum_tau : numpy.ndarray
        Cumulative (running sum) of the optical depth along the depth-direction (2D array with axes: (ray, depth)).
    tot_bin_tau : numpy.ndarray
        Optical depth contribution of each cell along the depth-direction (2D array with axes: (ray, depth)).
    found_lines : tuple
        Wavelengths and responsible species of the spectral lines included in this transit spectrum.
    notfound_lines : tuple
        Wavelengths and responsible species of spectral lines that are listed in the NIST database,
        but which could not be calculated due to their excitation state not being reported by Cloudy.
    """

    assert isinstance(wavAA, (float, int)), "Pass one wavelength in Å as a float or int"
    assert hasattr(sim, "p")
    assert (
        "v" in sim.ovr.columns
    ), "We need a velocity structure, such as that from adding a Parker object to the sim."

    nu = tools.c * 1e8 / wavAA  # Hz, converted c to AA/s

    be, bc, x, Te = project_1D_to_2D(
        sim.ovr.alt.values[::-1], sim.ovr.Te.values[::-1], sim.p.R
    )
    be, bc, x, vx = project_1D_to_2D(
        sim.ovr.alt.values[::-1], sim.ovr.v.values[::-1], sim.p.R, x_projection=True
    )

    tot_cum_tau, tot_bin_tau = np.zeros_like(vx), np.zeros_like(vx)

    if isinstance(species, str):
        species = [species]

    found_lines = (
        []
    )  # will store nu0 of all lines that were used (might be nice to make it a dict per species in future!)
    notfound_lines = []  # will store nu0 of all lines that were not found

    for spec in species:
        spNIST = read_NIST_lines(spec)

        for lineno in spNIST.index.values:  # loop over all lines in the spNIST table.
            gaus_sigma_max = (
                np.sqrt(
                    tools.k * np.nanmax(Te) / tools.get_mass(spec) + 0.5 * v_turb**2
                )
                * spNIST.nu0.loc[lineno]
                / tools.c
            )  # maximum stddev of Gaussian part
            max_voigt_width = (
                5 * (gaus_sigma_max + spNIST["lorgamma"].loc[lineno]) * width_fac
            )  # the max offset of Voigt components (=natural+thermal broad.)
            linenu_low = (1 + np.min(vx) / tools.c) * spNIST.nu0.loc[
                lineno
            ] - max_voigt_width
            linenu_hi = (1 + np.max(vx) / tools.c) * spNIST.nu0.loc[
                lineno
            ] + max_voigt_width

            if (nu < linenu_low) | (
                nu > linenu_hi
            ):  # then this line does not probe our requested wav and we skip it
                continue  # to next spectral line

            # get all columns in .den file which energy corresponds to this Ei
            colname, lineweight = tools.find_line_lowerstate_in_en_df(
                spec, spNIST.loc[lineno], sim.en
            )
            if colname is None:  # we skip this line if the line energy is not found.
                notfound_lines.append(spNIST["ritz_wl_vac(A)"][lineno])
                continue  # to next spectral line

            found_lines.append(
                (spNIST["ritz_wl_vac(A)"].loc[lineno], colname)
            )  # if we got to here, we did find the spectral line

            # multiply with the lineweight! Such that for unresolved J, a line originating from J=1/2 does not also get density of J=3/2 state
            _, _, _, ndens = project_1D_to_2D(
                sim.ovr.alt.values[::-1],
                sim.den[colname].values[::-1],
                sim.p.R,
                cut_at=cut_at,
            )
            ndens *= lineweight

            cum_tau, bin_tau = calc_cum_tau(
                x,
                ndens,
                Te,
                vx,
                nu,
                spNIST.nu0.loc[lineno],
                tools.get_mass(spec),
                spNIST.sig0.loc[lineno],
                spNIST["lorgamma"].loc[lineno],
                v_turb=v_turb,
            )
            tot_cum_tau += cum_tau  # add the tau values to the total (of all species & lines together)
            tot_bin_tau += bin_tau

    return tot_cum_tau, tot_bin_tau, found_lines, notfound_lines


def FinFout2RpRs(FinFout):
    """
    Converts the Fin/Fout (i.e., flux in-transit / flux out-of-transit) to
    Rp/Rs (i.e., the apparent size of the planet relative to the star).
    The continuum should be roughly Rp/Rs, but not exactly with limb-darkening.
    The reverse function of this is RpRs2FinFout().

    Parameters
    ----------
    FinFout : array-like
        In-transit / out-transit flux values, for example as returned by FinFout().

    Returns
    -------
    RpRs : array-like
        Transit spectrum in units of planet size / star size.
    """

    RpRs = np.sqrt(1 - FinFout)

    return RpRs


def RpRs2FinFout(RpRs):
    """
    Converts the Fin/Fout (i.e., flux in-transit / flux out-of-transit) to
    Rp/Rs (i.e., the apparent size of the planet relative to the star).
    The continuum should be roughly Rp/Rs, but not exactly with limb-darkening.
    The reverse function of this is FinFout2RpRs().

    Parameters
    ----------
    RpRs : array-like
        Transit spectrum in units of planet size / star size

    Returns
    -------
    FinFout : array-like
        In-transit / out-transit flux values
    """

    FinFout = 1 - RpRs**2

    return FinFout


def vac2air(wavs_vac):
    """
    Converts vacuum wavelengths to air. Wavelengths MUST be in Angstroms.
    From: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    The reverse function of this is vac2air().

    Parameters
    ----------
    wavs_vac : numeric or array-like
        Wavelength(s) in vacuum in units of Å.

    Returns
    -------
    wavs_air : numeric or array-like
        Wavelength(s) in air in units of Å.
    """

    s = 1e4 / wavs_vac
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    wavs_air = wavs_vac / n

    return wavs_air


def air2vac(wavs_air):
    """
    Converts air wavelengths to vacuum. Wavelengths MUST be in Angstroms.
    From: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    The reverse function of this is air2vac().

    Parameters
    ----------
    wavs_air : numeric or array-like
        Wavelength(s) in air in units of Å.

    Returns
    -------
    wavs_vac : numeric or array-like
        Wavelength(s) in vacuum in units of Å.
    """

    s = 1e4 / wavs_air
    n = (
        1
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s**2)
        + 0.0001599740894897 / (38.92568793293 - s**2)
    )
    wavs_vac = wavs_air * n

    return wavs_vac


def constantR_wavs(wav_lower, wav_upper, R):
    """
    Returns an array of wavelengths at a constant spectral resolution.

    Parameters
    ----------
    wav_lower : numeric
        Lower bound of the wavelength array.
    wav_upper : numeric
        Upper bound of the wavelength array.
    R : numeric
        Resolving power.

    Returns
    -------
    wavs : numpy.ndarray
        Wavelength array.
    """

    wav = wav_lower
    wavs = []
    while wav < wav_upper:
        wavs.append(wav)
        wav += wav / R
    wavs = np.array(wavs)

    return wavs


def convolve_spectrum_R(wavs, flux, R, verbose=False):
    """
    Convolves a spectrum with a Gaussian filter down to a lower spectral resolution.
    This function uses a constant gaussian width that is calculated from the middle wavelength point.
    This means that it only works properly when the wavs array spans a relatively small bandwidth.
    Since R = delta-lambda / lambda, if the bandwidth is too large, the assumption made here that
    delta-lambda is the same over the whole array will not be valid.

    Parameters
    ----------
    wavs : array-like
        Wavelengths.
    flux : array-like
        Flux values.
    R : numeric
        Resolving power.
    verbose : bool, optional
        Whether to print some diagnostics, by default False

    Returns
    -------
    convolved_spectrum : numpy.ndarray
        The convolved spectrum at resolution R.
    """

    assert wavs[1] > wavs[0], "Wavelengths must be in ascending order"
    assert np.allclose(
        np.diff(wavs), np.diff(wavs)[0], atol=0.0, rtol=1e-5
    ), "Wavelengths must be equidistant"
    if wavs[-1] / wavs[0] > 1.05:
        warnings.warn(
            "The wavelengths change by more than 5 percent in your array. Converting R into a constant delta-lambda becomes questionable."
        )

    delta_lambda = (
        wavs[int(len(wavs) / 2)] / R
    )  # width of the filter in wavelength - use middle wav point
    FWHM = delta_lambda / np.diff(wavs)[0]  # width of the filter in pixels
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))  # std dev. of the gaussian in pixels

    if verbose:
        print(
            f"R={R}, lamb={wavs[0]}, delta-lamb={delta_lambda}, FWHM={FWHM} pix, sigma={sigma} pix"
        )

    convolved_spectrum = gaussian_filter1d(flux, sigma)

    return convolved_spectrum
