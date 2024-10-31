#sunbather imports
import tools

#other imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import scipy.stats as sps
import os
import warnings


def calc_expansion(r, rho, v, Te, mu):
    """
    Calculates expansion cooling (Linssen et al. 2024 Eq. 3 second term).

    Parameters
    ----------
    r : numpy.ndarray
        Radius in units of cm
    rho : numpy.ndarray
        Density in units of g cm-3
    v : numpy.ndarray
        Velocity in units of cm s-1
    Te : numpy.ndarray
        Temperature in units of K
    mu : numpy.ndarray
        Mean particle mass in units of amu

    Returns
    -------
    expansion : numpy.ndarray
        Expansion cooling rate.
    """

    expansion = tools.k/tools.mH * Te * v / mu * np.gradient(rho, r)
    assert np.max(expansion) <= 0, "Found positive expansion cooling rates (i.e., heating)."

    return expansion 


def calc_advection(r, rho, v, Te, mu):
    """
    Calcules advection heating/cooling (Linssen et al. 2024 Eq. 3 first term).

    Parameters
    ----------
    r : numpy.ndarray
        Radius in units of cm
    rho : numpy.ndarray
        Density in units of g cm-3
    v : numpy.ndarray
        Velocity in units of cm s-1
    Te : numpy.ndarray
        Temperature in units of K
    mu : numpy.ndarray
        Mean particle mass in units of amu

    Returns
    -------
    advection : numpy.ndarray
        Advection heating/cooling rate.
    """

    advection = -1 * tools.k/(tools.mH * 2/3) * rho * v * np.gradient(Te/mu, r)

    return advection


def simtogrid(sim, grid):
    """
    Extracts various needed quantities from a Cloudy simulation and interpolates
    them onto the provided radius grid.

    Parameters
    ----------
    sim : tools.Sim
        Cloudy simulation.
    grid : numpy.ndarray
        Radius grid in units of cm.

    Returns
    -------
    Te : numpy.ndarray
        Temperature in units of K.
    mu : numpy.ndarray
        Mean particle mass in units of amu.
    rho : numpy.ndarray
        Density in units of g cm-3.
    v : numpy.ndarray
        Velocity in units of cm s-1.
    radheat : numpy.ndarray
        Radiative heating rate in units of erg s-1 cm-3.
    radcool : numpy.ndarray
        Radiative cooling rate in units of erg s-1 cm-3, as positive values.
    expcool : numpy.ndarray
        Expansion cooling rate in units of erg s-1 cm-3, as positive values.
    advheat : numpy.ndarray
        Advection heating rate in units of erg s-1 cm-3.
    advcool : numpy.ndarray
        Advection cooling rate in units of erg s-1 cm-3, as positive values.
    """

    #get Cloudy quantities
    Te = interp1d(sim.ovr.alt, sim.ovr.Te, fill_value='extrapolate')(grid)
    mu = interp1d(sim.ovr.alt[sim.ovr.alt < 0.999 * sim.altmax * sim.p.R], sim.ovr.mu[sim.ovr.alt < 0.999 * sim.altmax * sim.p.R], fill_value='extrapolate')(grid)
    radheat = interp1d(sim.ovr.alt, sim.cool.htot, fill_value='extrapolate')(grid)
    radcool = interp1d(sim.ovr.alt, sim.cool.ctot, fill_value='extrapolate')(grid)

    #get isothermal Parker wind quantities
    rho = interp1d(sim.par.prof.alt, sim.par.prof.rho, fill_value='extrapolate')(grid)
    v = interp1d(sim.par.prof.alt, sim.par.prof.v, fill_value='extrapolate')(grid)

    #calculate bulk terms
    expcool = -1 * calc_expansion(grid, rho, v, Te, mu) #minus sign to get expansion cooling rates as positive values
    adv = calc_advection(grid, rho, v, Te, mu)

    #apply very slight smoothing because the Cloudy .ovr quantities have mediocre reported numerical precision    
    expcool = tools.smooth_gaus_savgol(expcool, fraction=0.01)
    adv = tools.smooth_gaus_savgol(adv, fraction=0.01)

    advheat, advcool = np.copy(adv), -1 * np.copy(adv)
    advheat[advheat < 0] = 0.
    advcool[advcool < 0] = 0.

    return Te, mu, rho, v, radheat, radcool, expcool, advheat, advcool


def calc_HCratio(radheat, radcool, expcool, advheat, advcool):
    """
    Calculates the ratio of total heating to total cooling.

    Parameters
    ----------
    radheat : numpy.ndarray
        Radiative heating rate in units of erg s-1 cm-3.
    radcool : numpy.ndarray
        Radiative cooling rate in units of erg s-1 cm-3, as positive values.
    expcool : numpy.ndarray
        Expansion cooling rate in units of erg s-1 cm-3, as positive values.
    advheat : numpy.ndarray
        Advection heating rate in units of erg s-1 cm-3.
    advcool : numpy.ndarray
        Advection cooling rate in units of erg s-1 cm-3, as positive values.

    Returns
    -------
    HCratio : numpy.ndarray
        Total heating rate H divided by total cooling rate C when H > C,
        or -C / H when C > H. The absolute value of HCratio is always >=1,
        and the sign indicates whether heating or cooling is stronger.
    """

    totheat = radheat + advheat
    totcool = radcool + expcool + advcool #all cooling rates are positive values
    nettotal = (totheat - totcool)

    HCratio = np.sign(nettotal) * np.maximum(totheat, totcool) / np.minimum(totheat,totcool)

    return HCratio


def get_new_Tstruc(old_Te, HCratio, fac):
    """
    Returns a new temperature profile based on a previous non-converged
    temperature profile and the associated heating/cooling imbalance.

    Parameters
    ----------
    old_Te : numpy.ndarray
        Previous temperature profile in units of K.
    HCratio : numpy.ndarray
        Heating/cooling imbalance, output of the calc_HCratio() function.
    fac : numpy.ndarray
        Scaling factor that sets how large the temperature adjustment is.

    Returns
    -------
    newTe : numpy.ndarray
        New temperature profile.
    """

    deltaT = fac * np.sign(HCratio) * np.log10(np.abs(HCratio)) #take log-based approach to deltaT
    fT = np.copy(deltaT) #the temperature multiplication fraction
    fT[deltaT < 0] = 1 + deltaT[deltaT < 0]
    fT[deltaT > 0] = 1/(1 - deltaT[deltaT > 0])
    fT = np.clip(fT, 0.5, 2) #max change is a factor 2 up or down in temperature
    newTe = old_Te * fT
    newTe = np.clip(newTe, 1e1, 1e6) #set minimum temperature to 10K

    return newTe


def calc_cloc(radheat, radcool, expcool, advheat, advcool, HCratio):
    """
    Checks if there is a point in the atmosphere where we can use
    the construction algorithm. It searches for two criteria:
    1. If there is a point from where on advection heating is stronger than
    radiative heating, and the temperature profile is reasonably converged.
    2. If there is a point from where on radiative cooling is weak
    compared to expansion and advection cooling.

    Parameters
    ----------
    radheat : numpy.ndarray
        Radiative heating rate in units of erg s-1 cm-3.
    radcool : numpy.ndarray
        Radiative cooling rate in units of erg s-1 cm-3, as positive values.
    expcool : numpy.ndarray
        Expansion cooling rate in units of erg s-1 cm-3, as positive values.
    advheat : numpy.ndarray
        Advection heating rate in units of erg s-1 cm-3.
    advcool : numpy.ndarray
        Advection cooling rate in units of erg s-1 cm-3, as positive values.
    HCratio : numpy.ndarray
        Heating/cooling imbalance, output of the calc_HCratio() function.

    Returns
    -------
    cloc : int
        Index of the grid from where to start the construction algorithm.
    """

    def first_true_index(arr):
        """
        Return the index of the first True value in the array.
        If there are no True in the array, returns 0
        """
        return np.argmax(arr)

    def last_true_index(arr):
        """
        Return the index of the last True value in the array.
        If there are no True in the array, returns len(arr)-1
        """
        return len(arr) - np.argmax(arr[::-1]) - 1

    def last_false_index(arr):
        """
        Return the index of the last False value in the array.
        If there are no False in the array, returns len(arr)-1
        """
        return len(arr) - np.argmax(~arr[::-1]) - 1

    #check for advection dominated regime
    adv_cloc = len(HCratio) #start by setting a 'too high' value
    advheat_dominates = (advheat > radheat) #boolean array where advection heating dominates
    bothrad_dominate = ((radheat > advheat) & (radcool > advcool) & (radcool > expcool)) #boolean array where radiative heating dominates AND radiative cooling dominates
    highest_r_above_which_no_bothrad_dominate = last_true_index(bothrad_dominate)
    advheat_dominates[:highest_r_above_which_no_bothrad_dominate] = False #now the boolean array stores where advection heating dominates AND where there is no point at higher altitudes that is rad. heat and rad. cool dominated
    if True in advheat_dominates: #if there is no such point, adv_cloc stays default value
        advdomloc = first_true_index(advheat_dominates) #get lowest altitude location where advection dominates
        advheat_unimportant = (advheat < 0.25 * radheat) #boolean array where advection heating is relatively unimportant
        advunimploc = last_true_index(advheat_unimportant[:advdomloc]) #first point at lower altitude where advection becomes unimportant (if no point exists, it will become advdomloc)
        #then walk to higher altitude again to find converged point. We are more lax with H/C ratio if advection dominates more.
        almost_converged = (np.abs(HCratio[advunimploc:]) < 1.3 * np.clip((advheat[advunimploc:] / radheat[advunimploc:])**(2./3.), 1, 10))
        if True in almost_converged: #otherwise it stays default value
            adv_cloc = advunimploc + first_true_index(almost_converged)

    #check for regime where radiative cooling is weak. Usually this means that expansion cooling dominates, but advection cooling can contribute in some cases
    expcool_dominates = (radcool / (radcool+expcool+advcool) < 0.2) # boolean array storing whether radiation cooling is weak
    if False not in expcool_dominates: # if they are all True
        exp_cloc = 0
    else: # if they are not all True
        exp_cloc = last_false_index(expcool_dominates) #this way of evaluating it guarantees that all entries after this one are True

    cloc = min(adv_cloc, exp_cloc) #use the lowest radius point

    return cloc


def relaxTstruc(grid, path, itno, Te, HCratio):
    """
    Proposes a new temperature profile using a 'relaxation' algorithm.

    Parameters
    ----------
    grid : numpy.ndarray
        Radius grid in units of cm.
    path : str
        Full path to the folder where the simulations are saved and ran.
    itno : int
        Iteration number.
    Te : numpy.ndarray
        Temperature profile of the last iteration at the 'grid' radii, in units of K.
    HCratio : numpy.ndarray
        Heating/cooling imbalance of the temperature profile of the last iteration,
        output of the calc_HCratio() function.

    Returns
    -------
    newTe_relax : numpy.ndarray
        Adjusted temperature profile to use for the next iteration.
    """

    if itno == 2: #save for first time
        np.savetxt(path+'iterations.txt', np.column_stack((grid, np.repeat(0.3, len(grid)), Te)),
                    header='grid fac1 Te1', comments='', delimiter=' ', fmt='%.7e')

    iterations_file = pd.read_csv(path+'iterations.txt', header=0, sep=' ')
    fac = iterations_file['fac'+str(itno-1)].values

    newTe_relax = get_new_Tstruc(Te, HCratio, fac) #adjust the temperature profile
    newTe_relax = tools.smooth_gaus_savgol(newTe_relax, fraction = 1./(20*itno)) #smooth it
    newTe_relax = np.clip(newTe_relax, 1e1, 1e6) #smoothing may have pushed newTe_relax < 10K again.

    if itno >= 4: #check for fluctuations. If so, we decrease the deltaT factor
        prev_prevTe = iterations_file['Te'+str(itno-2)]
        previous_ratio = Te / prev_prevTe #compare itno-2 to itno-1
        this_ratio = newTe_relax / Te #compare itno-1 to the current itno (because of smoothing this ratio is not exactly the same as fT)
        fl = (((previous_ratio < 1) & (this_ratio > 1)) | ((previous_ratio > 1) & (this_ratio < 1))) #boolean indicating where temperature fluctuates
        fac[fl] = 2/3 * fac[fl] #take smaller changes in T in regions where the temperature fluctuates
        fac = np.clip(tools.smooth_gaus_savgol(fac, size=10), 0.02, 0.3) #smooth the factor itself as well
        newTe_relax = get_new_Tstruc(Te, HCratio, fac) #recalculate new temperature profile with updated fac
        newTe_relax = tools.smooth_gaus_savgol(newTe_relax, fraction = 1/(20*itno)) #smooth it
        newTe_relax = np.clip(newTe_relax, 1e1, 1e6)

    iterations_file['fac'+str(itno)] = fac
    iterations_file.to_csv(path+'iterations.txt', sep=' ', float_format='%.7e', index=False)

    return newTe_relax


def constructTstruc(grid, newTe_relax, cloc, v, rho, mu, radheat, radcool):
    """
    Proposes a new temperature profile based on a 'construction' algorithm, 
    starting at the cloc and at higher altitudes.

    Parameters
    ----------
    grid : numpy.ndarray
        Radius grid in units of cm.
    newTe_relax : numpy.ndarray
        Newly proposed temperature profile from the relaxation algorithm.
    cloc : int
        Index of the grid from where to start the construction algorithm.
    v : numpy.ndarray
        Velocity in units of cm s-1 at the 'grid' radii.
    rho : numpy.ndarray
        Density in units of g cm-3 at the 'grid' radii.
    mu : numpy.ndarray
        Mean particle mass in units of amu at the 'grid' radii.
    radheat : numpy.ndarray
        Radiative heating rate in units of erg s-1 cm-3, at the 'grid' radii.
    radcool : numpy.ndarray
        Radiative cooling rate in units of erg s-1 cm-3, as positive values, at the 'grid' radii.

    Returns
    -------
    newTe_construct : numpy.ndarray
        Adjusted temperature profile to use for the next iteration.
    """

    newTe_construct = np.copy(newTe_relax) #start with the temp struc from the relaxation function

    expansion_Tdivmu = tools.k/tools.mH * v * np.gradient(rho, grid) #this is expansion except for the T/mu term (still negative values)
    advection_gradTdivmu = -1 * tools.k/(tools.mH * 2/3) * rho * v #this is advection except for the d(T/mu)/dr term

    def one_cell_HCratio(T, index):
        expcool = expansion_Tdivmu[index] * T / mu[index]
        adv = advection_gradTdivmu[index] * ((T/mu[index]) - (newTe_construct[index-1]/mu[index-1]))/(grid[index] - grid[index-1])

        #instead of completely keeping the radiative heating and cooling rate the same while we are solving for T in this bin,
        #we adjust it a little bit. This helps to prevent that the temperature changes are too drastic and go into a regime where
        #radiation becomes important again. We guess a quadratic dependence of the rates on T. This is not the true dependence,
        #but it does reduce to the original rate when T -> original T, which is important.
        guess_radheat = radheat[index] * (newTe_construct[index] / T)**2
        guess_radcool = radcool[index] * (T / newTe_construct[index])**2

        totheat = guess_radheat + max(adv, 0) #if adv is negative we don't add it here
        totcool = guess_radcool - expcool - min(adv, 0) #if adv is positive we don't add it here, we subtract expcool and adv because they are negative

        HCratio = max(totheat, totcool) / min(totheat, totcool) #both entities are positive

        return HCratio - 1 #find root of this value to get H/C close to 1


    for i in range(cloc+1, len(grid)): #walk from cloc to higher altitudes
        result = minimize_scalar(one_cell_HCratio, method='bounded', bounds=[1e1,1e6], args=(i))
        newTe_construct[i] = result.x


    #smooth around the abrupt edge where the constructed part sets in
    smooth_newTe_construct = tools.smooth_gaus_savgol(newTe_construct, fraction=0.03) #first smooth the complete T(r) profile
    smooth_newTe_construct = np.clip(smooth_newTe_construct, 1e1, 1e6) #after smoothing we might have ended up below 10K
    #now combine the smoothed profile around 'cloc', and the non-smoothed version away from 'cloc'
    smooth_weight = np.zeros(len(grid))
    smooth_weight += sps.norm.pdf(range(len(grid)), cloc, int(len(grid)/30))
    smooth_weight /= np.max(smooth_weight) #normalize
    raw_weight = 1 - smooth_weight
    newTe_construct = smooth_newTe_construct * smooth_weight + newTe_construct * raw_weight

    return newTe_construct


def make_rates_plot(altgrid, Te, newTe_relax, radheat, radcool, expcool, advheat, advcool, rho, HCratio, altmax, fc, 
                    newTe_construct=None, cloc=None, title=None, savename=None):
    """
    Makes a plot of the previous and newly proposed temperature profiles,
    as well as the different heating/cooling rates and their ratio based on the
    previous temperature profile.

    Parameters
    ----------
    altgrid : numpy.ndarray
        Radius grid in units of Rp.
    Te : numpy.ndarray
        Temperature profile of the last iteration in units of K.
    newTe_relax : numpy.ndarray
        Proposed temperature profile based on the relaxation algorithm.
    radheat : numpy.ndarray
        Radiative heating rate in units of erg s-1 cm-3.
    radcool : numpy.ndarray
        Radiative cooling rate in units of erg s-1 cm-3, as positive values.
    expcool : numpy.ndarray
        Expansion cooling rate in units of erg s-1 cm-3, as positive values.
    advheat : numpy.ndarray
        Advection heating rate in units of erg s-1 cm-3.
    advcool : numpy.ndarray
        Advection cooling rate in units of erg s-1 cm-3, as positive values.
    rho : numpy.ndarray
        Density in units of g cm-3
    HCratio : numpy.ndarray
        Heating/cooling imbalance, output of the calc_HCratio() function.
    altmax : numeric
        Maximum altitude of the simulation in units of planet radius.
    fc : numeric
        Convergence threshold for H/C.
    newTe_construct : numpy.ndarray, optional
        Proposed temperature profile based on the construction algorithm, by default None
    cloc : int, optional
        Index of the grid from where the construction algorithm was ran, by default None
    title : str, optional
        Title of the figure, by default None
    savename : str, optional
        Full path + filename to save the figure to, by default None
    """

    HCratiopos, HCrationeg = np.copy(HCratio), -1 * np.copy(HCratio)
    HCratiopos[HCratiopos < 0] = 0.
    HCrationeg[HCrationeg < 0] = 0.

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(4,7))
    if title != None:
        ax1.set_title(title)
    ax1.plot(altgrid, Te, color='#4CAF50', label='previous')
    ax1.plot(altgrid, newTe_relax, color='#FFA500', label='relaxation')
    if newTe_construct is not None:
        ax1.plot(altgrid, newTe_construct, color='#800080', label='construction')
        ax1.scatter(altgrid[cloc], newTe_relax[cloc], color='#800080')
    ax1.set_ylabel('Temperature [K]')
    ax1.legend(loc='best', fontsize=8)

    ax2.plot(altgrid, radheat/rho, color='red', linewidth=2.)
    ax2.plot(altgrid, radcool/rho, color='blue')
    ax2.plot(altgrid, expcool/rho, color='blue', linestyle='dashed')
    ax2.plot(altgrid, advheat/rho, color='red', linestyle='dotted')
    ax2.plot(altgrid, advcool/rho, color='blue', linestyle='dotted')
    ax2.set_yscale('log')
    ax2.set_ylim(0.1*min(min(radheat/rho), min(radcool/rho)), 2*max(max(radheat/rho), max(radcool/rho), max(expcool/rho), max(advheat/rho), max(advcool/rho)))
    ax2.set_ylabel('Rate [erg/s/g]')
    ax2.legend(((Line2D([], [], color='red', linestyle=(0,(6,6))), Line2D([], [], color='blue', linestyle=(6,(6,6)))),
                 Line2D([], [], color='blue', linestyle='dashed'),
                (Line2D([], [], color='red', linestyle=(0,(1,2,1,8))), Line2D([], [], color='blue', linestyle=(6,(1,2,1,8))))),
                ('radiation', 'expansion', 'advection'), loc='best', fontsize=8)

    ax3.plot(altgrid, HCratiopos, color='red')
    ax3.plot(altgrid, HCrationeg, color='blue')
    ax3.axhline(fc, color='k', linestyle='dotted')
    ax3.set_yscale('log')
    ax3.set_ylim(bottom=1)
    ax3.set_ylabel('Ratio heat/cool')

    #use these with the altgrid:
    tools.set_alt_ax(ax1, altmax=altmax, labels=False)
    tools.set_alt_ax(ax2, altmax=altmax, labels=False)
    tools.set_alt_ax(ax3, altmax=altmax, labels=True)

    fig.tight_layout()
    if savename != None:
        plt.savefig(savename, bbox_inches='tight', dpi=200)
    plt.clf()
    plt.close()


def make_converged_plot(altgrid, altmax, path, Te, radheat, rho, radcool, expcool, advheat, advcool):
    """
    Makes a plot of the converged temperature profile, as well as the different
    heating/cooling rates.

    Parameters
    ----------
    altgrid : numpy.ndarray
        Radius grid in units of Rp.
    altmax : numeric
        Maximum altitude of the simulation in units of planet radius.
    path : _type_
        _description_
    Te : numpy.ndarray
        Converged temperature profile in units of K.
    radheat : numpy.ndarray
        Radiative heating rate in units of erg s-1 cm-3.
    rho : numpy.ndarray
        Density in units of g cm-3
    radcool : numpy.ndarray
        Radiative cooling rate in units of erg s-1 cm-3, as positive values.
    expcool : numpy.ndarray
        Expansion cooling rate in units of erg s-1 cm-3, as positive values.
    advheat : numpy.ndarray
        Advection heating rate in units of erg s-1 cm-3.
    advcool : numpy.ndarray
        Advection cooling rate in units of erg s-1 cm-3, as positive values.
    """

    fig, (ax1, ax2) = plt.subplots(2, figsize=(4,5.5))
    ax1.plot(altgrid, Te, color='k')
    ax1.set_ylabel('Temperature [K]')

    ax2.plot(altgrid, radheat/rho, color='red')
    ax2.plot(altgrid, radcool/rho, color='blue')
    ax2.plot(altgrid, expcool/rho, color='blue', linestyle='dashed')
    ax2.plot(altgrid, advheat/rho, color='red', linestyle='dotted')
    ax2.plot(altgrid, advcool/rho, color='blue', linestyle='dotted')
    ax2.set_yscale('log')
    ax2.set_ylim(0.1*min(min(radheat/rho), min(radcool/rho)), 2*max(max(radheat/rho), max(radcool/rho), max(expcool/rho), max(advheat/rho), max(advcool/rho)))
    ax2.set_ylabel('Rate [erg/s/g]')
    ax2.legend(((Line2D([], [], color='red', linestyle=(0,(6,6))), Line2D([], [], color='blue', linestyle=(6,(6,6)))),
                 Line2D([], [], color='blue', linestyle='dashed'),
                (Line2D([], [], color='red', linestyle=(0,(1,2,1,8))), Line2D([], [], color='blue', linestyle=(6,(1,2,1,8))))),
                ('radiation', 'expansion', 'advection'), loc='best', fontsize=8)


    #use these with the altgrid:
    tools.set_alt_ax(ax1, altmax=altmax, labels=False)
    tools.set_alt_ax(ax2, altmax=altmax)

    fig.tight_layout()
    plt.savefig(path+'converged.png', bbox_inches='tight', dpi=200)
    plt.clf()
    plt.close()


def check_converged(fc, HCratio, newTe, prevTe, linthresh=50.):
    """
    Checks whether the temperature profile is converged. At every radial cell,
    it checks for three conditions, one of which must be satisfied:
    1. The H/C ratio is less than fc (this is the "main" criterion).
    2. The newly proposed temperature profile is within the temperature difference
    that a H/C equal to fc would induce. In principle, we would expect that if
    this were the case, H/C itself would be < fc, but smoothing of the
    temperature profile can cause different behavior. For example, we can get stuck
    in a loop where H/C > fc, we then propose a new temperature profile that is 
    significantly different, but then after the smoothing step we end up with
    the profile that we had before. To break out of such a loop that never converges,
    we check if the temperature changes are less than we would expect for an
    "fc-converged" profile, even if H/C itself is still >fc. In practice, this
    means that the temperature profile changes less than 0.3 * log10(1.1),
    which is ~1%, so up to 100 K for a typical profile.
    3. The newly proposed temperature profile is less than `linthresh` different
    from the last iteration. This can be assumed to be precise enough convergence.

    Parameters
    ----------
    fc : numeric
        Convergence threshold for the total heating/cooling ratio.
    HCratio : numpy.ndarray
        Heating/cooling imbalance, output of the calc_HCratio() function.
    newTe : numpy.ndarray
        Newly proposed temperature profile based on both relaxation and
        construction algorithms, in units of K.
    prevTe : numpy.ndarray
        Temperature profile of the previous iteration in units of K.
    linthresh : numeric, optional
        Convergence threshold for T(r) as an absolute temperature difference
        in units of K, by default 50.

    Returns
    -------
    converged : bool
        Whether the temperature profile is converged.
    """

    ratioTe = np.maximum(newTe, prevTe) / np.minimum(newTe, prevTe) #take element wise ratio
    diffTe = np.abs(newTe - prevTe) #take element-wise absolute difference
    
    if np.all((np.abs(HCratio) < fc) | (ratioTe < (1 + 0.3 * np.log10(fc))) | (diffTe < linthresh)):
        converged = True
    else:
        converged = False

    return converged


def clean_converged_folder(folder):
    """
    Deletes all files in a folder that are not called "converged*".
    In the context of this module, it thus cleans all files of earlier
    iterations, as well as helper files, preserving only the final
    converged simulation.

    Parameters
    ----------
    folder : str
        Folder where the iterative algorithm is ran, typically:
        $SUNBATHER_PROJECT_PATH/sims/1D/*plname*/*dir*/parker_*T0*_*Mdot*/
    """

    if not os.path.isdir(folder):
        warnings.warn(f"This folder does not exist: {folder}")

    elif not os.path.isfile(folder+'/converged.in'):
        warnings.warn(f"This folder wasn't converged, I will not clean it: {folder}")

    else:
        for filename in os.listdir(folder):
            if filename[:9] != 'converged' and os.path.isfile(os.path.join(folder, filename)):
                os.remove(os.path.join(folder, filename))


def run_loop(path, itno, fc, save_sp=[], maxit=16):
    """
    Solves for the nonisothermal temperature profile of a Parker wind
    profile through an iterative convergence scheme including Cloudy.

    Parameters
    ----------
    path : str
        Folder where the iterative algorithm is ran, typically:
        $SUNBATHER_PROJECT_PATH/sims/1D/*plname*/*dir*/parker_*T0*_*Mdot*/.
        In this folder, the 'template.in' and 'iteration1.in' files must be
        present, which are created automatically by the convergeT_parker.py module.
    itno : int
        Iteration number to start from. Can only be different from 1 if
        this profile has been (partly) solved before.
    fc : float
        H/C convergence factor, see Linssen et al. (2024). A sensible value is 1.1.
    save_sp : list, optional
        A list of atomic/ionic species to let Cloudy save the number density profiles
        for in the final converged simulation. Those are needed when doing radiative
        transfer to produce transmission spectra. For example, to be able to make
        metastable helium spectra, 'He' needs to be in the save_sp list. By default [].
    maxit : int, optional
        Maximum number of iterations, by default 16.
    """

    if itno == 1: #iteration1 is just running Cloudy. Then, we move on to iteration2        
        tools.run_Cloudy('iteration1', folder=path)
        itno += 1

    #now, we have ran our iteration1 and can start the iterative scheme to find a new profile:
    while itno <= maxit:
        prev_sim = tools.Sim(path+f'iteration{itno-1}') #load Cloudy results from previous iteration
        Rp = prev_sim.p.R #planet radius in cm
        altmax = prev_sim.altmax #maximum radius of the simulation in units of Rp

        #make logspaced grid to use throughout the code, interpolate all quantities onto this grid.
        rgrid = np.logspace(np.log10(Rp), np.log10(altmax*Rp), num=1000)

        Te, mu, rho, v, radheat, radcool, expcool, advheat, advcool = simtogrid(prev_sim, rgrid) #get all needed Cloudy quantities on the grid
        HCratio = calc_HCratio(radheat, radcool, expcool, advheat, advcool) #H/C or C/H ratio, depending on which is larger

        #now the procedure starts - we first produce a new temperature profile
        newTe_relax = relaxTstruc(rgrid, path, itno, Te, HCratio) #apply the relaxation algorithm
        cloc = calc_cloc(radheat, radcool, expcool, advheat, advcool, HCratio) #look for a point from where we could use construction
        newTe_construct = None
        if cloc < len(rgrid) - 1:
            newTe_construct = constructTstruc(rgrid, newTe_relax, int(cloc), v, rho, mu, radheat, radcool) #apply construction algorithm

        make_rates_plot(rgrid/Rp, Te, newTe_relax, radheat, radcool, expcool, advheat, advcool,
                    rho, HCratio, altmax, fc, title=f'iteration {itno}',
                    savename=path+f'iteration{itno}.png', newTe_construct=newTe_construct, cloc=cloc)

        #get the final new temperature profile, based on whether the construction algorithm was applied
        if newTe_construct is None:
            newTe = newTe_relax
        else:
            newTe = newTe_construct

        #add this temperature profile to the 'iterations' file for future reference
        iterations_file = pd.read_csv(path+'iterations.txt', header=0, sep=' ')
        iterations_file['Te'+str(itno)] = newTe
        iterations_file.to_csv(path+'iterations.txt', sep=' ', float_format='%.7e', index=False)

        #now we check if the profile is converged.
        if itno <= 2: #always update the Te profile at least once - in case we start from a 'close' Parker wind profile that immediately satisfies fc
            converged = False
        else: 
            prevTe = iterations_file['Te'+str(itno-1)].values #read out from file instead of Sim because the file has higher resolution
            converged = check_converged(fc, HCratio, newTe, prevTe, linthresh=50.) #check convergence criteria

        if converged: #run once more with more output
            make_converged_plot(rgrid/Rp, altmax, path, Te, radheat, rho, radcool, expcool, advheat, advcool)
            #calculate these terms for the output converged.txt file - for fast access of some key parameters without loading in the Cloudy sim.
            np.savetxt(path+'converged.txt', np.column_stack((rgrid/Rp, rho, Te, mu, radheat, radcool, expcool, advheat, advcool)), fmt='%1.5e',
                        header='R rho Te mu radheat radcool expcool advheat advcool', comments='')
            
            #we run the last simulation one more time but with all the output files
            tools.copyadd_Cloudy_in(path+'iteration'+str(itno-1), path+'converged', 
                                    outfiles=['.heat', '.den', '.en'], denspecies=save_sp, 
                                    selected_den_levels=True, hcfrac=0.01)
            tools.run_Cloudy('converged', folder=path)
            tools.Sim(path+'converged') #read in the simulation, so we open the .en file (if it exists) and hence compress its size (see tools.process_energies())
            clean_converged_folder(path) #remove all non-converged files            
            print(f"Temperature profile converged: {path}")
            
            break

        else: #set up the next iteration
            Cltlaw = tools.alt_array_to_Cloudy(rgrid, newTe, altmax, Rp, 1000) #convert the temperature profile to a table format accepted by Cloudy

            tools.copyadd_Cloudy_in(path+'template', path+'iteration'+str(itno), tlaw=Cltlaw) #add temperature profile to the template input file
            if itno != maxit: #no use running it if we are not entering the next while-loop iteration
                tools.run_Cloudy(f'iteration{itno}', folder=path)
            else:
                print(f"Failed temperature convergence after {itno} iterations: {path}")

            itno += 1
