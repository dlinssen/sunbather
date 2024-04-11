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
import re
import warnings


def calc_expansion(r, rho, v, Te, mu):
    '''
    Calcules expansion cooling (Linssen et al. 2024 Eq. 3 second term)
    Requires that r is in the direction of v (i.e. usually in altitude scale).

    r:      radius in the atmosphere in cm
    rho:    density in g cm-3
    v:      velocity in cm s-1
    Te:     temperature in K
    mu:     mean particle mass in amu
    '''

    expansion = tools.k/tools.mH * Te * v / mu * np.gradient(rho, r)
    
    return expansion 


def calc_advection(r, rho, v, Te, mu):
    '''
    Calcules advection heating/cooling (Linssen et al. 2024 Eq. 3 first term)
    Requires that r is in the direction of v (i.e. usually in altitude scale).

    r:      radius in the atmosphere in cm
    rho:    density in g cm-3
    v:      velocity in cm s-1
    Te:     temperature in K
    mu:     mean particle mass in amu
    '''

    advection = -1 * tools.k/(tools.mH * 2/3) * rho * v * np.gradient(Te/mu, r)
    
    return advection


def simtogrid(sim, grid):
    '''
    Interpolates the 4 needed quantities to new grid.
    Extrapolate command is really only for boundary roundoff errors
    '''

    Te = interp1d(sim.ovr.depth, sim.ovr.Te, fill_value='extrapolate')(grid)
    mu = interp1d(sim.ovr.depth, sim.ovr.mu, fill_value='extrapolate')(grid)
    rho = interp1d(sim.ovr.depth, sim.ovr.rho, fill_value='extrapolate')(grid)
    v = interp1d(sim.ovr.depth, sim.ovr.v, fill_value='extrapolate')(grid)
    htot = interp1d(sim.ovr.depth, sim.cool.htot, fill_value='extrapolate')(grid) #radiative heat
    ctot = interp1d(sim.ovr.depth, sim.cool.ctot, fill_value='extrapolate')(grid) #radiative cool

    exp = calc_expansion(grid, rho, v, Te, mu) #We gave a depth-grid which has reverse direction from r-grid that function expects, but we don't correct it so that we get positively defined rates here
    adv = -1 * calc_advection(grid, rho, v, Te, mu) #add -1 here since we gave a depth-grid which has reverse direction from r-grid that function expects

    advheat, advcool = np.copy(adv), -1 * np.copy(adv)
    advheat[advheat < 0] = 0.
    advcool[advcool < 0] = 0.

    assert np.min(exp) >= 0, "Found negative expansion cooling rates. Check!"

    return Te, mu, rho, v, htot, ctot, exp, advheat, advcool


def getHCratio(htot, ctot, PdV, advecheat, adveccool):
    '''
    Combines the different heating and cooling rates into some useful (plotting) quantities
    '''

    totheat = htot + advecheat
    totcool = ctot + PdV + adveccool #all cooling rates are positive values
    nettotal = (totheat - totcool)

    hcratio = np.sign(nettotal) * np.maximum(totheat, totcool) / np.minimum(totheat,totcool)
    #for plotting purposes:
    hcratiopos, hcrationeg = np.copy(hcratio), np.copy(hcratio)
    hcratiopos[hcratiopos < 0] = 0
    hcrationeg[hcrationeg > 0] = 0

    return hcratio, hcratiopos, hcrationeg


def get_new_Tstruc(old_Te, hcratio, fac):
    '''
    Returns a new temperature structure based on the old structure and the heating/cooling ratio.
    '''

    deltaT = fac * np.sign(hcratio) * np.log10(np.abs(hcratio)) #take log-based approach to deltaT
    fT = np.copy(deltaT) #the temperature multiplication fraction
    fT[deltaT < 0] = 1 + deltaT[deltaT < 0]
    fT[deltaT > 0] = 1/(1 - deltaT[deltaT > 0])
    fT = np.clip(fT, 0.5, 2) #max change is a factor 2 up or down in temperature
    newTe = old_Te * fT
    newTe = np.clip(newTe, 1e1, 1e6) #set minimum temperature to 10K
    return newTe


def calc_cloc(path, itno, htot, ctot, PdV, advecheat, adveccool, hcratio):
    '''
    This function checks if there is a point in the atmosphere where we can use
    our construction instead of relaxation algorithm. It searches for two
    criteria; 1) if there is a point where advection heat dominates or 2) if
    there is a point where expansion cooling dominates.
    '''

    #clocs = pd.read_table(path+'clocs.txt', names=['cloc'], index_col=0, delimiter=' ')
    clocs = pd.read_csv(path+'clocs.csv', index_col='iteration')

    #check for advection dominated regime
    adcloc = np.nan
    boolar = (advecheat > htot) #boolean array where advection heating dominates
    bothradnotdom = np.argmax((htot > advecheat) & (ctot > adveccool) & (ctot > PdV))
    boolar[bothradnotdom:] = False #now the boolean array stores where advection heating dominates AND where there is no point at higher altitudes that is rad. heat and rad. cool dominated
    if True in boolar: #if there is no such point, adcloc is None as stated above
        advdomloc = len(htot) - 1 - np.argmax(boolar[::-1]) #get lowest altitude location where advection dominates
        boolar2 = (advecheat < 0.25 * htot)
        advunimploc = np.argmax(boolar2[advdomloc:]) + advdomloc #first point at lower altitude where advection becomes unimportant (if no point exists, will return just advdomloc)
        #then walk to higher altitude again to find converged point. We are more lax with "converged" point if advection dominates more.
        boolar3 = (np.abs(hcratio[:advunimploc][::-1]) < 1.3 * np.clip((advecheat[:advunimploc][::-1] / htot[:advunimploc][::-1])**(2./3.), 1, 10))
        if True in boolar3: #otherwise it stays None
            adcloc = advunimploc - np.argmax(boolar3)

    #check for expansion dominated regime
    cxcloc = np.nan
    boolar4 = (np.abs(hcratio) < 1.5) & (PdV / ctot > 8.) #boolean array where the structure is amost converged and expansion dominates the cooling
    boolar4 = (PdV / ctot > 8.) #try this new version without checking for convergence - maybe this works better in some cases and worse in others.
    if False in boolar4 and True in boolar4: #then there is at least an occurence where this is not true.
        cxcloc = np.argmax(~boolar4) - 1 #this is the last location from index 0 where it is true
        if cxcloc < 1:
            cxcloc = np.nan
    elif False not in boolar4: #then they are all True
        cxcloc = len(htot) - 1

    cloc = max(adcloc, cxcloc, clocs.loc[itno-1, 'construct_loc']) #max of these indices is the lowest altitude

    """
    if cloc == clocs.loc[itno-1, 'construct_loc']: #if new cloc is not lower than previous, start 5% closer to the planet surface anyway
        cloc_closer = np.max([cloc, min(len(htot)-1, cloc+int(0.05*len(htot)))])
        #but only if not pushed into rad. dom. regime
        if (ctot[cloc_closer] < PdV[cloc_closer]) or (ctot[cloc_closer] < adveccool[cloc_closer]) or (htot[cloc_closer] < advecheat[cloc_closer]):
            cloc = cloc_closer
    """

    clocs.loc[itno, 'construct_loc'] = cloc
    clocs.to_csv(path+'clocs.csv')

    return cloc


def relaxTstruc(grid, altmax, Rp, path, itno, Te, hcratio):
    '''
    This function finds a new temperature structure by relaxation:
    Add all rates, find ratio of heating to cooling rate,
    and change the temperature structure based on log of that value.
    If the radiation dominated part is converged, we 'overwrite' the
    advection/expansion dominated part of the atmosphere by the constructTstruc() function
    since that is harder to find by relaxation method and easier by construction.
    '''

    #make altgrid which contains the altitude values corresponding to the depth grid. In values of Rp
    altgrid = altmax - grid/Rp

    if itno == 2: #save for first time
        np.savetxt(path+'iterations.txt', np.column_stack((grid, np.repeat(0.3, len(grid)), Te)),
                    header='grid fac1 Te1', comments='', delimiter=' ', fmt='%.7e')

    iterations_file = pd.read_csv(path+'iterations.txt', header=0, sep=' ')
    fac = iterations_file['fac'+str(itno-1)].values

    newTe = get_new_Tstruc(Te, hcratio, fac)
    smoothsize = int(len(grid)/(20*itno))
    newTe = tools.smooth_gaus_savgol(newTe, size=smoothsize)
    newTe = np.clip(newTe, 1e1, 1e6) #smoothing may have pushed snewTe < 10K again.

    if itno >= 4: #check for fluctuations. If so, we decrease the deltaT factor
        pp_Te = iterations_file['Te'+str(itno-2)]
        previous_ratio = Te / pp_Te
        this_ratio = newTe / Te #because of smoothing this is not exactly the same as fT
        fl = (((previous_ratio < 1) & (this_ratio > 1)) | ((previous_ratio > 1) & (this_ratio < 1)))
        fac[fl] = 2/3 * fac[fl] #take smaller changes in T in regions where the temperature fluctuates
        fac = np.clip(tools.smooth_gaus_savgol(fac, size=10), 0.02, 0.3)
        newTe = get_new_Tstruc(Te, hcratio, fac) #recalculate new temperature profile with updated fac
        newTe = tools.smooth_gaus_savgol(newTe, size=smoothsize)
        newTe = np.clip(newTe, 1e1, 1e6)

    """
    #set up the figure with plotted quantities (if converged, we will make a plot later)
    make_rates_plot(altgrid, Te, snewTe, htot, ctot, PdV, advecheat, adveccool,
                    rho, hcratiopos, hcrationeg, altmax, fc, title='iteration '+str(itno)+' - relaxation',
                    savename=path+'iteration'+str(itno)+'_relax.png')
    """

    iterations_file['fac'+str(itno)] = fac
    iterations_file.to_csv(path+'iterations.txt', sep=' ', float_format='%.7e', index=False)

    return newTe


def make_rates_plot(altgrid, Te, snewTe, htot, ctot, PdV, advecheat, adveccool, rho, hcratiopos, hcrationeg, altmax, fc, 
                    cnewTe=None, cloc=None, title=None, savename=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(4,7))
    if title != None:
        ax1.set_title(title)
    ax1.plot(altgrid, Te, color='#4CAF50', label='previous')
    ax1.plot(altgrid, snewTe, color='#FFA500', label='relaxation')
    if cnewTe is not None:
        ax1.plot(altgrid, cnewTe, color='#800080', label='construction')
        ax1.scatter(altgrid[cloc], snewTe[cloc], color='#800080')
    ax1.set_ylabel('Temperature [K]')
    ax1.legend(loc='best', fontsize=8)

    ax2.plot(altgrid, htot/rho, color='red')
    ax2.plot(altgrid, ctot/rho, color='blue')
    ax2.plot(altgrid, PdV/rho, color='blue', linestyle='dashed')
    ax2.plot(altgrid, advecheat/rho, color='red', linestyle='dotted')
    ax2.plot(altgrid, adveccool/rho, color='blue', linestyle='dotted')
    ax2.set_yscale('log')
    ax2.set_ylim(0.1*min(min(htot/rho), min(ctot/rho)), 2*max(max(htot/rho), max(ctot/rho), max(PdV/rho), max(advecheat/rho), max(adveccool/rho)))
    ax2.set_ylabel('Rate [erg/s/g]')
    ax2.legend(((Line2D([], [], color='red', linestyle=(0,(6,6))), Line2D([], [], color='blue', linestyle=(6,(6,6)))),
                 Line2D([], [], color='blue', linestyle='dashed'),
                (Line2D([], [], color='red', linestyle=(0,(1,2,1,8))), Line2D([], [], color='blue', linestyle=(6,(1,2,1,8))))),
                ('radiation', 'expansion', 'advection'), loc='best', fontsize=8)

    ax3.plot(altgrid, hcratiopos, color='red')
    ax3.plot(altgrid, -hcrationeg, color='blue')
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


def make_converged_plot(altgrid, altmax, path, Te, htot, rho, ctot, PdV, advecheat, adveccool):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(4,5.5))
    ax1.plot(altgrid, Te, color='k')
    ax1.set_ylabel('Temperature [K]')

    ax2.plot(altgrid, htot/rho, color='red')
    ax2.plot(altgrid, ctot/rho, color='blue')
    ax2.plot(altgrid, PdV/rho, color='blue', linestyle='dashed')
    ax2.plot(altgrid, advecheat/rho, color='red', linestyle='dotted')
    ax2.plot(altgrid, adveccool/rho, color='blue', linestyle='dotted')
    ax2.set_yscale('log')
    ax2.set_ylim(0.1*min(min(htot/rho), min(ctot/rho)), 2*max(max(htot/rho), max(ctot/rho), max(PdV/rho), max(advecheat/rho), max(adveccool/rho)))
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


def constructTstruc(grid, snewTe, cloc, v, rho, mu, Te, htot, ctot):
    '''
    This function constructs the temperature structure from a given location (cloc),
    by minimizing the heating/cooling ratio of all terms.
    '''

    expansion_Tdivmu = tools.k/tools.mH * v * np.gradient(rho, grid) #this is expansion cooling except for the T/mu term 
    advection_gradTdivmu = tools.k/(tools.mH * 2/3) * rho * v #this is advection except for the d(T/mu)/dr term (with a minus sign due to depth != radius)

    ifuncPdVT = interp1d(grid, expansion_Tdivmu, fill_value='extrapolate')
    ifuncadvec = interp1d(grid, advection_gradTdivmu, fill_value='extrapolate')
    
    def calcHCratio(T, depth, mu, htot, ctot, currentT, T2, depth2, mu2):
        '''
        currentT is the temperature that corresponds to the htot and ctot values.
        '''

        PdV = ifuncPdVT(depth) * T / mu
        advec = ifuncadvec(depth) * ((T/mu) - (T2/mu2))/(depth - depth2)

        guess_htot = htot * (currentT / T) #so that if T > currentT, htot becomes lower. Trying some random (linear) functional form
        guess_ctot = ctot * (T / currentT) #vice versa

        totheat = guess_htot + max(advec, 0) #if advec is negative we don't add it here
        totcool = guess_ctot + PdV - min(advec, 0) #if advec is positive we don't add it here

        hcratio = max(totheat, totcool) / min(totheat, totcool) #both entities are positive

        return hcratio - 1 #find root of this value to get hcratio close to 1


    cnewTe = np.copy(snewTe) #start with the temp struc from other function
    for l in range(cloc-1, -1, -1): #walk 'backwards' to higher altitudes
        result = minimize_scalar(calcHCratio, method='bounded', bounds=[1e1,1e6], args=(grid[l], mu[l], htot[l], ctot[l], Te[l], cnewTe[l+1], grid[l+1], mu[l+1]))
        cnewTe[l] = result.x


    #get rid of the often abrupt edge where the constructed part sets in by smoothing it around that point
    scnewTe = tools.smooth_gaus_savgol(cnewTe, fraction=0.03)
    scnewTe = np.clip(scnewTe, 1e1, 1e6) #after smoothing we might have ended up below 10K.
    scweight = np.zeros(len(grid))
    scweight += sps.norm.pdf(range(len(grid)), cloc, int(len(grid)/30))
    scweight /= np.max(scweight) #normalize
    cweight = 1 - scweight
    cnewTe = scnewTe * scweight + cnewTe * cweight

    return cnewTe


def check_T_changing(fc, grid, newTe, prevgrid, prevTe, fac=0.5, linthresh=50.):
    '''
    This function checks if the newly proposed temperature structure is actually changing w.r.t.
    the previous iteration. Because maybe H/C is still > fc, but because of smoothing,
    we are running the same temperature structure still iteratively.
    '''

    prevTe = interp1d(prevgrid, prevTe, fill_value='extrapolate')(grid) #put on same grid as the newTe
    ratioTe = np.maximum(newTe, prevTe) / np.minimum(newTe, prevTe) #take element wise ratio
    diffTe = np.abs(newTe - prevTe)

    #if not converged, we would expect max of hcratio > fc, and thus temp to change by more than (1+np.log10(fc)).
    #If that's not the case, roundoffs or T=10K are preventing convergence.
    if np.all((ratioTe < (1 + fac * np.log10(fc))) | (diffTe < linthresh)):
        Tchanging = False
    else:
        Tchanging = True

    return Tchanging


def check_fc_converged(fc, depthgrid, hcratio):
    '''
    Checks whether the temperature profile is converged to a H/C ratio < fc.
    Inner region really suffers from Cloudy roundoff errors and log interpolation and thus we are twice as lax there.
    Also, when we start from nearby models, sometimes we converge immediately on the first iteration and we get the exact
    same temperature profile. To prevent that, I force here that we do two iterations before calling converged.
    '''

    if max(np.abs(hcratio[:int(0.95*len(depthgrid))])) < fc and max(np.abs(hcratio[int(0.95*len(depthgrid)):])) < 1+(fc-1)*2:
        converged = True
    else:
        converged = False

    return converged


def clean_converged_folder(folder):
    '''
    Removes all files that are not of the converged simulation (and thus part
    of earlier iterations / help files) from a folder.
    '''

    if not os.path.isdir(folder):
        warnings.warn(f"This folder does not exist: {folder}")

    elif not os.path.isfile(folder+'/converged.in'):
        warnings.warn(f"This folder wasn't converged, I will not clean it: {folder}")

    else:
        for filename in os.listdir(folder):
            if filename[:9] != 'converged' and os.path.isfile(os.path.join(folder, filename)):
                os.remove(os.path.join(folder, filename))


def run_loop(path, itno, fc, save_sp=[], maxit=16):
    '''
    Iteratively solves the temperature profile
    '''

    if itno == 1: #iteration1 is just running Cloudy. Then, we move on to iteration2
        tools.run_Cloudy('iteration1', folder=path)
        itno += 1

    #now, we have ran our iteration1 and can start the iterative scheme to find a new profile:
    while itno <= maxit:
        prev_sim = tools.Sim(path+f'iteration{itno-1}')
        altmax = prev_sim.altmax
        Rp = prev_sim.p.R

        #make logspaced grid to use throughout the code, interpolate all useful quantities to this grid.
        loggrid = altmax*Rp - np.logspace(np.log10(prev_sim.ovr.alt.iloc[-1]), np.log10(prev_sim.ovr.alt.iloc[0]), num=2500)[::-1]

        Te, mu, rho, v, htot, ctot, PdV, advecheat, adveccool = simtogrid(prev_sim, loggrid) #get all needed Cloudy quantities on the grid
        hcratio, hcratiopos, hcrationeg = getHCratio(htot, ctot, PdV, advecheat, adveccool)

        #now the procedure starts - we first produce a new temperature profile
        snewTe = relaxTstruc(loggrid, altmax, Rp, path, itno, Te, hcratio)
        cloc = calc_cloc(path, itno, htot, ctot, PdV, advecheat, adveccool, hcratio)
        cnewTe = None
        if ~np.isnan(cloc):
            cnewTe = constructTstruc(loggrid, snewTe, int(cloc), v, rho, mu, Te, htot, ctot)

        make_rates_plot(altmax - loggrid/Rp, Te, snewTe, htot, ctot, PdV, advecheat, adveccool,
                    rho, hcratiopos, hcrationeg, altmax, fc, title=f'iteration {itno}',
                    savename=path+f'iteration{itno}.png', cnewTe=cnewTe, cloc=cloc)

        if cnewTe is not None:
            snewTe = cnewTe

        #add this temperature profile to the iterations file
        iterations_file = pd.read_csv(path+'iterations.txt', header=0, sep=' ')
        iterations_file['Te'+str(itno)] = snewTe
        iterations_file.to_csv(path+'iterations.txt', sep=' ', float_format='%.7e', index=False)

        #now we check if the profile is converged. Either to H/C<fc, or if the Te profile does not change anymore (indicating smoothing prevents H/C<fc)
        converged = False #first assume not converged
        if itno > 2: #always update the Te profile at least once - in case we start from a 'close' Parker wind profile that immediately satisfies fc
            converged = check_fc_converged(fc, loggrid, hcratio)
            p_Te = iterations_file['Te'+str(itno-1)].values #previous-previous Te
            Tchanging = check_T_changing(fc, loggrid, snewTe, loggrid, p_Te, linthresh=50.)
            if not Tchanging: #then say this indicates convergence that is simply limited by smoothing.
                converged = True

        if converged: #run once more with all output
            print(f"Temperature profile converged: {path}")
            make_converged_plot(altmax - loggrid/Rp, altmax, path, Te, htot, rho, ctot, PdV, advecheat, adveccool)
            #calculate these terms for the output .txt file
            np.savetxt(path+'converged.txt', np.column_stack(((altmax-loggrid/Rp)[::-1], rho[::-1], Te[::-1],
                    mu[::-1], htot[::-1], ctot[::-1], PdV[::-1], advecheat[::-1], adveccool[::-1])), fmt='%1.5e',
                    header='R rho Te mu radheat radcool PdV advheat advcool', comments='')
            
            #we run the last simulation one more time but with all the output files
            tools.copyadd_Cloudy_in(path+'iteration'+str(itno-1), path+'converged', 
                                    outfiles=['.heat', '.den', '.en'], denspecies=save_sp, 
                                    selected_den_levels=True, hcfrac=0.01)
            tools.run_Cloudy('converged', folder=path)
            tools.Sim(path+'converged') #read in the simulation, so we open the .en file (if it exists) and hence compress its size (see tools.process_energies())
            clean_converged_folder(path) #remove all non-converged files            
            
            break

        else: #set up the next iteration
            Cltlaw = tools.alt_array_to_Cloudy(altmax*Rp - loggrid[::-1], snewTe[::-1], altmax, Rp, 1300)

            tools.copyadd_Cloudy_in(path+'template', path+'iteration'+str(itno), tlaw=Cltlaw)
            if itno != maxit:
                tools.run_Cloudy(f'iteration{itno}', folder=path)

            itno += 1
