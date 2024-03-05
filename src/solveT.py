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


def simtogrid(sim, grid):
    '''
    Interpolates the 4 needed quantities to new grid.
    Extrapolate command is really only for boundary roundoff errors
    '''

    Te = interp1d(sim.ovr.depth, sim.ovr.Te, fill_value='extrapolate')(grid)
    mu = interp1d(sim.ovr.depth, sim.ovr.mu, fill_value='extrapolate')(grid)
    htot = interp1d(sim.ovr.depth, sim.cool.htot, fill_value='extrapolate')(grid)
    ctot = interp1d(sim.ovr.depth, sim.cool.ctot, fill_value='extrapolate')(grid)
    rho = interp1d(sim.ovr.depth, sim.ovr.rho, fill_value='extrapolate')(grid)

    return Te, mu, htot, ctot, rho


def getexpadvrates(grid, Te, mu, PdVprof, advecprof, smoothsize=None):
    '''
    Calculates PdV and advection rates for a temperature/mu structure
    '''

    ifuncPdVT = interp1d(10**PdVprof[:,0], 10**PdVprof[:,1], fill_value='extrapolate') #this is -1*k*v*drhodr/mH, so multiply with T and divide by mu still
    ifuncadvec = interp1d(10**advecprof[:,0], 10**advecprof[:,1], fill_value='extrapolate') #this is v*rho*(5/2)*k/mH, so multiply with d(T/mu)/dr still

    PdV = ifuncPdVT(grid) * Te / mu #POSITIVE
    assert len(PdV[PdV < 0]) == 0, "Found negative PdV rates. Check!"
    advec = ifuncadvec(grid) * np.gradient(Te / mu, grid)
    if smoothsize != None:
        PdV = tools.smooth_gaus_savgol(PdV, size=smoothsize)
        advec = tools.smooth_gaus_savgol(advec, size=smoothsize)
    adveccool, advecheat = advec.copy(), advec.copy()
    adveccool[adveccool > 0] = 0 #only keep negative values
    advecheat[advecheat < 0] = 0 #only keep positive values
    adveccool = -1*adveccool #make positive for plotting

    return PdV, advecheat, adveccool


def getbulkrates(htot, ctot, PdV, advecheat, adveccool):
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

    return totheat, totcool, nettotal, hcratio, hcratiopos, hcrationeg


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


def calc_cloc(path, itno, advecheat, adveccool, htot, ctot, totheat, PdV, hcratio):
    '''
    This function checks if there is a point in the atmosphere where we can use
    our construction instead of relaxation algorithm. It searches for two
    criteria; 1) if there is a point where advection heat dominates or 2) if
    there is a point where expansion cooling dominates.
    '''

    clocs = pd.read_table(path+'clocs.txt', names=['cloc'], index_col=0, delimiter=' ')

    #check for advection dominated regime (then we can construct part of the atmosphere)
    adcloc = None
    boolar = (advecheat / totheat > 0.5) #boolean array where advection heating dominates
    bothradnotdom = np.argmax((htot > advecheat) & (ctot > adveccool) & (ctot > PdV))
    boolar[bothradnotdom:] = False #now the boolean array stores where advection heating dominates AND where there is no point at higher altitudes that is rad. heat and rad. cool dominated
    if True in boolar: #if there is no such point, adcloc is None as stated above
        advdomloc = len(htot) - 1 - np.argmax(boolar[::-1]) #get lowest altitude location where advection dominates
        boolar2 = (advecheat / totheat) < 0.2
        advunimploc = np.argmax(boolar2[advdomloc:]) + advdomloc #first point at lower altitude where advection becomes unimportant (if no point exists, will return just advdomloc)
        #then walk to higher altitude again to find converged point. We are more lax with "converged" point if advection dominates more.
        boolar3 = (np.abs(hcratio[:advunimploc][::-1]) < 1.3 * np.clip((advecheat[:advunimploc][::-1] / htot[:advunimploc][::-1])**(2./3.), 1, 10))
        if True in boolar3: #otherwise it stays None
            adcloc = advunimploc - np.argmax(boolar3)

    #check for expansion dominated regime (then we can construct part of the atmosphere)
    cxcloc = None
    boolar4 = (np.abs(hcratio) < 1.5) & (PdV / ctot > 8.) #boolean array where the structure is amost converged and expansion dominates the cooling
    boolar4 = (PdV / ctot > 8.) #try this new version without checking for convergence - maybe this works better in some cases and worse in others.
    if False in boolar4 and True in boolar4: #then there is at least an occurence where this is not true.
        cxcloc = np.argmax(~boolar4) - 1 #this is the last location from index 0 where it is true
        if cxcloc < 1:
            cxcloc = None
    elif False not in boolar4: #then they are all True
        cxcloc = len(htot) - 1


    if adcloc == None and cxcloc == None:
        cloc = None
        save_cloc = 0 #to make it integer type
    else: #start construction from the lowest altitude of adcloc, cxcloc and the cloc of the previous iteration
        cloc = np.max([l for l in [adcloc, cxcloc, clocs.cloc[itno-1]] if l is not None])
        if cloc == clocs.cloc[itno-1]: #if new cloc is not lower than previous, start 5% closer to the planet surface anyway
            cloc_closer = np.max([cloc, min(len(htot)-1, cloc+int(0.05*len(htot)))])
            #but only if not pushed into rad. dom. regime
            if (ctot[cloc_closer] < PdV[cloc_closer]) or (ctot[cloc_closer] < adveccool[cloc_closer]) or (htot[cloc_closer] < advecheat[cloc_closer]):
                cloc = cloc_closer
        save_cloc = cloc


    clocs.loc[itno] = save_cloc #add the new cloc
    clocs.to_csv(path+'clocs.txt', sep=' ', header=None)

    return cloc


def relaxTstruc(sim, grid, altmax, Rp, PdVprof, advecprof, fc, path, itno):
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
    Te, mu, htot, ctot, rho = simtogrid(sim, grid) #get all needed Cloudy quantities on this grid
    PdV, advecheat, adveccool = getexpadvrates(grid, Te, mu, PdVprof, advecprof) #calculate PdV and advection rates
    totheat, totcool, nettotal, hcratio, hcratiopos, hcrationeg = getbulkrates(htot, ctot, PdV, advecheat, adveccool)

    cloc = calc_cloc(path, itno, advecheat, adveccool, htot, ctot, totheat, PdV, hcratio)

    #Check convergence.
    #Inner region really suffers from Cloudy roundoff errors and log interpolation and thus we are twice as lax there.
    #Also, when we start from nearby models, sometimes we converge immediately on the first iteration and we get the exact
    #same temperature profile. To prevent that, I force here that we do two iterations before calling converged.
    if max(np.abs(hcratio[:int(0.95*len(grid))])) < fc and max(np.abs(hcratio[int(0.95*len(grid)):])) < 1+(fc-1)*2 and itno > 2:
        converged = True
    else:
        converged = False


    if itno == 2: #save for first time
        np.savetxt(path+'iterations.txt', np.column_stack((grid, np.repeat(0.3, len(grid)), Te, ctot, htot)),
                    header='grid fac1 Te1 ctot1 htot1', comments='', delimiter=' ', fmt='%.7e')

    iterations_file = pd.read_csv(path+'iterations.txt', header=0, sep=' ')
    iterations_file['ctot'+str(itno-1)] = ctot
    iterations_file['htot'+str(itno-1)] = htot
    fac = iterations_file['fac'+str(itno-1)].values

    newTe = get_new_Tstruc(Te, hcratio, fac)
    smoothsize = int(len(grid)/(20*itno))
    snewTe = tools.smooth_gaus_savgol(newTe, size=smoothsize)
    snewTe = np.clip(snewTe, 1e1, 1e6) #smoothing may have pushed snewTe < 10K again.

    if itno >= 4: #check for fluctuations. If so, we decrease the deltaT factor
        pp_Te = iterations_file['Te'+str(itno-2)]
        previous_ratio = Te / pp_Te
        this_ratio = snewTe / Te #because of smoothing this is not exactly the same as fT
        fl = (((previous_ratio < 1) & (this_ratio > 1)) | ((previous_ratio > 1) & (this_ratio < 1)))
        fac[fl] = 2/3 * fac[fl] #take smaller changes in T in regions where the temperature fluctuates
        fac = np.clip(tools.smooth_gaus_savgol(fac, size=10), 0.02, 0.3)
        snewTe = get_new_Tstruc(Te, hcratio, fac) #recalculate new temperature profile with updated fac
        snewTe = tools.smooth_gaus_savgol(snewTe, size=smoothsize)
        snewTe = np.clip(snewTe, 1e1, 1e6)

    #set up the figure with plotted quantities (if converged, we will make a plot later)
    if not converged:
        make_rates_plot(altgrid, Te, snewTe, htot, ctot, PdV, advecheat, adveccool,
                        rho, hcratiopos, hcrationeg, altmax, fc, title='iteration '+str(itno)+' - relaxation',
                        savename=path+'iteration'+str(itno)+'_relax.png')

    iterations_file['fac'+str(itno)] = fac
    iterations_file.to_csv(path+'iterations.txt', sep=' ', float_format='%.7e', index=False)

    return snewTe, cloc, converged


def make_rates_plot(altgrid, Te, snewTe, htot, ctot, PdV, advecheat, adveccool, rho, hcratiopos, hcrationeg, altmax, fc, title=None, savename=None, **kwargs):
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(4,7))
    if title != None:
        ax1.set_title(title)
    ax1.plot(altgrid, Te, color='grey', label='prev Te')
    ax1.plot(altgrid, snewTe, color='black', label='new Te')
    if 'cnewTe' in kwargs:
        ax1.plot(altgrid, kwargs['cnewTe'], color='purple', label='constructed Te')
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

    if 'cloc' in kwargs:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(altgrid[kwargs['cloc']], color='k', linewidth=0.5, zorder=-10)

    fig.tight_layout()
    if savename != None:
        plt.savefig(savename, bbox_inches='tight', dpi=200)
    plt.clf()
    plt.close()


def make_converged_plot(sim, grid, PdVprof, advecprof, altmax, Rp, path):
    altgrid = altmax - grid/Rp
    Te, mu, htot, ctot, rho = simtogrid(sim, grid)
    PdV, advecheat, adveccool = getexpadvrates(grid, Te, mu, PdVprof, advecprof)

    #set up the figure with plotted quantities
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


def constructTstruc(sim, grid, snewTe, cloc, PdVprof, advecprof, altmax, Rp, itno, path, fc):
    '''
    This function constructs the temperature structure from a given location (cloc),
    by minimizing the heating/cooling ratio of all terms.
    '''

    Te, mu, htot, ctot, rho = simtogrid(sim, grid) #get all needed Cloudy quantities on this grid

    ifuncPdVT = interp1d(10**PdVprof[:,0], 10**PdVprof[:,1], fill_value='extrapolate') #this is -1*k*v*drhodr/mH, so multiply with T and divide by mu still
    ifuncadvec = interp1d(10**advecprof[:,0], 10**advecprof[:,1], fill_value='extrapolate') #this is v*rho*(5/2)*k/mH, so multiply with d(T/mu)/dr still


    def calchcratiohi(T, depth, mu, htot, ctot, T2, depth2, mu2):
        '''
        This function will be optimized to zero (=> h/c = 1) by the constructTstruc
        function. Goes up in altitude.
        '''

        PdV = ifuncPdVT(depth) * T / mu
        advec = ifuncadvec(depth) * ((T/mu) - (T2/mu2))/(depth - depth2)

        totheat = htot + max(advec, 0) #if advec is negative we don't add it here
        totcool = ctot + PdV - min(advec, 0) #if advec is positive we don't add it here

        hcratio = max(totheat, totcool) / min(totheat, totcool) #both entities are positive

        return hcratio - 1 #find root of this value to get hcratio close to 1
    

    def calchcratiohi2(T, depth, mu, htot, ctot, currentT, T2, depth2, mu2):
        '''
        Same as calchcratiohi() but this one also guesses a change in
        the radiative heating and cooling rates based on T.
        currentT is the temperature that corresponds to the htot and ctot values.
        '''

        PdV = ifuncPdVT(depth) * T / mu
        advec = ifuncadvec(depth) * ((T/mu) - (T2/mu2))/(depth - depth2)

        guess_htot = htot * (currentT / T) #so that if T > currentT, htot becomes lower. Trying some random functional form
        guess_ctot = ctot * (T / currentT) #vice versa (no sqrt because it seems to have a stronger T dependence)

        totheat = guess_htot + max(advec, 0) #if advec is negative we don't add it here
        totcool = guess_ctot + PdV - min(advec, 0) #if advec is positive we don't add it here

        hcratio = max(totheat, totcool) / min(totheat, totcool) #both entities are positive

        return hcratio - 1 #find root of this value to get hcratio close to 1


    cnewTe = np.copy(snewTe) #start with the temp struc from other function
    for l in range(cloc-1, -1, -1): #walk 'backwards' to higher altitudes
        #result = minimize_scalar(calchcratiohi, method='bounded', bounds=[1e1,1e6], args=(grid[l], mu[l], htot[l], ctot[l], cnewTe[l+1], grid[l+1], mu[l+1]))
        result = minimize_scalar(calchcratiohi2, method='bounded', bounds=[1e1,1e6], args=(grid[l], mu[l], htot[l], ctot[l], Te[l], cnewTe[l+1], grid[l+1], mu[l+1]))
        cnewTe[l] = result.x


    #get rid of the often abrupt edge where the constructed part sets in by smoothing it around that point
    scnewTe = tools.smooth_gaus_savgol(cnewTe, fraction=0.03)
    scnewTe = np.clip(scnewTe, 1e1, 1e6) #after smoothing we might have ended up below 10K.
    scweight = np.zeros(len(grid))
    scweight += sps.norm.pdf(range(len(grid)), cloc, int(len(grid)/30))
    scweight /= np.max(scweight) #normalize
    cweight = 1 - scweight
    cnewTe = scnewTe * scweight + cnewTe * cweight

    #for the new T structure:
    PdV, advecheat, adveccool = getexpadvrates(grid, cnewTe, mu, PdVprof, advecprof) #calculate PdV and advection rates
    totheat, totcool, nettotal, hcratio, hcratiopos, hcrationeg = getbulkrates(htot, ctot, PdV, advecheat, adveccool)

    #make altgrid which contains the altitude values corresponding to the depth grid. In values of Rp
    altgrid = altmax - grid/Rp
    make_rates_plot(altgrid, Te, snewTe, htot, ctot, PdV, advecheat, adveccool,
                    rho, hcratiopos, hcrationeg, altmax, fc, title='iteration '+str(itno)+' - construction',
                    savename=path+'iteration'+str(itno)+'_construct.png', cnewTe=cnewTe, cloc=cloc)

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


def run_once(path, itno, fc, altmax, Rp, PdVprof, advecprof):
    prev_iteration = tools.Sim(path+'iteration'+str(itno-1))
    #make logspaced grid to use throughout the code, interpolate all useful quantities to this grid. 5000 steps is usually enough (>> Cloudy internal grid)
    loggrid = altmax*Rp - np.logspace(np.log10(prev_iteration.ovr.alt.iloc[-1]), np.log10(prev_iteration.ovr.alt.iloc[0]), num=2500)[::-1]


    #now the procedure starts
    snewTe, cloc, converged = relaxTstruc(prev_iteration, loggrid, altmax, prev_iteration.p.R, PdVprof, advecprof, fc, path, itno)

    if converged:
        print("\nTemperature profile converged: "+path+"\nRun one more time with level output files and then stop.\n")
        make_converged_plot(prev_iteration, loggrid, PdVprof, advecprof, altmax, Rp, path)
        #calculate these terms for the output .txt file
        conv_Te, conv_mu, conv_htot, conv_ctot, conv_rho = simtogrid(prev_iteration, loggrid)
        conv_PdV, conv_advecheat, conv_adveccool = getexpadvrates(loggrid, conv_Te, conv_mu, PdVprof, advecprof)
        np.savetxt(path+'converged.txt', np.column_stack(((altmax-loggrid/Rp)[::-1], conv_rho[::-1], conv_Te[::-1],
            conv_mu[::-1], conv_htot[::-1], conv_ctot[::-1], conv_PdV[::-1], conv_advecheat[::-1], conv_adveccool[::-1])), fmt='%1.5e',
            header='R rho Te mu radheat radcool PdV advheat advcool', comments='')
        return converged

    if cloc != None:
        snewTe = constructTstruc(prev_iteration, loggrid, snewTe, cloc, PdVprof, advecprof, altmax, Rp, itno, path, fc)

    iterations_file = pd.read_csv(path+'iterations.txt', header=0, sep=' ')
    iterations_file['Te'+str(itno)] = snewTe
    iterations_file.to_csv(path+'iterations.txt', sep=' ', float_format='%.7e', index=False)

    #if we get to here, it means that we were not converged to <fc. Then still check if Te structure is changing. If not, we call it converged.
    if itno >= 3:
        p_Te = iterations_file['Te'+str(itno-1)].values #previous-previous Te
        Tchanging = check_T_changing(fc, loggrid, snewTe, loggrid, p_Te, linthresh=50.)
        if not Tchanging:
            print("\nTemperature profile not converged to H/C < fc, but it does not change substantially anymore between iterations. "
                    +"Smoothing can cause this. The temperature profile should be accurate enough, I'll call it converged now: "+path+"\n")
            make_converged_plot(prev_iteration, loggrid, PdVprof, advecprof, altmax, Rp, path)
            #calculate these terms for the output .txt file
            conv_Te, conv_mu, conv_htot, conv_ctot, conv_rho = simtogrid(prev_iteration, loggrid)
            conv_PdV, conv_advecheat, conv_adveccool = getexpadvrates(loggrid, conv_Te, conv_mu, PdVprof, advecprof)
            np.savetxt(path+'converged.txt', np.column_stack(((altmax-loggrid/Rp)[::-1], conv_rho[::-1], conv_Te[::-1],
                conv_mu[::-1], conv_htot[::-1], conv_ctot[::-1], conv_PdV[::-1], conv_advecheat[::-1], conv_adveccool[::-1])), fmt='%1.5e',
                header='R rho Te mu radheat radcool PdV advheat advcool', comments='')
            converged = True
            return converged



    #we cannot use the tools.cl_table function because that expects the arrays to be ordered from low to high altitude
    #also, that one uses the grid you give it, instead of extrapolating from Rp to altmax*Rp
    Clgridr1 = np.logspace(np.log10(Rp), np.log10(altmax*Rp), num=1000)
    Clgridr1 = (Clgridr1 - Clgridr1[0])
    #sample the first 10 points better since Cloudy messes up with log-space interpolation there
    Clgridr2 = np.logspace(-2, np.log10(Clgridr1[9]), num=300)
    Clgridr = np.concatenate((Clgridr2, Clgridr1[10:]))
    Clgridr[0] = 1e-35

    ifunc_snewTe = interp1d(loggrid, snewTe, fill_value='extrapolate') #to put on coarse grid for cloudy
    ClgridT = ifunc_snewTe(Clgridr)
    Cltlaw = np.log10(np.column_stack((Clgridr, ClgridT)))

    tools.copyadd_Cloudy_in(path+'template', path+'iteration'+str(itno), tlaw=Cltlaw)

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


def run_loop(path, itno, fc, altmax, Rp, PdVprof, advecprof, save_sp=[], maxit=16):
    if itno == 0: #this means we resume from the highest found previously ran iteration
        pattern = r'iteration(\d+)\.out'
        max_iteration = -1
        for filename in os.listdir(path):
            if os.path.isfile(os.path.join(path, filename)):
                if re.search(pattern, filename):
                    iteration_number = int(re.search(pattern, filename).group(1))
                    if iteration_number > max_iteration:
                        max_iteration = iteration_number
        if max_iteration == -1:
            warnings.warn(f"This folder does not have any 'iteration' files, I cannot resume from the highest one: {path}")
            return
        else:
            print("\nFound the highest iteration "+path+"iteration"+str(max_iteration)+", will resume there.\n")
            itno = max_iteration+1

    if itno == 1: #then first Cloudy (for itno>1, first solve script)
        os.system("cd "+path+" && "+tools.cloudyruncommand+" iteration1 && cd "+tools.projectpath+"/sims/1D")
        itno += 1

    converged = False
    while not converged and itno <= maxit:
        converged = run_once(path, itno, fc, altmax, Rp, PdVprof, advecprof)
        if converged: #we run the last simulation one more time but with all the output files
            if save_sp == []:
                tools.copyadd_Cloudy_in(path+'iteration'+str(itno-1), path+'converged', outfiles=['.heat'], hcfrac=0.01)
            else:
                tools.copyadd_Cloudy_in(path+'iteration'+str(itno-1), path+'converged', outfiles=['.heat', '.den', '.en'], denspecies=save_sp, selected_den_levels=True, hcfrac=0.01)
            os.system("cd "+path+" && "+tools.cloudyruncommand+" converged && cd "+tools.projectpath+"/sims/1D")
            tools.Sim(path+'converged') #by reading in the simulation, we open the .en file (if it exists) and hence compress its size.
            clean_converged_folder(path) #remove all non-converged files
            break
        if itno != maxit:
            os.system("cd "+path+" && "+tools.cloudyruncommand+" iteration"+str(itno)+" && cd "+tools.projectpath+"/sims/1D")

        itno += 1
