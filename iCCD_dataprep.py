import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy import interpolate
import pandas as pd

#%% Plot of timepoints for during acquistion
def one_window(t0,dt,n):
    """Calculates the timepoints for a single window of iCCD acquisition.

    Args:
        t0 (float): start time of iCCD acquistiion (ns)
        dt (float): Gate Width (ns)
        n (float): Number of steps
        

    Returns:
        _type_: _description_
    """
    t_final = t0 + dt*n # Time zero + gate width * number of steps = final time
    t_real = t_final - t0 # Real time after time zero
    t_overlap = t0 + 0.7*t_real # Indicates where next window starts

    return [t0, dt, n, round(t_final,3), round(t_overlap,3), round(t_real,3)]

def timesplot(tdf,t0):
    """Creates a dataframe and plots the overlaps of the timepoints for each window of iCCD acquisition.

    Args:
        tdf (DataFrame): DataFrame of timepoints for each window of iCCD acquisition using one_window function
        t0 (float): Time zero of first window of iCCD acquisition
    """
    fig = plt.figure(figsize = (10,3))
    ax = fig.add_subplot(111)
    for i in range(len(tdf)):
        pts_i = (np.linspace(tdf.iloc[i]['t_start']-t0,tdf.iloc[i]['t_end']-t0,int(tdf.iloc[i]['n']+1))[:-1])
        plt.plot(pts_i,np.repeat(i,len(pts_i)), c='b',lw=0, marker='.', ms=3)
        if i != len(tdf)-1:
            plt.plot([[tdf.iloc[i]['t_end']-t0],[tdf.iloc[i]['t_end']-t0]],[i,i+1], c='r',lw=1)
        
        plt.xlim(0.003,500)
        plt.xscale('log'),plt.xlabel('Time (μs)')
    return
#%% iCCD Corrections
def ICCDcorrections(data,cwl,trans_calib = r'C:/Users/bail2.BLUE/OneDrive - University of Cambridge/Documents/1PhD/2023/iCCD/FELH0550_Transmission.txt', 
                    sensitivity_calib = r'C:/Users/bail2.BLUE/OneDrive - University of Cambridge/Documents/1PhD/2023/iCCD/ICCD_cWL_sensitivity_ed.txt', display=False):
    """Correct the data for grating sensitivity and transmission through 550nm LP filter.

    Args:
        data (array): 3D array of data with time, wavelength and signal
        cwl (float): Central wavelength of grating
        trans_calib (regexp, optional): _description_. Defaults to r'C:/Users/bail2.BLUE/OneDrive - University of Cambridge/Documents/1PhD/2023/iCCD/FELH0550_Transmission.txt'.
        sensitivity_calib (regexp, optional): _description_. Defaults to r'C:/Users/bail2.BLUE/OneDrive - University of Cambridge/Documents/1PhD/2023/iCCD/ICCD_cWL_sensitivity_ed.txt'.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    wl = data[1:,0]
    
    try:
        # Load transmission file from file path
        calib_data = np.genfromtxt(trans_calib, dtype=float, delimiter='\t')
        calib_wl = calib_data[:, 0]  # Wavelengths of transmission file
        calib_trans = calib_data[:, 1]  # Transmission at wavelength of LP filter
    except OSError:
        raise ValueError("Invalid trans_calib file path. Please specify a valid file path.")
    trans_filter = np.interp(wl, calib_wl, calib_trans)/100 #interpolates the transmission file WLs with relevant transmission file wls
    # ICCD sensitivity calib

    try:
        column = int((cwl - 400) / 25)  # chooses column of file
        calib_wl_ICCD = np.genfromtxt(sensitivity_calib, dtype=float, delimiter='\t', skip_header=1, usecols=(column))
        nanlist = np.isnan(calib_wl_ICCD)  # gets rid of nan values
        calib_wl_ICCD = calib_wl_ICCD[~nanlist]
        calib_sens_ICCD = np.genfromtxt(sensitivity_calib, dtype=float, delimiter='\t', skip_header=1, usecols=(column + 1))
        calib_sens_ICCD = calib_sens_ICCD[~nanlist]
    except OSError:
        raise ValueError("Invalid sensitivity_calib file path. Please specify a valid file path.")
        
    #interpolates data
    trans_sens = np.interp(wl, calib_wl_ICCD, calib_sens_ICCD) #interpolates wavelengths
    #total transmission calibration:
    trans =   trans_filter*  trans_sens # overall filter
    
    data[1:,1:] = data[1:,1:]/trans[:,None]
    if display:
        plt.plot(wl,trans_filter)
        plt.show()
        plt.plot(wl,trans_sens)
        plt.show()
        plt.plot(wl,trans)
        plt.show()
        plt.plot(wl,data[1:,2]/trans)
        plt.show()
    
    return data, trans_sens,trans_filter


#%% Save to sorted

def save_sorted(master_decay, filename=None):
    if not os.path.exists("Sorted"):
        os.makedirs("Sorted")
    else:
        if filename is None:
            print('NO FILENAME SET - call filename/path or file will not save')
        else:
            print(os.listdir())
            print(filename)
            # Save the file as a numpy file in the Sorted folder
            np.save('Sorted/' + filename + '.npy', master_decay)
        

#%% Load ASCII
# Load data from the ASC file and apply corrections



def loadASC(file,cwl=700,plot=False,adjust_skips = 0):
    """Loads data from an ASC file and applies corrections for ICCD sensitivity and transmission through a 550nm LP filter.

    Args:
        file (str): File name of the ASC file.
        wlmin (int, optional): Minimum limit of wavelengths. Defaults to 200.
        wlmax (int, optional): Maximum limit of wavelengths. Defaults to -200.
        cwl (int, optional): Central wavelength of grating. Defaults to 700.
        plot (bool, optional): Plots walvelengths vs signal for each t. Defaults to False.
        adjust_skips (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: data with first column = wavelengths, first row = timepoints, and the rest of the data is the signal
    """
    #Load the file, cutting out metadata from bottom
    data = np.genfromtxt(file,skip_footer=34,usecols=None)

    #Wavelengths may change so good to define so we can use the length to get straight to the metadata in all files
    wls = list(data[:,0])
    
    #Define start time, gate step and number in series from metadata
    tZero = float(np.genfromtxt(file,skip_header=len(wls)+25,skip_footer=10,delimiter=None,dtype=str,usecols=(3)))
    width = int(np.genfromtxt(file,skip_header=len(wls)+24,skip_footer=11,delimiter=None,dtype=str,usecols=(3)))
    step = int(np.genfromtxt(file,skip_header=len(wls)+26,skip_footer=9,delimiter=None,dtype=str,usecols=(4)))
    its = int(np.genfromtxt(file,skip_header=len(wls)+11,skip_footer=24,delimiter=None,dtype=str,usecols=(4)))
    cwl = float(np.genfromtxt(file,skip_header=len(wls)+31,skip_footer=4,delimiter=None,dtype=str,usecols=(2)))
    print('------------------------------------------------')
    print('       ',file[:-4])
    print('')
    print('        Start Time (ns):         ', tZero)
    print('        Gate Width (ns):         ', width)
    print('        Gate Step (ns):          ', step)
    print('        Number in Series:        ',its)
    print('        End Time (ns):           ', tZero+its*step)
    print('        Elapsed Time (ns):       ', (tZero+(its-1)*step)-tZero)
    print('        Central Wavlength (nm):  ', cwl)
    print('------------------------------------------------')
    
    #Define timepoints, adjust for iccd gate times - important not to adjust to zero to be able to compare where different decays start
    time = np.arange(-step,its*step,step) # -step because when we concatenate later, the first timepoint will be in the wavelength column
    #its*step means generated time will be before 'end time' by one step

    time = time+step*0.5

    time = np.array([x+tZero for x in time])
    print('Times:',time[1:])

    
    #Add time to make full array
    data1 = np.vstack([time,data])
    data1, trans_sens, trans_filter = ICCDcorrections(data1,cwl)
    if plot:
        for i in list(range(1,data.shape[1])):
            plt.plot(wls, data[:,i],'.',label=i)
            plt.yscale('log')
            #plt.ylim(100,data[:,2].max())
            plt.legend(ncol=3,fontsize='x-small')
        plt.show()
    
    return data1

#%% Fit gauss pump scatter - only use if pump scatter unavoidable

def gauss_scatter(data,subtract=False,plot=True):
    
    x = data[1:,0]
    #descattered = np.zeros((len(x),3))
    #descattered[:,0] = x
    for i in list(range(1,3)):
        yselect = data[1:,i]
        yselect[yselect<=0] = 1
        y = np.log10(yselect)
        # define  Gaussian function
        def gaussian(x, amp, cen, wid):
                return amp * np.exp(-(x-cen)**2 / (2*(wid**2)))
            
        # define the function to fit two overlapping Gaussians

        def double_gaussian(x,a1, c1, a2, b2, c2):
            return gaussian(x,a1, y.max(), x[y.argmax()], c1) + gaussian(x, a2, b2, c2)
        
        
        p0gaus = [y.max(),20,y.max()*.8,x[y.argmax()]-10,100]

                # fit the data with the function
        popt, pcov = curve_fit(double_gaussian, x, y,p0=p0gaus, maxfev=10000000)
        # plot the data and the fit
        if plot:
            plt.plot(x, 10**y, 'b.', label='data')
            plt.plot(x, 10**double_gaussian(x, *popt), 'r', label='fit')
            plt.legend()
            plt.show()
        
        if subtract == True:
            subbed = 10**(y - gaussian(x,popt[0],x[y.argmax()],popt[2]))
            plt.plot(x,subbed)
            plt.show()
            data[1:,i] = subbed
            
    return data


#%% PLdecay
from matplotlib import cm
from matplotlib.colors import Normalize

def PLdecay(data, wlmin=10, wlmax=-10, normalise=False, label='give a for loop', plot=True, yscale='log', smooth=False):
    #Find index closest to wavelength limits
    wlmin = np.abs(data[:,0]-wlmin).argmin()
    wlmax = np.abs(data[:,0]-wlmax).argmin()
    wls = data[wlmin:wlmax, 0]  # Select WLs
    time = data[0, 1:]  # Time axis
    len_time1 = len(time)
    
    PL = np.sum(data[wlmin:wlmax, 1:], axis=0)  # Create the signal we track, sum of PL (can be mean, same thing really)
    zero_vals = np.where(PL <= 0)[0]
    
    for z in zero_vals:
        PL[z] = 1
    pl = PL[PL.argmax():]  # Signal going to plot from max
    time = time[PL.argmax():]
    print('No. of data points cropped by max:', (len_time1 - len(time)))
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Normalize the colorscale to the range of indices
        norm = Normalize(vmin=0, vmax=data.shape[1] - 2)  # -2 because the first column is wavelengths and time axis starts from index 1
        colormap = cm.get_cmap('viridis')
        
        for i in range(data.shape[1] - 1):  # - 1 because we add one in next line to account for wavelengths
            signal = data[wlmin:wlmax, i + 1]  # +1 because first column is wavelengths
            signal[signal <= 0] = 1
            
            if normalise:
                signal = signal / signal.max()
           
            color = colormap(norm(i))
            ax1.plot(wls, signal, label=i, alpha = 0.5,color=color)  # Plot spectrum at second #timepoint, usually the first point
        
        ax1.set_xlabel('Wavelengths/ nm')
        ax1.set_ylabel('Counts')
        ax1.set_yscale('symlog', linthresh=100)
        ax1.legend()
        
        ax2.plot(time, pl, 'o', label=label)
        ax2.set_yscale(yscale)
        ax2.set_xlabel('Time/ ns')
        ax2.set_ylabel('Counts')
        ax2.legend()
        
        plt.suptitle(label)
        plt.show()
    
    decay = np.array([time, pl])
    
    return decay




#%%

def scaling(decays):
    mean_ratios = [1]
    for i in range(0,len(decays)-1):
        PL1 = decays[i]     #Selects the decay to interpolate 
        PL2 = decays[i+1]   # New decay to fit to
        interp_function = interpolate.interp1d(PL1[0],PL1[1],fill_value='extrapolate') #creates a function of first decay
        interp_vals  = interp_function(PL2[0][:3])                                     #interpolates the first 3 timepoints of 2nd decay
        ratio = np.array([0,0,0])
        for i2 in range(3):
            ratio[i2] = PL2[1][i2]/interp_vals[i2]  #finds ratio of real data points to interpolated data points to find scaling factor
        
        non_zero_values = ratio[ratio != 0]  # Filter out zero values
        mean_ratio = np.mean(non_zero_values)#mean of the ratio of the 3 or 2 points
        mean_ratios.append(mean_ratio) #appends the mean ratio to a list
        print(ratio)
        print('done stitch',i, f'mean={mean_ratio}')
    
    ratios = [mean_ratios[0]]

    for i in range(1, len(mean_ratios)):
        ratios.append(ratios[-1] * mean_ratios[i]) #cascades the ratios, as you will need to divide by each of previous for new curves

    return ratios


#%%

def plot_aligned(decays, ratios, save=False, filename=None, adjust=[1,1,1],xscale='linear'):
    """Aligns stitches decays and plots them

    Args:
        decays (list): Individual PL decay stitches
        ratios (list): Ratios between decay curve interpolated points
        save (bool, optional): Saves final decay curve. Defaults to False.
        filename (str, optional): File name of new final. Defaults to None.
        adjust (list, optional): Adjust ratios if the original calculation hasn't worked. Defaults to [1,1,1].

    Returns:
        _type_: _description_
    """
    decays2 = []
    for i in range(len(decays)):
        plt.plot(decays[i][0], decays[i][1]/ratios[i]/adjust[i], 'o', label=f'PL{i+1}')
        plt.legend()
        plt.yscale('log')
        
        new_decay = [decays[i][0] ,decays[i][1]/ratios[i]/adjust[i]] # make a new set of decays so that you can repeatedly call the function without saving new values
        decays2.append(new_decay)
    
    plt.show()
    
    master_decay = np.concatenate((decays2), axis=1)
    idx = np.argsort(master_decay[0])
    #print(idx)
    master_sorted = master_decay[:, idx]
    time = master_decay[0] - master_decay[0][0]
    signal = master_decay[1]
    if normalise:
        signal = signal/signal[0]
    plt.plot(time, signal, 'o')
    plt.yscale('log')
    plt.xscale(xscale)
    print(filename)
    
    if save:
        if filename is None:
            print('NO FILENAME SET - call filename or file will not save')
        else:
            # Create the 'Sorted' folder if it doesn't exist
            sorted_folder = 'Sorted'
            if not os.path.exists(sorted_folder):
                os.makedirs(sorted_folder)
            
            np.save(sorted_folder + '/' + filename + '.npy', master_sorted)
    
    return master_decay

    
#%% Spectrum

def PLspec(data,wlmin=1,wlmax=-1,time=[2]):
    """Plot PL Spectra

    Args:
        data (_type_): _description_
        wlmin (int, optional): _description_. Defaults to 1.
        wlmax (int, optional): _description_. Defaults to -1.
        time (list, optional): _description_. Defaults to [2].

    Returns:
        _type_: _description_
    """
    wls = data[wlmin:wlmax,0]
    ts = data[0,1:]
    signals = data[wlmin:wlmax,1:]
    PLSteps = []
    for i, t in enumerate(ts):
        step = signals[:,i]
        plt.plot(wls,step,label=t)
        #plt.legend()
        plt.yscale('linear')
        PLSteps.append(step)
    
    return signals, PLSteps

        #need to change this to select one or two times


        
    
    
    
    

    # if wlmin or wlmax != None:
    #     WLmin= wls.index(wlmin)
    #     WLmax = wls.index(wlmax)
        
    #     PL = np.mean(data[WLmin:WLmax,1:],axis=0)
    # PL = np.mean(data[:,1:],axis=0)
    # PL = PL[PL.argmax():]
    # time = time[PL.argmax()+1:]
    # plt.plot(time,PL,'o')
    # plt.yscale('log')
    
#%%
def calc_fluences(P=[1,5,10],A = 1, d = 500, wl = 400, di = 1605, 
                  f=1000,areaType='gauss',a=2500,b=1950,
                  filename=None, save=False):
    hv = (6.626E-34*2.998E8)/(wl*1e-9)                    #Calculate photon energy
    absorption = 1-10**(-A)                               #Calculate % of Photons absorbed
    if areaType == 'elliptical':                          #Calculate area in cm3 based on shape
        area = np.pi*((a)/2)*((b)/2)            #Ellipse is pi*rad1*rad2
    else:
        area = np.pi*(di/2)**2                       #Circle pi*r^2 in cm2
   
    nlist = []
    for p in P:
        n = (absorption*p*1e-6)/(area*f*hv*d*1e-7)        # in cm-3
        
        nlist.append(n)
    
    pflist = []
    for p in P:
        fl = (p*1e-6)/(area) #uW/cm-2
        
        pflist.append(fl)
        
    eflist = []
    for p in P:
        el = (p*1e3)/(area*f) #uW/cm-2*1000
        
        eflist.append(el)
    
    Parameters = ['Measured Power', 'Absorption', 'Spot Area/ cm^2', 'Sample Thickness/ nm', 'Pump Wavelength/ nm', 'Carrier Density/ cm^-3', 'Power Fluence/ uJcm-2', 'Energy Fluence/ nJcm-2']
    carrier_info = pd.DataFrame({'Parameters': Parameters, 'Values': [P, absorption, area, d, wl, nlist, pflist, eflist]})
    print(carrier_info)

    if save:
        if filename is None:
            print('NO FILENAME SET - call filename or file will not save')
        else:
            # Create the 'Sorted' folder if it doesn't exist
            sorted_folder = 'Sorted'
            if not os.path.exists(sorted_folder):
                os.makedirs(sorted_folder)

            # Generate the full file path
            filepath = os.path.join(sorted_folder, filename + '_iCCD_info.csv')
            carrier_info.to_csv(filepath, index=False)

    return nlist, pflist, eflist, carrier_info

#%% Spline fit

def spline_function(x, y, plot=False):
    def spline(x, a,b,c,d,e,h,i,j,k):
        return a + b*x + c*x**2 + d*x**3 + e*x**4 + h*x**5 + i*x**6 + j*x**7 + k*x**8
    popt, pcov = curve_fit(spline, x, np.log10(y), p0 = [1,1,1,1,1,1,1,1,1], maxfev = 10000000)
    spline_fit = 10**spline(x, *popt)
    if plot:
        plt.plot(x,y,'.',label = 'Data')
        plt.plot(x,spline_fit,label = 'Spline Fit')
        plt.yscale('log')
        plt.legend()
        plt.show()
    return spline_fit

#%% Differenttial Liftime vs Time


def diff_lifetime(PL, time, ideality =2, plot=False, title = '',xscale = 'linear'):
    time_in_ns = time*1e-9
    diff = (-np.gradient(np.log(PL),time_in_ns)/2)**-1
    if plot:
        plt.plot(time,diff,'.')
        plt.xlabel('Time (ns)')
        plt.ylabel(r'Differential lifetime $\tau_{diff}$ (s$^{-1}$)')
        plt.title('Differential lifetime against time for' + title)
        plt.yscale('log')
        plt.xscale(xscale)
        plt.show()
    return np.array((time,diff))

#%% QFLS Change

def QFLS(PL, ni = 1e4):
    k_B = 1.38e-23 #jK-1
    T = 300 #K
    e = 1.6e-19 #C
    QFLS = k_B*T*np.log(PL/ni**2)/e
    print(QFLS)
    return QFLS

#%% Differential Lifetime vs Wavelength

def QFLS_difflifetime(PL, time, ni = 1e5, ideality = 2, title = ''):
    time_in_ns = time*1e-9
    qfls = QFLS(PL, ni)
    diff = diff_lifetime(PL, time, ideality)[1]
    plt.plot(qfls,diff,'.')
    plt.xlabel('QFLS (eV)')
    plt.ylabel(r'Differential lifetime $\tau_{diff}$ (s$^{-1}$)')
    plt.yscale('log')
    plt.title('Differential lifetime against QFLS change for' + title)
    plt.show()
    #return np.array((QFLS,diff))

#%%

""" FROM BELOW ARE CURRENTLY UNUSED BUT COULD BE USEFUL IN FUTURE"""




#%% Despike - needs love

def despike(data,despike=2):
    
    # Define x and y
    x = data[1:,0]
    for i in range(7):
        y = data[1:,i]
    
        # plot original data
        plt.plot(x[1000:1500], y[1000:1500])
        plt.title('Original data')
        plt.show()
        
        # calculate difference between adjacent points
        diff = np.diff(y)
        
        # calculate median absolute deviation
        mad = np.median(np.abs(diff - np.median(diff)))
        
        # set threshold to 5 times the median absolute deviation
        threshold = despike * mad
        
        # find indices of spikes (points that deviate from neighboring points by more than threshold)
        spike_indices = np.where(np.abs(diff) > threshold)[0] + 1
        
        for idx in spike_indices:
            if idx > 1 and idx < len(y) - 2:
                y[idx] = (y[idx+2] + y[idx+1]+y[idx-1] + y[idx-2]) / 2
             
        # plot despiked data
        plt.plot(x[1000:1500], y[1000:1500])
        plt.title('Despiked data')
        plt.show()

#%% Eliminate pump scatter - only use if pump scatter unavoidable
def subtractpumpscatter(data,wl1=700,wl2=800, remove=False):
    wls = data[1:,0]
    for i in list(range(1,3)): #only need first few as should be gone after 2
        plt.plot(wls, data[1:,i])
        plt.axvspan(wl1,wl2,alpha=0.1,color='r')
    plt.show()
    if remove == True:
        WL1 = np.abs(wls - wl1).argmin()
        WL2 = np.abs(wls - wl2).argmin()
        #data = np.concatenate((data[:WL1,:],data[WL2:,:]),axis=0)
        data_slice1 = data[:WL1,:]
        data_slice2 = data[WL2:,:]
        data_concatenated = np.concatenate((data_slice1, data_slice2), axis=0)
        for i in list(range(1,data.shape[1])):
        

            plt.plot(data_concatenated[1:,0],data_concatenated[1:,i])
            plt.title('Cropped pump scatter')
    plt.show()
    return data_concatenated


#%% HeatMap
def PLheatmap(data,wlmin=20,wlmax=-20,tUnit= 'ns',title=None,save=False):
    x_labels = data[0,1:]-data[0,1]
    y_labels = data[wlmin:wlmax,0]
    d = data[wlmin:wlmax,1:]
    fig,ax = plt.subplots()
    a = ax.pcolormesh(x_labels, y_labels, d,cmap='plasma', linewidth=0,rasterized = True,shading='gouraud')
    ax.set_yscale('log')
    cbar = fig.colorbar(a,ax=ax)
    cbar.set_label('Counts')
    ax.set_ylabel('Wavelength (nm)')
    ax.set_xlabel('Time ('+tUnit+')')
    ax.set_title(title)
    if save == True:
        newfolder = 'heatmaps/'
        plt.savefig(newfolder+file[:-4]+'.svg',transparent=True)
