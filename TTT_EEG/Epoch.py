# @author: Xuefei Wang
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import mne
import pandas as pd
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider, Button

from .AlignedEpoch import AlignedEpoch
from .utils.Interactive_ax import Interactive_ax


def _gfp(data):
    '''
    calculate GFP (Global Field Potential)

    Parameter
    ---------
    data: numpy 2d array of shape (n_sensors, n_times)

    Return
    ------
    gfp: numpy 1d array of shape (n_times, )
    '''
    return np.std(data,0)


def _smooth(data, points = 5):
    '''
    Average Smooth using a moving window, length = number of points

    Parameter
    ---------
    data: numpy 2d array of shape (times, channels)
        the data to be smoothed
    points: int, Default 5
        number of points within one window

    Return
    ------
    smoothed_data: numpy 2d array of shape (times, channels)

    '''
    kernel = np.ones(points)/points
    return np.convolve(data, kernel, mode='same')

class Epoch:
    '''
    Identify the ERP components from raw data.

    Parameter
    ---------
    data： numpy 2d array of shape (times, channels) or a string
        - the single subject averaged epoch data(or evoked data)
        - epo.fif filename
    poi_left: int | None
        beginning of region of interest, if not specified, please call narrowDownpoi() to interactively choose one
    poi_right: int | None
        end of region of interest, if not specified, please call narrowDownpoi() to interactively choose one
    times： numpy 1d array (times, )
        the time point array
    ch_names: list
        a list of channel names, in the format of ['Fp1', 'Fz', ...], can be accessed from epoch.ch_names from mne-python

    Return
    ------
    A Epoch object

    '''

    def __init__(self, data, times, ch_names, sfreq, events_list, events_dict):
        self.__data = data # (n_trials, n_sensors, n_times)
        self.__times = times
        self.__ch_names = ch_names
        self.__sfreq = sfreq
        self.__events_list = events_list
        self.__events_dict = events_dict
        self.__avg_data = np.mean(self.__data,0) # (n_sensors, n_times)
        #TODO: differentiate between conditions
        self.__baseline_dur = - int(self.__times[0])
        print(self.__baseline_dur)
        self.__gfp = _gfp(self.__avg_data)
        self.__num_trials = data.shape[0]

        self.__left_poi = None
        self.__right_poi = None
        self.__peak = None
    def getTimes(self):
        return self.__times

    def getOnset(self):
        return self.__onset

    def getOffset(self):
        return self.__offset

    def getPeak(self):
        return self.__peak

    def getSig(self):
        return self.__data

    def getCorr(self):
        return self.__corr

    def getGrad(self):
        return self.__grad

    def get_gfp(self):
        return self.__gfp


    def narrow_down_poi(self, interactive = True, poi_left = None, poi_right = None):
        '''
        Pre-select the period of interest (POI)
        interactive = True: click in a gui window, otherwise, poi_left/poi_right parameters are necessary
        Butterfly plot of raw data will be displayed and the click on the screen will be recorded as the time point of beginning and end.

        Parameter
        ---------
        interactive：bool, default True
            interactively choose or pre-specify
        poi_left, poi_right: int, default none
            necessary when interactive = False
            Period of interest (POI), in ms unit

        Return
        ------
        poi_left, poi_right: int
            Period of interest (POI), in ms unit
        '''

        #TODO: check arguments here

        max_val = np.max(self.__avg_data)
        min_val = np.min(self.__avg_data)
        
        min_time = self.__times[0]
        max_time = self.__times[-1]

        if(interactive):
            
            fig= plt.figure()
            fig.set_size_inches(12,6)
            ax = fig.add_axes([0.1, 0.3, 0.8, 0.65])
            
            ax.plot(self.__times, self.__avg_data.T)
            ax.set_xticks(self.__times[::100])
            ax.grid()

            ax1 = plt.axes([0.1, 0.15, 0.8, 0.03])
            ax2 = plt.axes([0.1, 0.2, 0.8, 0.03])

            slider_poi_left = Slider(ax1, 'POI left', min_time, max_time, valinit=min_time)
            slider_poi_right = Slider(ax2, 'POI right', min_time, max_time, valinit=max_time)
            shades = ax.fill_between(range(int(min_time), int(max_time)),min_val*1.1, max_val*1.1, color = 'green' ,alpha = 0.3)
            
            ax_text = plt.axes([0.4, 0.08, 0.1, 0.04])
            ax_text.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
            ax_text.spines['left'].set_visible(False)
            ax_text.spines['bottom'].set_visible(False)
            ax_text.spines['right'].set_visible(False)
            ax_text.spines['top'].set_visible(False)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            textstr = "None - None"
            texts = ax_text.text(0.5, 0.5, textstr, fontsize=14, verticalalignment='top', bbox=props)

            gui = interactive_ax(ax,ax_text, shades, texts, min_val, max_val, min_time, max_time)

            slider_poi_left.on_changed(gui.update_left)
            slider_poi_right.on_changed(gui.update_right)
            
            ax_button = plt.axes([0.8, 0.05, 0.1, 0.04])
            button = Button(ax_button, 'Confirm')
            button.on_clicked(gui.on_click) #TODO: seems not very responsive, need to move the mouse away

            plt.show()
            
            self.__left_poi, self.__right_poi = gui.return_poi()

        else:
            self.__left_poi, self.__right_poi = poi_left, poi_right
        return self.__left_poi, self.__right_poi


    def find_peak(self, order = 5):
        '''
        find the peak of ERP component

        Parameter
        ---------
        order: int, Default 2
            How many points on each side to use for the comparison to consider comparator(n, n+x) to be True.
         
            
        Return
        ------
        peak: int
            the time point of peak latency
        '''
        # plt.plot(self.__gfp)
        # plt.show()
        index = argrelmax(self.__gfp[self.__left_poi: self.__right_poi], order = order)[0] + self.__left_poi
        pairs = [[ind, self.__gfp[ind]] for ind in index]
        pairs = np.array(pairs)
        # for i in range(pairs.shape[0]):
        #     print("time point %d, gfp value %.3f"%(pairs[i,0],pairs[i,1]))

        maxv = np.max(pairs[:,1])
        # print(maxv)

        pair = pairs[pairs[:,1] == maxv]

        self.__peak = int(pair[0][0])
        
        return self.__peak

    def _cal_grad(self):
        '''calculate the Gradient Magnitude Matrix'''
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.signal import convolve2d
        corr = cosine_similarity(self.__avg_data.T)
        self.__corr = corr
        scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                       [-10+0j, 0+ 0j, +10 +0j],
                       [ -3+3j, 0+10j,  +3 +3j]])
        grad = convolve2d(corr.T, scharr, boundary='symm', mode='same')
        abs_grad = np.absolute(grad)
        
        norm_abs_grad = (abs_grad - np.min(abs_grad)) / (np.max(abs_grad) - np.min(abs_grad))
        self.__grad = norm_abs_grad


    def _find_onset_offset(self, strength, alpha = 0.3, beta = 0):
        '''helper function for finding onset and offset'''
        # print(self.__grad.shape)
        a = [self.__grad[self.__peak,j] - self.__grad[self.__peak, self.__peak] for j in range(self.__peak + 1 ,self.__right_poi)]
        a = np.array(a)
        a = _smooth(a)

        
        maxv_a = np.max(a)
        minv_a = np.min(a)
        t = argrelmax(a,order = strength)[0]
        t = [item for item in t if a[item] > alpha* maxv_a  + beta* minv_a]
#         print(t)
        t = t[0]
    #     print('delta', t)
        offset = t + self.__peak
        b = [self.__grad[self.__peak,j] - self.__grad[self.__peak, self.__peak] for j in range(self.__left_poi, self.__peak - 1)]
        b = np.array(b)
        b = _smooth(b)
        maxv_b = np.max(b)
        minv_b = np.min(b)
        t = argrelmax(b,order = strength)[0]
        t = [item for item in t if b[item] > alpha * maxv_b + beta * minv_b]
        t = t[-1]
        onset = t + self.__left_poi
        return onset, offset

        
    def find_onset_offset(self, alpha = 0.3, beta = 0):
        '''
        find onset and offset

        Parameter
        ---------
        None

        Return
        ------
        onset: int
            the time point of ERP onset
        offset: int
            the time point of ERP offset
        '''
        self._cal_grad() # calculate grad
        # specify peak-finding strength according to the sample rates, for 1000Hz, strength = 12, for 250Hz, strength = 3
        strength = int (len(self.__times) / (self.__times[-1] - self.__times[0] )  * 12)
        # if(self.__haspeak):
        self.__onset, self.__offset = self._find_onset_offset(strength, alpha, beta)
        # else:
            # self.__onset, self.__offset = self._findOnsetOffset_nopeak(strength,alpha, beta)
        return self.__onset, self.__offset


    def find_duration(self):
        return self.__offset - self.__onset

    def find_rise_speed(self):
        '''
        Calculate rise speed

        Parameter
        ---------
        None

        Return
        ------
        RS: float, (μV/ms)
            rise speed
        '''
        assert self.__peak != None
        assert self.__onset != None
        assert self.__offset != None
        RS = (self.__gfp[self.__peak] - self.__gfp[self.__onset]) / (self.__peak - self.__onset) 
        
        return RS
    
    def find_fall_speed(self):
        '''
        Calculate fall speed

        Parameter
        ---------
        None

        Return
        ------
        FS: float
            the fall speed
        '''
        assert self.__peak != None
        assert self.__onset != None
        assert self.__offset != None
        FS = (self.__gfp[self.__peak] - self.__gfp[self.__offset]) / (self.__offset - self.__peak)
        
        return FS

    def find_evolution(self):
        peak = self.find_peak(order = 5)
        onset, offset = self.find_onset_offset()
        duration = self.find_duration()
        rise_speed = self.find_rise_speed()
        fall_speed = self.find_fall_speed()
        return peak, onset, offset, duration, rise_speed, fall_speed


    def visualize_evolution(self, title = 'Epoch Result'):
        '''
        Visualize the result, with onset, offset, peak, gfp

        Parameter
        ---------
        title: string, default 'Epoch Result'
            the title for the figure

        Return
        ------
        fig,ax
        '''
        fig, ax = plt.subplots(1)
        fig.set_size_inches(8,6)
        ax.set_title(title)
        cax = ax.imshow(self.__grad, cmap = 'gist_gray_r')
        ax.plot(range(len(self.__grad)),range(len(self.__grad)), 'w')
        ax.set_xticks(np.arange(len(self.__grad),step = 100))
        ax.set_xticklabels(self.__times[0::100])
        ax.set_yticks(np.arange(len(self.__grad),step = 100))
        ax.set_yticklabels(self.__times[0::100])
        cbar = plt.colorbar(cax)
        
        ax1 = ax.twinx()
        ax1.plot(self.__gfp, 'c', linewidth = 2)
        ax1.plot(np.arange(self.__baseline_dur + self.__left_poi,self.__baseline_dur + self.__right_poi),self.__gfp[self.__baseline_dur + self.__left_poi:self.__baseline_dur + self.__right_poi],'r', linewidth = 2)
        # if(self.__haspeak):
        ax1.plot(self.__baseline_dur + self.__peak, self.__gfp[self.__baseline_dur + self.__peak], 'y*', markersize = 14)
        # else:
        #     pseudopeak = round((self.__left_poi + self.__right_poi)/2)
        #     ax1.plot(pseudopeak, self.__gfp[pseudopeak], 'y*', markersize = 14)
        
        ax1.plot(self.__baseline_dur + self.__onset, self.__gfp[self.__baseline_dur + self.__onset], 'go', markersize = 10)
        ax1.plot(self.__baseline_dur + self.__offset, self.__gfp[self.__baseline_dur + self.__offset], 'go', markersize = 10)
        ax1.set_axis_off()

        
        plt.show()
        #TODO: plot topography to check, and make this compatible with future all-components-at-one-time version
        return fig, ax

#     def align_single_trial(self):
#         '''
#         Align single trials using a template

#         Parameter
#         ---------
#         None

#         Return
#         ------

#         '''
        


    def to_AlignedEpoch(self):
        print(self.__data.shape) # (n_trials, n_sensors, n_times)
        template = self.__avg_data[:, self.__peak + self.__baseline_dur] 
        
        projected_curves = []
        for num in range(self.__num_trials): # for each trial
            single_trial = self.__data[num, :, :]
            dur_on = int(self.__onset - (self.__peak - self.__onset) * 0.5 + self.__baseline_dur)
            dur_off = int(self.__offset + (self.__offset - self.__peak) * 0.5 + self.__baseline_dur)
            for i in range(dur_on, dur_off): # for each timepoint
                projected_curves.append(np.dot(single_trial[:,i], template)/ np.linalg.norm(template))
        
        projected_curves = np.array(projected_curves).reshape(self.__num_trials, dur_off - dur_on)

        single_trial_peaks = np.nanargmax(projected_curves, 1) + dur_on
        single_trial_onsets = (single_trial_peaks - (self.__peak - self.__onset) * 1.2).astype(int)
        single_trial_offsets = (single_trial_peaks + (self.__offset - self.__peak) * 1.2).astype(int)

        all_onset = (self.__peak - (self.__peak - self.__onset) * 1.2).astype(int)
        all_offset = (self.__peak + (self.__offset - self.__peak) * 1.2).astype(int)

        # print(single_trial_peaks)

        return AlignedEpoch(self.__data, single_trial_peaks, single_trial_onsets, single_trial_offsets, 
            template, self.__times, self.__peak, all_onset, all_offset)