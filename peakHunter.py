# @author: Xuefei Wang
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import convolve2d
from Utils.interactive_plotter import SnaptoCursor
from Utils.asynchrony_getter import getAsynchrony
from Utils.asynchrony_plotter import plotAsynchrony
import mne
import pandas as pd
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Chinese display in plt figures
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False

def _gfp(data):
    '''
    calculate GFP (Global Field Potential)

    Parameter
    ---------
    data: numpy 2d array of shape (times, channels)

    Return
    ------
    gfp: numpy 1d array of shape (times, )
    '''
    return np.std(data,1)

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

class peakHunter:
    '''
    Identify the ERP components from raw data.

    Parameter
    ---------
    data： numpy 2d array of shape (times, channels) or a string
        - the single subject averaged epoch data(or evoked data)
        - epo.fif filename
    roi1: int | None
        beginning of region of interest, if not specified, please call narrowDownROI() to interactively choose one
    roi2: int | None
        end of region of interest, if not specified, please call narrowDownROI() to interactively choose one
    raw: boolean
        whether it's raw data or an epoch file
    times： numpy 1d array (times, )
        the time point array
    ch_names: list
        a list of channel names, in the format of ['Fp1', 'Fz', ...], can be accessed from epoch.ch_names from mne-python

    Return
    ------
    A peakhunter object

    '''

    def __init__(self, data, roi1 = 0, roi2 = None, haspeak = True, raw = True,times = None, ch_names = None):

        if raw == False:
            assert '-epo.fif' in data
            avg = mne.read_epochs(data).average()
            self.__sig = avg.data.T
            self.__times = avg.times
        else:
            assert any(times) != None
            self.__sig = data
            self.__times = times

        self.__sig = self.__sig * 10 ** 6 # in the unit of mV

        self.__gfp = _gfp(self.__sig)
        self.__roi1 = roi1
        if roi2 == None:
            self.__roi2 = len(self.__sig)
        else:
            self.__roi2 = roi2

        self.__haspeak = haspeak
        self.__peak = None
        self.__onset = None
        self.__offset = None
        self.__risefallratio = None
        self.__corr = None
        self.__grad = None
        self.__riseArea = None
        self.__fallArea = None
        self.__allArea = None
        self.__ch_names = ch_names
        self.__baselineTime = times[0]


    def getTimes(self):
        return self.__times

    def getOnset(self):
        return self.__onset + self.__baselineTime

    def getOffset(self):
        return self.__offset + self.__baselineTime

    def getPeak(self):
        return self.__peak + self.__baselineTime

    def getSig(self):
        return self.__sig

    def getCorr(self):
        return self.__corr

    def getGrad(self):
        return self.__grad

    def getGfp(self):
        return self.__gfp

    def getChnames(self):
        return self.__ch_names


    def getERP(self, mode):
        '''
        get the raw data, left and right end of the ERP component

        Parameter
        ---------
        mode: string, 'all' | 'rise' | 'fall', Default: 'all'

        Return
        ------
        erp: numpy 2d array of shape(times, channels)
            raw data of the ERP component
        left: int
            the left beginning time point
        right: int
            the right beginning time point

        '''
        if mode == 'rise':
            left = self.__onset
            right = self.__peak
            erp = self.__sig[self.__onset: self.__peak,:]
        elif mode == 'fall':
            left = self.__peak
            right = self.__offset
            erp = self.__sig[self.__peak: self.__offset,:]
        elif mode == 'all':
            left = self.__onset
            right = self.__offset
            erp = self.__sig[self.__onset: self.__offset, :]
        else:
            left = None
            right = None
            erp = None
        return erp,left, right

    def narrowDownROI(self):
        '''
        Interactively choose the region of interest.
        Butterfly plot of raw data will be displayed and the click on the screen will be recorded as the time point of beginning and end.

        Parameter
        ---------
        None

        Return
        ------
        roi1, roi2: int
            numbers of chosen ROIs
        '''
        fig, ax = plt.subplots(1)
        fig.set_size_inches(12,6)
        ax.plot(self.__sig)
        ax.set_xticks(range(0, len(self.__sig), int(len(self.__sig)/20 )))
        ax.grid()
        # ax.legend()
        cursor = SnaptoCursor(ax, range(0,len(self.__gfp)))
        cid = fig.canvas.mpl_connect('motion_notify_event', cursor.mouseMove)
        cid = fig.canvas.mpl_connect('button_press_event', cursor.onClick)
        plt.show()
        self.__roi1 = int(cursor.getClicks(-2))
        self.__roi2 = int(cursor.getClicks(-1))
        print('selected ROI is: (%d, %d)' %(self.__roi1, self.__roi2, ))

        return self.__roi1, self.__roi2

    def findpeak(self, order = 2):
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
        # print(self.__gfp)
        # print(self.__roi1)
        # print(self.__roi2)
        
        index = argrelmax(self.__gfp[self.__roi1: self.__roi2], order = order)[0] + self.__roi1
        pairs = [[ind, self.__gfp[ind]] for ind in index]
        pairs = np.array(pairs)

        maxv = np.max(pairs[:,1])
        # print(maxv)

        pair = pairs[pairs[:,1] == maxv]
        self.__peak = int(pair[0][0])
        
        return self.__peak

    def _calGrad(self):
        '''calculate the Gradient Magnitude Matrix'''
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.signal import convolve2d
        corr = cosine_similarity(self.__sig)
        self.__corr = corr
        scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                       [-10+0j, 0+ 0j, +10 +0j],
                       [ -3+3j, 0+10j,  +3 +3j]])
        grad = convolve2d(corr, scharr, boundary='symm', mode='same')
        self.__grad = np.absolute(grad)

    def _findOnsetOffset(self, strength, alpha = 0.3, beta = 0):
        '''helper function for finding onset and offset'''
        a = [self.__grad[self.__peak,j] - self.__grad[self.__peak, self.__peak] for j in range(self.__peak + 1 ,self.__roi2)]
        a = np.array(a)
        a = _smooth(a)
    #     print('self.__peak',self.__peak)

        plt.figure()
        # plt.plot(a)
        
        maxv_a = np.max(a)
        minv_a = np.min(a)
        t = argrelmax(a,order = strength)[0]
        t = [item for item in t if a[item] > alpha* maxv_a  + beta* minv_a]
#         print(t)
        t = t[0]
    #     print('delta', t)
        offset = t + self.__peak
        b = [self.__grad[self.__peak,j] - self.__grad[self.__peak, self.__peak] for j in range(self.__roi1, self.__peak - 1)]
        b = np.array(b)
        b = _smooth(b)
        maxv_b = np.max(b)
        minv_b = np.min(b)
        t = argrelmax(b,order = strength)[0]
        t = [item for item in t if b[item] > alpha * maxv_b + beta * minv_b]
        t = t[-1]
        onset = t + self.__roi1
        return onset, offset

        
        
        
    def _findOnsetOffset_nopeak(self, strength,alpha = 0.3, beta = 0):
        '''helper function for finding onset and offset when there's no GFP peak'''
        # latter half
        pseudopeak = round((self.__roi1 + self.__roi2)/2) # if there's no peak, choose the middle of roi1 and roi2 as a pseudopeak
        
        a = [self.__grad[pseudopeak,j] - self.__grad[pseudopeak, pseudopeak] for j in range(pseudopeak + 1 ,self.__roi2)]
        a = np.array(a)
        a = _smooth(a)
        # plt.figure()
        # plt.plot(a)
        # plt.plot(a_smooth)
        
        maxv_a = np.max(a)
        minv_a = np.min(a)
        t = argrelmax(a,order = strength)[0]
        t = [item for item in t if a[item] > alpha* maxv_a  + beta * minv_a]

        t = t[0]
        offset = t + pseudopeak
        
        # early half
        b = [self.__grad[pseudopeak,j] - self.__grad[pseudopeak, pseudopeak] for j in range(self.__roi1, pseudopeak - 1)]
        b = np.array(b)
        b = _smooth(b)
        maxv_b = np.max(b)
        minv_b = np.min(b)
        t = argrelmax(b,order = strength)[0]
        t = [item for item in t if b[item] > alpha* maxv_b + beta * minv_b]
        t = t[-1]
        onset = t + self.__roi1
        
        return onset, offset
        
    def findOnsetOffset(self, alpha = 0.3, beta = 0):
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
        self._calGrad() # calculate grad
        # specify peak-finding strength according to the sample rates, for 1000Hz, strength = 12, for 250Hz, strength = 3
        strength = int (len(self.__times) / (self.__times[-1] - self.__times[0] )  * 12)
        if(self.__haspeak):
            self.__onset, self.__offset = self._findOnsetOffset(strength, alpha, beta )
        else:
            self.__onset, self.__offset = self._findOnsetOffset_nopeak(strength,alpha, beta)
        return self.__onset, self.__offset

    def calRS(self):
        '''
        Calculate the rise-fall ratio, to determine skewness

        Parameter
        ---------
        None

        Return
        ------
        ratio: float
            the rise-speed, delta-gfp / delta-time
        '''
        assert self.__peak != None
        assert self.__onset != None
        assert self.__offset != None
        ratio = (self.__gfp[self.__peak] - self.__gfp[self.__onset]) / (self.__peak - self.__onset) 
        
        return ratio
    
    def calFS(self):
        '''
        Calculate the rise-fall ratio, to determine skewness

        Parameter
        ---------
        None

        Return
        ------
        ratio: float
            the rise-speed, delta-gfp / delta-time
        '''
        assert self.__peak != None
        assert self.__onset != None
        assert self.__offset != None
        ratio = (self.__gfp[self.__peak] - self.__gfp[self.__offset]) / (- self.__peak + self.__offset) 
        
        return ratio

    def calRFArea(self):
        '''
        calculate the areas of rise edge, fall edge, and whole component

        Parameter
        ---------
        None

        Return
        ------
        riseArea
        fallArea
        allArea
        '''
        self.__riseArea = np.sum(self.__gfp[self.__onset: self.__peak])
        self.__fallArea = np.sum(self.__gfp[self.__peak: self.__offset])
        self.__allArea = self.__riseArea + self.__fallArea
        return self.__riseArea, self.__fallArea, self.__allArea

    def visualizeResult(self, title = 'peakHunter Result'):
        '''
        Visualize the result, with onset, offset, peak, gfp

        Parameter
        ---------
        title: string, default 'peakHunter Result'
            the title for the figure

        Return
        ------
        fig,ax
        '''
        fig, ax = plt.subplots(1)
        fig.set_size_inches(8,8)
        ax.set_title(title)
        ax.imshow(1 - self.__grad, cmap = 'gist_gray')
        ax.plot(range(len(self.__grad)),range(len(self.__grad)), 'w')
        ax.set_xticks(np.arange(len(self.__grad),step = 100))
        ax.set_xticklabels(self.__times[0::100])
        ax.set_yticks(np.arange(len(self.__grad),step = 100))
        ax.set_yticklabels(self.__times[0::100])

        # ax.imshow(self.__corr, cmap = 'gist_gray')
        # ax.plot(range(len(self.__corr)),range(len(self.__corr)), 'w')
        # ax.set_axis_off()
        ax1 = ax.twinx()
        ax1.plot(self.__gfp, 'c', linewidth = 2)
        ax1.plot(np.arange(self.__roi1,self.__roi2),self.__gfp[self.__roi1:self.__roi2],'r', linewidth = 2)
        if(self.__haspeak):
            ax1.plot(self.__peak, self.__gfp[self.__peak], 'y*', markersize = 14)
        else:
            pseudopeak = round((self.__roi1 + self.__roi2)/2)
            ax1.plot(pseudopeak, self.__gfp[pseudopeak], 'y*', markersize = 14)
        
        ax1.plot(self.__onset, self.__gfp[self.__onset], 'go', markersize = 10)
        ax1.plot(self.__offset, self.__gfp[self.__offset], 'go', markersize = 10)
        ax1.set_axis_off()
        # plt.show()


        # fig2, ax2 = plt.subplots()
        # fig2.set_size_inches(8,4)

        # rawHandle = ax2.plot(self.__times, self.__sig * 10**6, color = 'c', linewidth = 1, label = "Raw Data")
        # peakHandle = ax2.axvline(x = self.__peak - 200, color = 'y', label = "Peak")
        # onsetHandle = ax2.axvline(x = self.__onset - 200, color = 'g', label = "Onset")
        # offsetHandle = ax2.axvline(x = self.__offset - 200, color = 'g', label = "Offset")
        # plt.legend(handles =[peakHandle, onsetHandle, offsetHandle] )
        # ax2.set_title("Butterfly Plot of Component")
        plt.show()
        return fig, ax

    def visualizeAsynchrony(self, montage_kind):
        '''
        visualize asynchrony of the ERP component

        Parameter
        ---------
        montage_kind: string
            the name of montage used in the experiment

        Return
        ------
        fig, ax

        Valid 'montage_kind' arguments:
        ===================   =====================================================
        Kind                  Description
        ===================   =====================================================
        standard_1005         Electrodes are named and positioned according to the
                              international 10-05 system (343+3 locations)
        standard_1020         Electrodes are named and positioned according to the
                              international 10-20 system (94+3 locations)
        standard_alphabetic   Electrodes are named with LETTER-NUMBER combinations
                              (A1, B2, F4, ...) (65+3 locations)
        standard_postfixed    Electrodes are named according to the international
                              10-20 system using postfixes for intermediate
                              positions (100+3 locations)
        standard_prefixed     Electrodes are named according to the international
                              10-20 system using prefixes for intermediate
                              positions (74+3 locations)
        standard_primed       Electrodes are named according to the international
                              10-20 system using prime marks (' and '') for
                              intermediate positions (100+3 locations)

        biosemi16             BioSemi cap with 16 electrodes (16+3 locations)
        biosemi32             BioSemi cap with 32 electrodes (32+3 locations)
        biosemi64             BioSemi cap with 64 electrodes (64+3 locations)
        biosemi128            BioSemi cap with 128 electrodes (128+3 locations)
        biosemi160            BioSemi cap with 160 electrodes (160+3 locations)
        biosemi256            BioSemi cap with 256 electrodes (256+3 locations)

        easycap-M1            EasyCap with 10-05 electrode names (74 locations)
        easycap-M10           EasyCap with numbered electrodes (61 locations)

        EGI_256               Geodesic Sensor Net (256 locations)

        GSN-HydroCel-32       HydroCel Geodesic Sensor Net and Cz (33+3 locations)
        GSN-HydroCel-64_1.0   HydroCel Geodesic Sensor Net (64+3 locations)
        GSN-HydroCel-65_1.0   HydroCel Geodesic Sensor Net and Cz (65+3 locations)
        GSN-HydroCel-128      HydroCel Geodesic Sensor Net (128+3 locations)
        GSN-HydroCel-129      HydroCel Geodesic Sensor Net and Cz (129+3 locations)
        GSN-HydroCel-256      HydroCel Geodesic Sensor Net (256+3 locations)
        GSN-HydroCel-257      HydroCel Geodesic Sensor Net and Cz (257+3 locations)

        mgh60                 The (older) 60-channel cap used at
                              MGH (60+3 locations)
        mgh70                 The (newer) 70-channel BrainVision cap used at
                              MGH (70+3 locations)
        ===================   =====================================================

        '''
        asynchronyData, _ = getAsynchrony(self.__sig, self.__onset, self.__offset, self.__peak)
        fig, ax = plotAsynchrony(asynchronyData, self.__ch_names, montage_kind)
        plt.show()
        return fig, ax

#     def alignSingleTrial(self):
        #TODO: implement based on the following codes
        # var = hunter.alignSingleTrials()
        # print(“=== Temporal Variance ===\n\
        #     %.3f”%var)
        # hunter.plotAlignResult()

        # from numpy import linalg
        #
        # tt = []
        # for num in range(len(self.__sig)): # for each trial
        #     st = broad[num,:,:].T
        #     curve = []
        #     for i in range(Dur):
        #         curve.append(np.dot(st[i,:], template)/ linalg.norm(template))
        #
        #     tt.append(curve)
        # tt = np.array(tt).T
        #
        # plt.hist(np.nanargmax(tt, 0))
        # plt.hist(np.nanargmax(tt, 0))
        # plt.title('Broad but Same Waveform')
        # # %%
        # pk_list = np.nanargmax(tt,0)
        # on_list = pk_list - 20
        # off_list = pk_list + 20
        # # %%
        # broad_align = []
        # for num, (on, off) in enumerate(zip(on_list, off_list)):
        #     print(num, on,off)
        #
        #     if on < 0:
        #         temp = broad[num,:,0:off]
        #         temp = np.pad(temp, ((0,0), (-on+1,0)), 'constant', constant_values=np.nan)
        #         print('before pad %d'%(-on+1))
        #     if off > Dur-1:
        #         temp = broad[num,:,on:off]
        #         temp = np.pad(temp, ((0,0), (0,off-Dur+1)), 'constant', constant_values = np.nan)
        #         print('after pad %d'%(off-Dur+1))
        #
        #     print('*** Now shape is (%d, %d)' %(temp.shape))
        #     broad_align.append(temp.tolist())
        # # %%
        # broad_align = np.array(broad_align)
        # # %%
        # plt.plot(np.mean(broad_align,0)[0])
