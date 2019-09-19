# @author: Xuefei Wang

import numpy as np
import matplotlib.pyplot as plt

class AlignedEpoch:
    def __init__(self, data, pk_list, on_list, off_list, template, times, peak, all_onset, all_offset):
        self.__data = data
        self.__pk_list = pk_list
        
        self.__on_list = on_list
        self.__off_list = off_list
        self.__template = template
        self.__times = times
        self.__baseline_dur = int(-times[0])
        self.__peak = peak
        self.__all_onset = all_onset
        self.__all_offset = all_offset

        self.__pk_list_relative = self.__pk_list - self.__peak - self.__baseline_dur

        align = []
        for num, (on, off) in enumerate(zip(on_list, off_list)):
            # print(num, on, off)
            if on < 0:
                temp = self.__data[num,:,0:off]
                temp = np.pad(temp, ((0,0), (-on+1,0)), 'constant', constant_values=np.nan)
                # print('before pad %d'%(-on+1))
            else:
                temp = self.__data[num,:,on:off]
            if off > self.__times[-1]-1:                
                temp = np.pad(temp, ((0,0), (0,off-self.__times[-1]+1)), 'constant', constant_values = np.nan)
                # print('after pad %d'%(off-self.__times[-1]+1))
        
            # print('*** Now shape is (%d, %d)' %(temp.shape))
            align.append(temp.tolist())
        align = np.array(align)
        self.__aligned = align # (n_trials, n_sensors, n_times)
        print(align.shape)
        

        
    def plot_latency_distribution(self):
        fig, ax = plt.subplots()
        plt.hist(self.__pk_list_relative)
        plt.title('Single trial peak latency')
        plt.show()
        #TODO: maybe fit a curve, and return params if necessary
        return fig, ax
    
    # plot gfp before and after alignment
    def plot_alignment_waveform(self):
        fig, ax = plt.subplots(2,1, sharex = True, sharey = True)
        ax[0].plot(self.__times[self.__all_onset:self.__all_offset], np.std(np.nanmean(self.__data,0), 0)[self.__all_onset:self.__all_offset])
        ax[0].set_title('before alignment')
        ax[1].plot(self.__times[self.__all_onset:self.__all_offset], np.std(np.nanmean(self.__aligned,0), 0))
        ax[1].set_title('after alignment')
        plt.show() #TODO: check this, seems SNR too large
        #TODO: plot topo
        return fig, ax
   
    def find_SNR_boost(self,):
        snr_boost = np.max(np.std(np.nanmean(self.__aligned,0), 0)) / np.max(np.std(np.nanmean(self.__data,0), 0)[self.__all_onset:self.__all_offset]) 
        return snr_boost


    def plot_aligned_topo(self,):
        #TODO: also make sure this is compatible with future all-components-at-once version
        pass
