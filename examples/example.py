# @author: Xuefei Wang
import sys
# sys.path.insert(0,'../../')
from Release import Epoch
import numpy as np
# import EasyEEG
import mne
from Release.io import load

#TODO: make it jupyter

# load data from mne
file_path = './example-data-epo.fif'
mne_epoch = mne.read_epochs(file_path, verbose = False)
ttt_epoch = load.from_mne(mne_epoch)


# load data from EasyEEG
# EasyEEG_epoch = EasyEEG.
# ttt_epoch = io.from_EasyEEG() #TODO

gfp = ttt_epoch.get_gfp()
# print(gfp.shape)
# interactively choose POI or setting them manually
# left_poi, right_poi = ttt_epoch.narrow_down_poi(interactive = True)
poi_left = 150
poi_right = 230
ttt_epoch.narrow_down_poi(interactive = False, poi_left = poi_left, poi_right = poi_right) # TODO: write the pois according to the data

# temporal evolution of components
#   onset, offset, peak information
#   and result visualization
# call them all together, and return a text summary

peak, onset, offset, duration, rise_speed, fall_speed = ttt_epoch.find_evolution() # consider find evolution all at once, automatically
# or call them separately
# peak = ttt_epoch.find_peak(order = 5) #TODO: set parameters if necessary
# onset, offset = ttt_epoch.find_onset_offset() #TODO: make it the case that no matter which of this group is called, it won't matter
# duration = ttt_epoch.find_duration()
# rise_speed = ttt_epoch.find_rise_speed()
# fall_speed = ttt_epoch.find_fall_speed()
print("Let's make a summary: \n"
        "preselected window: \t%d, %d ms\n"
        "peak: \t%d ms\b\n"
        "onset, offset: \t%d, %d ms\n"
        "duration: \t%d ms\b\n"
        "rise speed, fall speed: \t%.3f, %.3f uV/ms\n"
        %(poi_left, poi_right, peak, onset, offset, duration, rise_speed, fall_speed))


# # make a visual summary
# fig, ax = ttt_epoch.visualize_evolution() #TODO: uncomment this

# # align single trials
ttt_aligned_epoch = ttt_epoch.to_AlignedEpoch()

# # plot single trial latency distribution
ttt_aligned_epoch.plot_latency_distribution()

# # plot gfp before and after alignment
ttt_aligned_epoch.plot_alignment_waveform()

SNR = ttt_aligned_epoch.find_SNR_boost()

# # plot topography before and after alignment
# ttt_aligned_epoch.plot_aligned_topo()



