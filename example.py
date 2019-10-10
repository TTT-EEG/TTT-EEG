import numpy as np
import mne
from TTT_EEG import Epoch
from TTT_EEG.io import load
# import EasyEEG


''' Load data from mne or easyeeg
''' 

file_path = './example-data-epo.fif'
mne_epoch = mne.read_epochs(file_path, verbose = False)
ttt_epoch = load.from_mne(mne_epoch)
# EasyEEG_epoch = EasyEEG.io.load(file_path)
# ttt_epoch = io.from_EasyEEG(file_path)


''' Find epoch within the poi
        1. when interactive = True, a window will pop up, 
           please choose the poi window with the two sliders.
        2. when interactive = False, please specify the 
           pre-determined pois(`poi_left` and `poi_right`).
'''

# poi_right, poi_left = ttt_epoch.narrow_down_poi(interactive = True)
poi_left, poi_right = 150, 230
ttt_epoch.narrow_down_poi(interactive = False, poi_left = poi_left, poi_right = poi_right)


''' Find epoch indices and plot result
'''

peak, onset, offset, duration, rise_speed, fall_speed = ttt_epoch.find_evolution() 
# peak = ttt_epoch.find_peak()
# onset, offset = ttt_epoch.find_onset_offset()
# duration = ttt_epoch.find_duration()
# rise_speed = ttt_epoch.find_rise_speed()
# fall_speed = ttt_epoch.find_fall_speed()
print("Let's make a summary: \n"
        "preselected window: \t%d, %d ms\n"
        "peak: \t%d ms\b\n"
        "onset, offset: \t%d, %d ms\n"
        "duration: \t%d ms\b\n"
        "rise speed, fall speed: \t%.3f, %.3f uV/s\n" # TODO: change this from ms to s
        %(poi_left, poi_right, peak, onset, offset, duration, rise_speed, fall_speed))

## make a visual summary
# fig, ax = ttt_epoch.visualize_evolution()

''' Align single trials and plot result
'''

ttt_aligned_epoch = ttt_epoch.to_AlignedEpoch()

# ttt_aligned_epoch.plot_latency_distribution()
# ttt_aligned_epoch.plot_alignment_waveform()
ttt_aligned_epoch.plot_aligned_topo()
plt.show() # TODO: fix this

SNR_boost = ttt_aligned_epoch.find_SNR_boost()
print('SNR boost after alignment: %.3f'%(SNR_boost))