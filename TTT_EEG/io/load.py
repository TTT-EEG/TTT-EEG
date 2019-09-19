# @author: Xuefei Wang
from .. import Epoch

def from_mne(mne_epoch):
    # convert from mne to TTT
    data = mne_epoch.get_data() * 10**6 # (n_trials, n_sensors, n_times), convert to unit of 'uV'
    times = mne_epoch.times *10**3 # (n_times,), convert to unit of 'ms'
    ch_names = mne_epoch.ch_names # channel names
    sfreq = mne_epoch.info['sfreq'] # sample frequency
    
    events_list = mne_epoch.events[:,2] #(n_trial, )
    events_dict = mne_epoch.event_id # dict, cond:number mappings

    ttt_epoch = Epoch.Epoch(data, times, ch_names, sfreq, events_list, events_dict)

    return(ttt_epoch)


def from_EasyEEG():
    # convert from EasyEEG to TTT

    ttt_epoch = Epoch()
    return (ttt_epoch)

def from_nparray(data, times, ch_names = None, sfreq = None, events_list = None, events_dict = None):
    ttt_epoch = Epoch(data, times, ch_names, sfreq, events_list, events_dict)
    return (ttt_epoch)