import pickle
from . import Epoch
import numpy as np

from .myio import load

# with open("C:\EEG_N400\T_example.pkl", 'rb') as f:
#     T_example = pickle.load(f)

# data = T_example

# times = np.arange(-200, 801)
# T_epoch = io.load.from_nparray(data, times)
# T_epoch.narrow_down_poi(interactive = True)
# pk = T_epoch.find_peak()