import pickle
from TTT_EEG import Epoch
import numpy as np
import matplotlib.pyplot as plt
from TTT_EEG.io import load

with open("C:\EEG_N400\T_example.pkl", 'rb') as f:
    T_example = pickle.load(f)

data = T_example
times = np.arange(-200, 801)
T_epoch = load.from_nparray(data, times)

data_points = []
for i in range(5):
    left, right = T_epoch.narrow_down_poi(interactive = True)
    pk = T_epoch.find_peak()

    onset, offset  = T_epoch.find_onset_offset()
    data_points.append([left, right, pk, onset, offset])

print(data_points)
with open('N400_data_points,pkl', 'wb') as f:
    pickle.dump(data_points, f)
with open('N400_data_points,pkl', 'rb') as f:
    data_points = pickle.load(f)

gfp = np.std(np.mean(data,0),0)*10**6
fig, ax = plt.subplots()

ax.plot(times, gfp, 'r', linewidth=3, alpha=0.4)
for left, right, pk, onset, offset in data_points:
    ax.plot(pk, gfp[pk+200], 'y*')
    ax.plot(onset, gfp[onset+200], 'bo')
    ax.plot(offset, gfp[offset+200], 'bo')
plt.xticks(np.arange(-200,801,50))
plt.grid()
# axx = ax.twinx()
# axx.plot(times, np.mean(data*10**6,0).T, 'k')
plt.show()