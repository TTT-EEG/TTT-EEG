
from scipy.optimize import brute
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
import pickle
from scipy.spatial.distance import cdist
from scipy.signal import resample
import numpy as np

# # Customize a Distance Function
# def dist_func(a,b):
#     return np.sum(np.abs(a-b))


def _dtw(x, y, dist_func, warp=1):
    '''
    dynamic time warping 
    
    Parameter
    ---------
    x,y: numpy 2d array of shape (times, ndim)
    dist_func: function handle
    warp: Default 1
    
    Return
    ------
    dist: distance matrix
    cost: cost matrix
    acc: accumulated distance matrix
    path
    
    '''
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist_func)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r - 1), j],
                             D0[i, min(j + k, c - 1)]]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)
    dist, cost, acc, path = D1[-1, -1] / sum(D1.shape), C, D1, path
    return dist, cost, acc, path 


def _traceback(D):
    '''trace back and find the path of dynamic time warping'''
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)

# Use Dynamic Time Warping to Test if Two Signals are the Same
def test_Same(reference, target,dist_func):
    dist, cost, acc, path = _dtw(reference, target, dist_func= dist_func)
              
    
#     plt.figure()
#     plt.imshow(acc.T, origin='lower', interpolation='nearest')
#     plt.plot(path[0], path[1], 'w')
#     plt.xlim((-0.5, acc.shape[0]-0.5))
#     plt.ylim((-0.5, acc.shape[1]-0.5))
#     plt.show()

#     x1 = reference[path[0]]
#     y1 = target[path[1]]
#     plt.figure()
#     plt.plot(x1, label = 'reference')
#     plt.plot(y1, label = 'target')
#     plt.legend()
#     plt.show()
    return dist

               
def _pad(reference, target):
    ''' padding '''
    maxLength = max(len(reference),len(target))
    reference = np.pad(reference, ((0,maxLength - len(reference)),(0,0)),'edge')
    target = np.pad(target, ((0,maxLength - len(target)),(0,0)),'edge')
#     print(reference.shape)
#     print(target.shape)
    return maxLength, reference, target

# # Scale Signals
# def scaleSig(reference, target, dist_func, rrange, stepSize):
#     '''
#     reference, target: (time, channels)
                                          
                                             
                                           
    
#     '''
#     a,b,c,d = rrange
                                                
                                         
    
#     # pad reference or target so that they are the same length, using edge padding mode
#     length, refernce, target = _pad(reference, target)

#     def func2min(x, dist_func = dist_func):
#         left, right = x
#         left = int(left * length)
#         right = int(right * length)
#         target_trunc = target[left: right]
#         scaled = resample(target_trunc, length)
#         return dist_func(reference, scaled)
    
#     rrange = (slice(a,b,stepSize),slice(c,d,stepSize),)
#     result = brute(func2min, rrange, full_output = True, finish = None)
#     left, right = result[0]
#     left = int(left * length)
#     right = int(right * length)
#     cost = result[1]
#     target_trunc = target[left: right]
#     scale_coef = len(target_trunc) / len(target)
#     scaled = resample(target_trunc, length)
    
#     # return
#     return scaled, cost, [left, right], scale_coef

# Shift Signals
def shiftSig(reference, target, dist_func, offset, bound, stepSize):
    '''shift signals and test the distance'''
    # pad reference or target so that they are the same length, using edge padding mode
    length, refernce, target = _pad(reference, target)
    
    shiftMode = 'nearest'
    # the function to minimize
    def func2min_shift(x, dist_func = dist_func):
        
        shiftl = np.append(x, np.array([0 for i in range(len(target.shape)-1)]))
        #print(shiftl)
        #print(target.shape)
        shifted = shift(target, shiftl, mode = shiftMode)
        return dist_func(reference, shifted)

    rrange = (slice(offset - bound, offset + bound, stepSize),)
    result = brute(func2min_shift, rrange, full_output = True, finish=None) # the real offset to shift
    # shiftLength, cost, grids, costs_for_grids
    shiftLength = result[0]
    if shiftLength < 0:
        left = 0
        right = len(target) + shiftLength
    else:
        right = len(target)
        left = 0 + shiftLength
    cost = result[1]
    shifted = shift(target, shiftLength, mode = shiftMode)
   
    # return
    return result
    

