import mne 
import numpy as np
import matplotlib.pyplot as plt
import pickle 


def getAsynchrony(data, onset, offset, peak):
    '''
    Parameter
    ---------
    data: numpy 2d array of shape(times, channels)
    onset: int
    offset: int
    peak: int
    
    Return
    ------
    asynchronyData: numpy 1d array (channels,)
        the time difference across channels
    asynchronySTD: int
        standard deviation across channels
    '''
    from scipy.signal import argrelextrema
    ntimes, nchannels = data.shape[0],data.shape[1]

    # channel peak might be a peak or a valley
    array = np.zeros((ntimes,nchannels))
    for i in range(nchannels):
        x = data[:,i]
        
        
        points = 20
        kernel = np.ones(points)/points
        xhat = np.convolve(x, kernel, mode='same')

        #xhat = x
        # for local maxima
        maxima = argrelextrema(xhat, np.greater, order = 12)

        # for local minima
        minima = argrelextrema(xhat, np.less, order = 12)

        array[maxima, i] = 1
        array[minima, i] = -1

    tick = np.empty((nchannels,2))

    # search each channel
    for i in range(nchannels):
        # # find the channel peak that is earlier than the component peak
        # peak_b = np.where(array[:peak + 1,i] == 1)[-1][-1]
        # valley_b = np.where(array[:peak + 1,i] == -1)[-1][-1]
        # tick[i,0] = max(peak_b,valley_b)

        # # find the channel peak that is later than the component peak
        # peak_a = np.where(array[peak - 1:, i] == 1)[0][0] + peak - 1
        # valley_a = np.where(array[peak - 1:, i] == -1)[0][0] + peak - 1
        # tick[i,1] = min(peak_a, valley_a)

        # find the channel peak that is earlier than the component peak
        peak_b, valley_b = np.NINF, np.NINF
        peaks_b = np.where(array[:peak + 1,i] == 1)[-1]
        valleys_b = np.where(array[:peak + 1,i] == -1)[-1]
        if peaks_b.shape != (0, ):
            peak_b = peaks_b[-1]
        if valleys_b.shape != (0,):
            valley_b = valleys_b[-1]
        tick[i,0] = max(peak_b,valley_b)

        # find the channel peak that is later than the component peak
        peak_a, valley_a =np.inf, np.inf
        peaks_a = np.where(array[peak - 1:, i] == 1)[0]
        if peaks_a.shape != (0,):
            peak_a = peaks_a[0] + peak - 1
        valleys_a = np.where(array[peak - 1:, i] == -1)[0]
        if valleys_a.shape !=(0,):
            valley_a = valleys_a[0] + peak - 1
        tick[i,1] = min(peak_a, valley_a)

        
        
    # choose the earlier or later peak, find the more plausible one
    temp = np.less(np.abs(tick - peak)[:,0],np.abs(tick - peak)[:,1])
    mask = np.array([temp*1, 1- temp*1]).T
    asynchronyData = tick - peak
    asynchronyData = np.sum(asynchronyData* mask, 1)

    # If any one of the channel peak time is earlier than onset time or later than offset time, we identify that as an outlier.
    # In most case, there's just no peak for this channel, so we set it to zero, and collect the number of outliers as an index of confidence.

    # find the ealier outliers, and set them to zero
    earlierOutliers = sum(asynchronyData < (onset - peak))
    asynchronyData[asynchronyData < (onset - peak) ] = 0

    # find the number of later outliers, and set them to zero
    lateOutliers = sum(asynchronyData > (offset - peak))
    asynchronyData[asynchronyData > (offset - peak) ] = 0


    asynchronySTD = np.std(asynchronyData) # in the unit of milliseconds

    return asynchronyData, asynchronySTD


