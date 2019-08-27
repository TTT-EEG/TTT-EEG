import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from .layout_getter import getLayout


def plotLayout(montage_kind, ch_names):
    '''
    scatter plot of the layout of all the sensors

    Parameter
    ---------
    montage_kind: str
    ch_names: list
        channel names

    Return
    ------
    fig, ax
    '''
    layout = getLayout(montage_kind, ch_names)
    x,y = layout['x'].tolist(), layout['y'].tolist()
    name = layout['name']
    fig, ax = plt.subplots()
    fig.set_size_inches(5,5)
    ax.scatter(x,y)
    for i in range(len(x)):
        ax.annotate(name[i], xy = (x[i],y[i]))
    ax.set_title('Cap Layout and Channel Positions')
    ax.axis('off')
    return fig, ax

def plotTopo(montage_kind, ch_names, data, times):
    '''
    plot topomaps

    Parameter
    ---------
    montage_kind: str
    ch_names: list
        list of channel names
    data: numpy 2d array of shape (times, channels)
    times: numpy 1d array of shape (numOfMaps,)

    Return
    ------
    fig, ax

    Notes
    -----
    Examples of usage:
    _ = plotTopo(layout, data1[0] ,times = np.arange(200,600,50))
    _ = plotTopo(layout, data2[0] ,times = np.arange(200,600,50))

    '''
    title = " "
    layout = getLayout(montage_kind, ch_names)
    x,y = layout['x'].tolist(), layout['y'].tolist()
    name = layout['name']
    z = data.T
    vmax = np.max(z)
    vmin = np.min(z)
    from scipy.interpolate import griddata
    fig, ax = plt.subplots(1, len(times))
    fig.set_size_inches(2*len(times),3)
    fig.suptitle("Topography " + title)
    fig.subplots_adjust(top=0.8)
    # define grid.
    xi = np.linspace(-2.1,2.1,100)
    yi = np.linspace(-2.1,2.1,100)
    for i in range(len(times)):
        # grid the data.
        #print('points: ' + len(x) + ' ' + len(y))
        #print('values: ' + len(z[:,times[i]]))

        zi = griddata((x, y), z[:,times[i]], (xi[None,:], yi[:,None]), method='linear')
        # contour the gridded data, plotting dots at the randomly spaced data points.
        ax[i].axis('off')
        CS = ax[i].contour(xi,yi,zi,5,linewidths=0.5,colors='k')
        CS = ax[i].contourf(xi,yi,zi,15, vmin = vmin, vmax = vmax, cmap = 'RdBu_r',  norm=plt.Normalize(vmin, vmax))
        ax[i].set_title(str(times[i]))
        # plot data points.
        #plt.scatter(x,y,marker='o',c='b',s=5)
        plt.xlim(-2,2)
        plt.ylim(-2,2)
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.8)
    cax = plt.axes([0.92, 0.1, 0.01, 0.8])
    plt.colorbar(CS,cax=cax)

    return fig, ax
