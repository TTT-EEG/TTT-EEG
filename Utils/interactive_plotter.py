import matplotlib.pyplot as plt
import numpy as np

class SnaptoCursor(object):
    '''
    interactive plot and select
    
    '''
    def __init__(self, ax, x): # , y
        self.ax = ax
        # self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.x = x
        # self.y = y
        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)
        self.clicks = []
        
    def mouseMove(self, event):

        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata

        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        # y = self.y[indx]
        # update the line positions
        # self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        self.txt.set_text('x=%d' % (x,))
        plt.draw()
        
    
    def onClick(self, event):
        x = event.xdata
        self.clicks.append(x)
    
    def getClicks(self, index):
        return self.clicks[index]

# t = np.arange(0.0, 1.0, 0.01)
# s = np.sin(2*2*np.pi*t)
# fig, ax = plt.subplots()

# cursor = SnaptoCursor(ax, t, s)

# cid = fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
# cid = fig.canvas.mpl_connect('button_press_event', cursor.onclick)

# ax.plot(t, s)
# plt.axis([0, 1, -1, 1])
# plt.show()
