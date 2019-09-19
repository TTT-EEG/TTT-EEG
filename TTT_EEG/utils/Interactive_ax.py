# @author: Xuefei Wang
import numpy as np
import matplotlib.pyplot as plt

class Interactive_ax:
    def __init__(self,ax_butterfly, ax_text, shades, texts, min_val, max_val, min_time, max_time):
        self.left_poi = None
        self.right_poi = None
        self.left_val = min_time
        self.right_val = max_time
        self.ax_butterfly = ax_butterfly
        self.ax_text = ax_text
        self.texts = texts
        self.shades = shades
        self.min_val = min_val
        self.max_val = max_val

    def update_left(self, val):
        self.left_val = val
        if(self.left_val < self.right_val):
            color = 'green'
        else:
            color = 'red'

        self.ax_butterfly.collections.remove(self.shades)
        self.shades = self.ax_butterfly.fill_between(range(int(self.left_val), int(self.right_val)), self.min_val*1.1, self.max_val*1.1 ,alpha = 0.3, color = color)
        
    
    def update_right(self, val):
        if(self.left_val < self.right_val):
            color = 'green'
        else:
            color = 'red'

        self.right_val = val
        self.ax_butterfly.collections.remove(self.shades)
        self.shades = self.ax_butterfly.fill_between(range(int(self.left_val), int(self.right_val)), self.min_val*1.1, self.max_val*1.1 ,alpha = 0.3, color = color)
        
    def on_click(self,event):
        #TODO: check to make sure left is smaller than right, raise error 
        self.left_poi = int(self.left_val)
        self.right_poi = int(self.right_val)

        # display numbers
        textstr = "%d - %d"%(self.left_poi, self.right_poi)
        self.texts.set_text(textstr)

    def return_poi(self):
        return self.left_poi, self.right_poi