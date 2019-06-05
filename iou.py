# -*- coding: utf-8 -*-
"""
Created on Sun May 26 12:42:48 2019

@author: dwubu
"""

def iou(min1_x, min1_y, max1_x, max1_y, min2_x, min2_y, max2_x, max2_y):

    #Find the coordinates of the intersection box
    mini_x = max(min1_x, min2_x)
    mini_y = max(min1_y, min2_y)
    maxi_x = min(max1_x, max2_x)
    maxi_y = min(max1_y, max2_y)
    
    #If the boxes don't overlap, return 0
    if (mini_x > maxi_x) or (mini_y > maxi_y):
        return 0
    
    intersection = (maxi_x - mini_x)*(maxi_y - mini_y)
    
    #Find the union
    union = (max1_x - min1_x)*(max1_y - min1_y) + (max2_x - min2_x)*(max2_y - min2_y) - intersection
    
    return intersection/union

