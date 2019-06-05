# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:46:44 2019

@author: dwubu
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def hyperparam_heatmap():
    mAPs = np.array([[0.496, 0.636, 0.489], 
                     [0.570, 0.620, 0.538], 
                     [6.57e-4, 2.71e-4, 1.54e-5]])
    
    ARs = np.array([[0.365,0.353,0.337],
                    [0.325,0.370,0.34],
                    [2.91e-2,1.2e-2,7.85e-3]])
    
    ax = sns.heatmap(mAPs, annot=True, xticklabels=[1e-4, 1e-3, 1e-2], yticklabels=[4e-4, 4e-3, 4e-2])
    plt.title("SSD MobileNet V2 Hyperparameter Tuning: mAP")
    plt.xlabel("Regularization Strength")
    plt.ylabel("Learning Rate")
    plt.show()

if __name__ == "__main__":
    hyperparam_heatmap()