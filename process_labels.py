# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:38:09 2019

Convenience script that converts all bounding polygons to minimum rectangular bounding boxes

VIA Image format: We convert
{"name":"polygon","all_points_x":[22,13,256,289,271,22],"all_points_y":[881,921,1045,1025,983,881]}
to
{"name":"rect","x":710,"y":405,"width":78,"height":137}

@author: Daniel Wu
"""

import sys
import pandas as pd
import json
import os

#Treats first command line option as filepath, unless unspecified
try:
    filepath = str(sys.argv[1])
except:
    filepath = r'C:\Users\dwubu\Desktop\TESE thaw 3-5-2019_Annotated_ImageData\via_region_data.csv'
    
labels = pd.read_csv(filepath)
new_labels = labels.copy(deep=True)
#Go through row by row
for i in range(labels.shape[0]):
    
    box = json.loads(labels.iloc[i].region_shape_attributes)
    if box["name"] == "polygon":
        max_x = max(box['all_points_x'])
        min_x = min(box['all_points_x'])
        max_y = max(box['all_points_y'])
        min_y = min(box['all_points_y'])
        
        new_box = {"name":"rect","x":min_x,"y":min_y,"width":max_x - min_x,"height":max_y - min_y}
        
        new_labels.at[i, "region_shape_attributes"] = json.dumps(new_box)
    
#Save and export
outpath = os.path.dirname(filepath) + "/via_regions_filtered.csv"
new_labels.to_csv(outpath, index=False)
