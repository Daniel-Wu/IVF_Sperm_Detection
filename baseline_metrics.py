# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:21:39 2019

@author: dwubu
"""

import numpy as np
import pandas as pd
import json
import os
import copy
from iou import iou
from collections import defaultdict
import tensorflow as tf
from PIL import Image
import dataset_util
import io

def parse_via_labels(filepath):
    labels = pd.read_csv(filepath)
    parsed_labels = {}
    new_image = defaultdict(list)
    for i in range(labels.shape[0]):
        
        row = labels.iloc[i]

        #If the image has no bounding boxes, skip it
        if row.region_count == 0:
            continue
        
                
        #Unpack bounding box dimensions
        box = json.loads(row.region_shape_attributes)
        min_x = box["x"]
        min_y = box["y"]
        max_x = box["width"] + min_x
        max_y = box["height"] + min_y
                
        #Store this information in the new_image object
        new_image["xmins"].append(min_x)
        new_image["ymins"].append(min_y)
        new_image["xmaxs"].append(max_x)
        new_image["ymaxs"].append(max_y)
        
        #If this is the last region for this image
        if row.region_id == row.region_count - 1:
            #Store the regions and initialize a new image
            parsed_labels[row[r"#filename"]] = new_image
            new_image = defaultdict(list)
        
    return parsed_labels

def create_tf_example(filename, dirpath, boxes):
    
    #Read in the image
    with tf.gfile.GFile(os.path.join(dirpath, filename), 'rb') as fid:
        encoded_image_data = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_image_data)
    image = Image.open(encoded_jpg_io)
    
    width, height = image.size
    _, ext = os.path.splitext(filename)
    image_format = ext[1:].encode('UTF-8') # b'jpeg' or b'png'

    xmins = [x/width for x in boxes['xmins']] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [x/width for x in boxes['xmaxs']] # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [y/height for y in boxes['ymins']] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [y/height for y in boxes['ymaxs']] # List of normalized bottom y coordinates in bounding box (1 per box)
  
    #DEFAULT LABELS AND CLASSES FOR IVF
    classes_text = [b'sperm']*len(xmins) # List of string class name of bounding box (1 per box)
    classes = [1]*len(xmins) # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
                              'image/height': dataset_util.int64_feature(height),
                              'image/width': dataset_util.int64_feature(width),
                              'image/filename': dataset_util.bytes_feature(filename.encode('UTF-8')),
                              'image/source_id': dataset_util.bytes_feature(filename.encode('UTF-8')),
                              'image/encoded': dataset_util.bytes_feature(encoded_image_data),
                              'image/format': dataset_util.bytes_feature(image_format),
                              'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                              'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                              'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                              'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                              'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                              'image/object/class/label': dataset_util.int64_list_feature(classes),
                              }))
    return tf_example

def find_consensus(box_labels):
    '''
    Function which finds the consensus across a set of n image bounding coordinates
    Takes in a list containing n dictionaries with the keys xmaxs, xmins, ymaxs, ymins
    each key mapping to a list of coordinates
    
    returns a dictionary of the consensus coordinates in the same format
    and a count of the number of errors
    '''
    consensus = defaultdict(list)
    error_count = 0

    #Find the number of total boxes in this image
    num_boxes = []
    for box_label in box_labels:
        num_boxes.append(len(box_label["xmaxs"]))
    #mode
    num_boxes = max(num_boxes, key = num_boxes.count)
    
    #Find a label set with the correct number of boxes
    for box_label in box_labels:
        if len(box_label["xmaxs"]) == num_boxes:
            truth = copy.deepcopy(box_label)
            break
        
    #Find and remove extra erronous boxes, by removing the box with worst max iou
    for box_label in box_labels:
        
        if len(box_label["xmaxs"]) < num_boxes:
            
            error_count += num_boxes - len(box_label["xmaxs"]) 
        
        if len(box_label["xmaxs"]) > num_boxes:
            
            error_count += len(box_label["xmaxs"]) - num_boxes
            
            best_iou = []
            for i in range(len(box_label["xmaxs"])):
                curr_best_iou = -1
                for j in range(len(truth["xmaxs"])):
                    temp_iou = iou(box_label["xmins"][i], box_label["ymins"][i],
                                   box_label["xmaxs"][i], box_label["ymaxs"][i],
                                   truth["xmins"][j], truth["ymins"][j],
                                   truth["xmaxs"][j], truth["ymaxs"][j])
                    
                    
                    if temp_iou > curr_best_iou:
                        curr_best_iou = temp_iou
                
                best_iou.append(curr_best_iou)
            
            #Find and remove the worst index
            wrong_box_idx = np.argmin(best_iou)
            del box_label["xmaxs"][wrong_box_idx]
            del box_label["ymaxs"][wrong_box_idx]
            del box_label["xmins"][wrong_box_idx]
            del box_label["ymins"][wrong_box_idx]
            
    #Calculate average boxes
    for i in range(num_boxes):
        true_xmax = truth["xmaxs"][i]
        true_ymax = truth["ymaxs"][i]
        true_xmin = truth["xmins"][i]
        true_ymin = truth["ymins"][i]
        
        sample_xmaxs = []
        sample_ymaxs = []
        sample_xmins = []
        sample_ymins = []
        
        for box_label in box_labels:
            best_iou = -1
            best_iou_idx = -1
            for j in range(len(box_label["xmaxs"])):
                temp_iou = iou(box_label["xmins"][j], box_label["ymins"][j],
                               box_label["xmaxs"][j], box_label["ymaxs"][j],
                               true_xmin, true_ymin,
                               true_xmax, true_ymax)
                    
                    
                if temp_iou > best_iou:
                    best_iou = temp_iou
                    best_iou_idx = j        
            
            #If empty list, skip
            if best_iou_idx == -1:
                continue
            sample_xmaxs.append(box_label["xmaxs"].pop(best_iou_idx))
            sample_ymaxs.append(box_label["ymaxs"].pop(best_iou_idx))
            sample_xmins.append(box_label["xmins"].pop(best_iou_idx))
            sample_ymins.append(box_label["ymins"].pop(best_iou_idx))
            
        #Populate our consensus with the average boxes
        consensus["xmaxs"].append(np.mean(sample_xmaxs))
        consensus["ymaxs"].append(np.mean(sample_ymaxs))
        consensus["xmins"].append(np.mean(sample_xmins))
        consensus["ymins"].append(np.mean(sample_ymins))
        
    return consensus, error_count

def calc_IOUs(box_labels, gt):
    '''
    Calculate the best IOU of each proposed box compared to gt boxes
    '''
    num_boxes = len(gt['xmaxs'])
    print(num_boxes)
    ious = np.zeros((len(box_labels), num_boxes))
    #Calculate average boxes
    for i in range(num_boxes):
                
        true_xmax = gt["xmaxs"][i]
        true_ymax = gt["ymaxs"][i]
        true_xmin = gt["xmins"][i]
        true_ymin = gt["ymins"][i]
        
        for usr_num in range(len(box_labels)):
            
            box_label = box_labels[usr_num]
            best_iou = 0
            for j in range(len(box_label["xmaxs"])):
                temp_iou = iou(box_label["xmins"][j], box_label["ymins"][j],
                               box_label["xmaxs"][j], box_label["ymaxs"][j],
                               true_xmin, true_ymin,
                               true_xmax, true_ymax)
                    
                    
                if temp_iou > best_iou:
                    best_iou = temp_iou
                    
            ious[usr_num, i] = best_iou
            
    #Return the list of ious
    return ious

def calc_AR(ious):
    '''
    Calculates COCO AR given a list of IOU values
    '''
    iou_threshes = np.linspace(0.50, 0.95, 10)
    results = np.zeros_like(iou_threshes)
    for i in range(len(iou_threshes)):
        thresh = iou_threshes[i]
        temp = ious > thresh
        results[i] = sum(temp)/len(temp)
        
    return results
        

if __name__ == "__main__":
    
    write = False
    
    label_path = r"C:\Users\dwubu\Desktop\Embryologist_baseline_data_set\Baseline"
    label_filenames = ["OB_labels.csv", "Moon_labels.csv", "VR_labels.csv"]
        
    writer = tf.python_io.TFRecordWriter(os.path.join(label_path, "baseline.record"))

    total_errors = 0
    total_sperm = 0
    all_best_ious = np.array([])
    
    #Load in and parse all the labels
    parsed_labels = []
    for filename in label_filenames:
        parsed_labels.append(parse_via_labels(os.path.join(label_path, filename)))
        
    #Go through each image
    for img in parsed_labels[0]:
        #Get the labels for this image
        img_labels = []
        for parsed_label in parsed_labels:
            img_labels.append(parsed_label.get(img, defaultdict(list)))
            
        #Get the consensus ground truth
        usr_labels = copy.deepcopy(img_labels)
        consensus_labels, error_count = find_consensus(img_labels)
        ious = calc_IOUs(usr_labels, consensus_labels)
        all_best_ious = np.append(all_best_ious, ious)
        
        if(write):
            tf_example = create_tf_example(img, label_path, consensus_labels)
            writer.write(tf_example.SerializeToString())

        total_errors += error_count
        total_sperm += len(consensus_labels["xmaxs"])
        print("ERRORS: ", total_errors, total_sperm)
        print(img)
        print(consensus_labels)
    
    print("Total error rate: {} errors out of {} labels, {}%".format(total_errors, total_sperm*len(label_filenames), 100*total_errors/(total_sperm*len(label_filenames))))
        
    print("Average recall is {}".format(np.average(calc_AR(all_best_ious))))
  

    writer.close()