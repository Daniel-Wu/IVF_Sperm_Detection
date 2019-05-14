# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:42:40 2019

Converts CSV annotations from via and associated images to TFRecord files
--directory should contain all images and the csv file

Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
  
  
@author: Daniel Wu

"""

import os
import io
import pandas as pd
import tensorflow as tf
import json
from collections import defaultdict

from PIL import Image
import dataset_util

flags = tf.app.flags
flags.DEFINE_string('csv_input', r'C:\Users\dwubu\Desktop\CS231n Data\via_regions_test.csv', 'Path to the CSV input')
flags.DEFINE_string('output_path', r'C:\Users\dwubu\Desktop\test_data.record', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', r'C:\Users\dwubu\Desktop\CS231n Data', 'Path to the image directory')
FLAGS = flags.FLAGS


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


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  
  image_labels = parse_via_labels(FLAGS.csv_input)

  for image_name in image_labels.keys():
    tf_example = create_tf_example(image_name, FLAGS.image_dir, image_labels[image_name])
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
  