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

from PIL import Image
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def parse_via_labels(filepath):
    labels = pd.read_csv(filepath)
    parsed_labels = {}
    new_image = {}
    for i in range(labels.shape[0]):
        row = labels.iloc[i]
        
        #Unpack bounding box dimensions
        box = json.loads(row.region_shape_attributes)
        min_x = box["x"]
        min_y = box["y"]
        max_x = box["width"] + min_x
        max_y = box["height"] + min_y
        
        #Store this information in the new_image object
        new_image["xmins"] = new_image.get("xmins", []).append(min_x)
        new_image["ymins"] = new_image.get("ymins", []).append(min_y)
        new_image["xmaxs"] = new_image.get("xmaxs", []).append(max_x)
        new_image["ymaxs"] = new_image.get("ymaxs", []).append(max_y)
        
        #If this is the last region for this image
        if row.region_id == row.region_count - 1:
            #Store the regions and initialize a new image
            parsed_labels[row[r"#filename"]] = new_image
            new_image = {}
        
    return 




def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = None # Image height
  width = None # Image width
  filename = None # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = None # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
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

  # TODO(user): Write code to read in your dataset to examples variable

  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


#if __name__ == '__main__':
#  tf.app.run()
  
  
  
# =============================================================================
# 
# =============================================================================

flags = tf.app.flags
flags.DEFINE_string('csv_input', r'C:\Users\dwubu\Desktop\TESE thaw 3-5-2019_Annotated_ImageData\via_regions_filtered.csv', 'Path to the CSV input')
flags.DEFINE_string('output_path', r'C:\Users\dwubu\Desktop\sperm_data.record', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', r'C:\Users\dwubu\Desktop\TESE thaw 3-5-2019_Annotated_ImageData', 'Path to the image directory')
FLAGS = flags.FLAGS


#Only single image labels - return 1. Implement your own label processing!
def class_text_to_int(row_label):
    return 1



def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
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
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


#if __name__ == '__main__':
#    tf.app.run()