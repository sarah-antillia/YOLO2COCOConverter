# This is based on the following code by Tae Young Kim

# https://github.com/Taeyoung96/Yolo-to-COCO-format-converter
## License
"""
Copyright (c) 2021 Tae Young Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# YOLOtoCOCOConverter.py
# 2022/04/05  Copyright (C) 2022 Toshiyuki Arai Antillia.com


import sys
import os
import glob
import cv2
import argparse
import json
import numpy as np
import pprint
import traceback

# classes_file is a text file which contains all classes(labels)
# Example classes.txt
"""
Bicycles_Only
Bumpy_Road
Buses_Priority
Centre_Line
Closed_To_Pedestrians
Crossroads
Dangerous_Wind _Gusts
Directions_Indicator
Falling_Rocks
Keep_Left
...
"""

class YOLO2COCOConverter:

  def __init__(self, classes_file):
    self.classes = []
    with open(classes_file, "r") as f:
      all_class_names = f.readlines()
      for class_name in all_class_names:
        class_name = class_name.strip()
        if class_name.startswith("#") ==False:
          self.classes.append(class_name)
    print("==== classes {}".format(self.classes))


  def run(self, dataset_dir, coco_json):
    print("=== YOLO2COCOConverter.run() start")
    NL = "\n"
    coco_annotation = {}
    info = {
      "description": "COCO 2022 Dataset",
      "url": "http://cocodataset.org",
      "version": 1.0,
      "year": 2022,
      "contributor": "antillia.com",
      "date_created": "2022/04/05"
    }
    license = [
        {
          "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
          "id":  1,
          "name": "Attribution-NonCommercial-ShareAlike License"
        },
    ]
    coco_annotation["info"]        = info
    coco_annotation["license"]     = license
    
    images, annotations = self.create_images_and_annotations_sections(dataset_dir)
 
    coco_annotation['images']      = images
    coco_annotation['annotations'] = annotations
    coco_annotation['categories']  = []
    
    for index, label in enumerate(self.classes):
      category = {
        "supercategory": "None",
        "id": index + 1, 
        "name": label
      }
      coco_annotation['categories'].append(category)

    with open(coco_json, 'w') as outfile:
      json.dump(coco_annotation, outfile, indent=4)

    print("=== YOLO2COCOConverter.run() end")


  def create_images_and_annotations_sections(self, dataset_dir):
    # The dataset_dir(train, valid, or test) contains both something_image.jpg and something_image.txt , and so on.
   
    # We assume the following folder structure:
    # dataset_dir/train/
    #  +- image1.jpg
    #  +- image1.txt  (YOLO format annotation file to image1.jpg)
    #  ...
    #  ...
    #  +- imageN.jpg
    #  +- imageN.txt  (YOLO format annotation file to imageN.jpg)
    #       
    annotations = []
    images      = []
    pattern     = dataset_dir + "/*.jpg"
    all_jpg_files   = glob.glob(pattern)

    image_id      = 0
    annotation_id = 1   # In COCO dataset format, you must start annotation id with '1'

    for jpg_file in all_jpg_files:
        # Check how many items have progressed
        print("==== YOLOtoCOCOConverter.run() processing  annotation_id {} filename {}".format(annotation_id, jpg_file))
        
        image = cv2.imread(jpg_file)

        annotation_file = jpg_file.replace(".jpg", ".txt")
        if os.path.exists(annotation_file) == False:
          raise Exception("Not found annotation file {}".format(annotation_file))

        annotation_f = open(annotation_file, "r")
        all_annotations  = annotation_f.readlines()
        annotation_f.close()

        image_height, image_width, _ = image.shape

        jpg_base_filename = os.path.basename(jpg_file)
  
        image = self.create_image_section(jpg_base_filename, image_width, image_height, image_id)
        images.append(image)

        image_height = float(image_height)
        image_width  = float(image_width)

        # YOLO format - (class_id, x_center, y_center, width, height)
        # where class_id starts from 0, and x_center, y_center, width, height are float, and in range [0.0, 1.0]

        # COCO format - (annotation_id, x_upper_left, y_upper_left, width, height)
        # where annotation_id starts from 1, and x_upper_left, y_upper_left, width, height are integer, and real coordinates of the bbox rectangle. 

        for annotation in all_annotations:
            
            class_id, x_center, y_center, width, height = annotation.split(" ")

            category_id   = int(class_id) + 1    
            x_center      = float(x_center) 
            y_center      = float(y_center) 
            width         = float(width)
            height        = float(height)
            
            real_x_center = int(x_center * image_width)
            real_y_center = int(y_center * image_height)
            real_width    = int(width    * image_width)
            real_height   = int(height   * image_height)

            real_x        = real_x_center - real_width/2
            real_y        = real_y_center - real_height/2

            annotation    = self.create_annotation_section(real_x, real_y, real_width, real_height, image_id, category_id, annotation_id)
            annotations.append(annotation)
            annotation_id += 1

        image_id += 1  
 
    return images, annotations

  def create_image_section(self, file_name, width, height, image_id):
    image_section = {
        'file_name': file_name,
        'height':    height,
        'width':     width,
        'id':        image_id
    }
    return image_section


  def create_annotation_section(self, x, y, width, height, image_id, category_id, annotation_id):
    bbox = (x, y, width, height)
    area = width * height

    annotation_section = {
        'id': annotation_id,
        'image_id': image_id,
        'bbox': bbox,
        'area': area,
        'iscrowd': 0,
        'category_id': category_id,
        'segmentation': []
    }

    return annotation_section


# python YOLO2COCOConverter.py ./classes.txt ./train ./train/annotation.json
#
# python YOLO2COCOConverter.py ./classes.txt ./valid ./valid/annotation.json

if __name__ == '__main__':
  classes_file     = ""
  dataset_dir      = ""
  output_coco_json = ""
  try:
    if len(sys.argv) == 4:
      classes_file     = sys.argv[1]
      dataset_dir      = sys.argv[2]
      output_coco_json = sys.argv[3]
    else:
      raise Exception("Invalid argment")

    if os.path.exists(classes_file) == False:
      raise Exception("Not found classes files:{}".format(classes_file))
    if os.path.exists(dataset_dir) == False:
      raise Exception("Not found dataset_dir  :{}".format(dataset_dir))

    converter = YOLO2COCOConverter(classes_file)
    converter.run(dataset_dir, output_coco_json)

  except:
    traceback.print_exc()
  
