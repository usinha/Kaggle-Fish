#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 01:27:24 2017

@author: alexander
"""

#import IPython.display as dp
from PIL import Image
import os
import json


species = ['ALB','BET','DOL','LAG','OTHER','SHARK','YFT']
work_dir = '/home/icarus/kaggle/Kaggle-Fish/data/NCFM/'
json_dir = '/home/icarus/kaggle/Kaggle-Fish/data/NCFM/boxes/'
output_dir = work_dir + 'train_bb_images/'
os.chdir(work_dir)

for spec in species:
  print spec  
  input_dir =  work_dir + 'train/' + spec + '/'
  #output_image_dir = work_dir + 'train_bb_images/'
  json_file =  spec +  '_labels.json'

  #Read json
  with open(json_dir + json_file) as json_data:
      d = json.load(json_data)

  for i in range(len(d)):
    # Crop image
    fn =  str(d[i]['filename'])[-13:]    
    im = Image.open(os.path.join(input_dir,fn))
    base_fn, ext = os.path.splitext(fn)
    im_width, im_height = im.size
    for j in range(len(d[i]['annotations'])):
      y_low =   max(d[i]['annotations'][j]['y'], 0.0)
      y_high = min(y_low +  d[i]['annotations'][j]['height'], im_height)
      x_low = max(d[i]['annotations'][j]['x'], 0.0)
      x_high = min(x_low + d[i]['annotations'][j]['width'], im_width)
      im = im.crop((x_low,y_low,x_high,y_high))
      im_name = output_dir + spec + '_' + base_fn + '_' + str(j) + '.jpg'
      folder = os.path.dirname(im_name)
      if not os.path.exists(folder):
        os.makedirs(folder)
      im.save(im_name)


    
