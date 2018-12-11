#-*- coding: UTF-8 -*-
# @author Ke Ma for final project
# @contributor: qiqi jiryi

import os
from PIL import Image
import matplotlib.pyplot as plt

file_dir = os.getcwd()
for root, dirs, files in os.walk(file_dir):
    file_list = files

for photo in files:
    x = photo.split('-')
    if len(x) > 1:
        im = Image.open(photo)
        # im.show(100)

        im_rotate = im.rotate(15)
        # im_rotate.show(100)
        im_rotate.save(photo, quality=100, optimize=True)
