import os
import cv2
import numpy as np
from PIL import Image
from transparent_background import Remover

# Load model
# remover = Remover() # default setting
remover = Remover(mode='fast', jit=True, device='cuda:0') # custom setting
# remover = Remover(mode='base-nightly') # nightly release checkpoint

# Usage for image
# img = Image.open('image/total/input02/001.png').convert('RGB') # read image
img = Image.open('/home/bcml1/WBC_project/data/image/dvs event recording/dvs_event_210914_14/000.png').convert('RGB') # read image

# out = remover.process(img) # default setting - transparent background
# out = remover.process(img, type='rgba') # same as above
# out = remover.process(img, type='map') # object map only
# out = remover.process(img, type='green') # image matting - green screen
# out = remover.process(img, type='white') # change backround with white color
# out = remover.process(img, type=[255, 0, 0]) # change background with color code [255, 0, 0]
# out = remover.process(img, type='blur') # blur background
# out = remover.process(img, type='overlay') # overlay object map onto the image
out = remover.process(img, type='/home/bcml1/WBC_project/preprocessing/filter/Solid_black.jpg', threshold=0.05) # use another image as a background

# out = remover.process(img, threshold=0.5) # use threhold parameter for hard prediction.

# out = remover.process(img, type='white', threshold=0.5)

dirname = os.path.abspath('')
result_path = './'
if not os.path.exists(result_path):
     os.makedirs(result_path)
out.save(f'{result_path}000.png') # save result
'''
for i in range(85):
    img = Image.open(f'image/total/input05/{i:03d}.png').convert('RGB') # read image
    
    out = remover.process(img, type='solid_black.png', threshold=0.05) # use image as a background
    
    dirname = os.path.abspath('')
    result_path = dirname + '/image/sod/input05/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    out.save(f'{result_path}{i:03d}.png') # save result
'''