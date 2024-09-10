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
#영상 경로
cap = cv2.VideoCapture("/home/bcml1/WBC_project/data/video/video_20240829_204726_crop.avi")
assert cap.isOpened(), "Error reading video file"
#영상 fps및 frame 크기 가져오기
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


# 영상 저장
video_writer = cv2.VideoWriter("/home/bcml1/WBC_project/data/video/video_20240829_204726_crop_sod.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=True)

# out = remover.process(img) # default setting - transparent background
# out = remover.process(img, type='rgba') # same as above
# out = remover.process(img, type='map') # object map only
# out = remover.process(img, type='green') # image matting - green screen
# out = remover.process(img, type='white') # change backround with white color
# out = remover.process(img, type=[255, 0, 0]) # change background with color code [255, 0, 0]
# out = remover.process(img, type='blur') # blur background
# out = remover.process(img, type='overlay') # overlay object map onto the image

# out = remover.process(img, threshold=0.5) # use threhold parameter for hard prediction.

# out = remover.process(img, type='white', threshold=0.5)

while cap.isOpened():
    success, color_img = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    out = remover.process(color_img, type='/home/bcml1/WBC_project/preprocessing/sod/Solid_black.jpg', threshold=0.05) # use another image as a background
    #gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    #_, binary_img = cv2.threshold(gray_img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    video_writer.write(out)

cap.release()
video_writer.release()