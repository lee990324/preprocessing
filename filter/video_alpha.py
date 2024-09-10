import cv2
import numpy as np

#영상 경로
cap = cv2.VideoCapture("/home/bcml1/WBC_project/data/video/video_20240829_204726_crop.avi")
assert cap.isOpened(), "Error reading video file"
#영상 fps및 frame 크기 가져오기
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
alpha = 1.0

# 영상 저장
video_writer = cv2.VideoWriter("/home/bcml1/WBC_project/data/video/video_20240829_204726_crop_alpha.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)

while cap.isOpened():
    success, color_img = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    dst = np.clip((1+alpha) * color_img - 128 * alpha, 0, 255).astype(np.uint8)
    video_writer.write(dst)

cap.release()
video_writer.release()