import cv2
import numpy as np

# 비디오 파일 열기
video_capture = cv2.VideoCapture('C:/Users/gram15/Desktop/visual studio/data/video_20240829_204726_crop.avi')

if not video_capture.isOpened():
    print("Error: Could not open video file.")
    exit()

# 비디오의 프레임 크기 가져오기
frame_width = 1920
frame_height = 1280

# 비디오 저장을 위한 설정 (crop된 크기에 맞추기)
video = cv2.VideoWriter('C:/Users/gram15/Desktop/visual studio/data/video_20240829_204726_crop_test.avi',
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        30,
                        (frame_width, frame_height),
                        isColor=True)  # grayscale 비디오를 저장하기 위해 isColor를 False로 설정

while True:
    # 비디오에서 프레임 읽기
    ret, frame = video_capture.read()

    if not ret:
        break  # 프레임을 더 이상 읽을 수 없으면 루프 종료

    interpolation_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LANCZOS4)
    
    video.write(interpolation_frame)

# 자원 해제
video_capture.release()
video.release()
cv2.destroyAllWindows()
