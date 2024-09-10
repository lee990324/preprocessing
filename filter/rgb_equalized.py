import cv2
import numpy as np

# 비디오 파일 열기
video_capture = cv2.VideoCapture('C:/Users/gram15/Desktop/visual studio/data/video_20240829_204726.avi')

if not video_capture.isOpened():
    print("Error: Could not open video file.")
    exit()

# 비디오의 프레임 크기 가져오기
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

gamma = 0.65  # You can adjust this value depending on the desired correction
inv_gamma = 1.0 / gamma

def equalized_frame_RGB(frame):
    r, g, b = cv2.split(frame)
    re, ge, be = cv2.equalizeHist(r), cv2.equalizeHist(g), cv2.equalizeHist(b)
    equalized_frame = cv2.merge([re, ge, be])
    return equalized_frame

def equalized_frame_grayscale(frame):
    # 프레임을 grayscale로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 히스토그램 평활화 적용
    equalized_frame = cv2.equalizeHist(gray_frame)
    return equalized_frame

def gamma_correction_frame(frame):
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    gamma_correction_frame = cv2.LUT(frame, table)
    return gamma_correction_frame


# 비디오 저장을 위한 설정 (crop된 크기에 맞추기)
video = cv2.VideoWriter('C:/Users/gram15/Desktop/visual studio/data/video_20240829_204726_rgb_hist.mp4',
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        30,
                        (frame_width, frame_height),
                        isColor=True)  # grayscale 비디오를 저장하기 위해 isColor를 False로 설정

while True:
    # 비디오에서 프레임 읽기
    ret, frame = video_capture.read()

    if not ret:
        break  # 프레임을 더 이상 읽을 수 없으면 루프 종료

    # crop 영역만큼 프레임 crop
    view_hist_frame = equalized_frame_RGB(frame)
    # view_grayscale_frame = equalized_frame_grayscale(frame)
    # view_gamma_frame = gamma_correction_frame(frame)
    # view_gamma_hist_frame = equalized_frame_RGB(view_gamma_frame)
    view_hist_gamma_frame = gamma_correction_frame(view_hist_frame)

    # crop된 프레임 저장
    video.write(view_hist_gamma_frame)

# 자원 해제
video_capture.release()
video.release()
cv2.destroyAllWindows()
