import cv2
import numpy as np

# 입력 및 출력 비디오 파일 경로
input_video_path = 'C:/Users/gram15/Desktop/Working_dir/d/video3_.mp4'
output_video_path = 'C:/Users/gram15/Desktop/Working_dir/d/video3_.mp4'

# Scharr filter 사용한 edge detection
def scharr_edge_detection(frame: np.ndarray) -> np.ndarray: 
    scharr_x = cv2.Scharr(frame, cv2.CV_64F, 1, 0)  # X 방향 엣지
    scharr_y = cv2.Scharr(frame, cv2.CV_64F, 0, 1)  # Y 방향 엣지
    edges = np.sqrt(scharr_x**2 + scharr_y**2)  # 기울기 계산
    edges = np.uint8(np.clip(edges, 0, 255))  # 0-255 범위로 클리핑
    
    return edges

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "Error reading video file"

# 비디오의 프레임 너비, 높이, FPS 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 비디오 라이터 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

while cap.isOpened():
    success, color_img = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)  # 커널 크기 (5, 5) 사용

    #_, binary_img = cv2.threshold(gray_img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    edges = scharr_edge_detection(blurred_img)

    # 결과 프레임을 비디오에 저장
    video_writer.write(edges)

# 자원 해제
cap.release()
video_writer.release()