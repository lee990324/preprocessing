import cv2
import numpy as np

# Sobel filter 사용한 edge detection
def sobel_edge_detection(image: np.ndarray) -> np.ndarray: 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 그레이스케일 변환
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5) # Sobel 필터 적용 (X 방향)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5) # Sobel 필터 적용 (Y 방향)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2) # 기울기 강도 계산
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255)) # 기울기 강도를 0-255 범위로 변환
    
    return sobel_magnitude

# Scharr filter 사용한 edge detection
def scharr_edge_detection(image: np.ndarray) -> np.ndarray: 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 그레이 스케일링
    scharr_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)  # X 방향 엣지
    scharr_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)  # Y 방향 엣지
    edges = np.sqrt(scharr_x**2 + scharr_y**2)  # 기울기 계산
    edges = np.uint8(np.clip(edges, 0, 255))  # 0-255 범위로 클리핑
    
    return edges

# Hough Transform 사용한 edge detection, 저장 이슈 있음
def hough_transform(image: np.ndarray):
    frame = cv2.resize(image, (frame_width, frame_height)) # 해상도 맞추기
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 그레이 스케일링
    edges = cv2.Canny(gray_image, 21, 23) # Canny 엣지 감지
    # Hough Transform을 사용하여 직선 감지
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    # 감지된 직선을 원본 이미지에 그리기
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 입력 및 출력 비디오 파일 경로
input_video_path = 'C:/Users/gram15/Desktop/Working_dir/d/video3_.mp4'  # 입력 비디오 파일 경로
output_video_path = 'C:/Users/gram15/Desktop/Working_dir/d/video3_.mp4'  # 출력 비디오 파일 경로

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(input_video_path)

# 비디오의 프레임 너비, 높이, FPS 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 비디오 라이터 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break  # 비디오의 끝에 도달하면 종료

    # # Sobel filter
    edges = sobel_edge_detection(frame)

    # Scharr filter
    # edges = scharr_edge_detection(frame)
    
    # Hough Transform
    # edges = hough_transform(frame)

    # 결과 프레임을 비디오에 저장
    out.write(edges)

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()