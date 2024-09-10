import cv2
import numpy as np

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

# 강화된 5x5 y 방향 Prewitt 커널 정의
kernely_strong = np.array([[ 3,  3,  3,  3,  3],
                           [ 2,  2,  2,  2,  2],
                           [ 0,  0,  0,  0,  0],
                           [-2, -2, -2, -2, -2],
                           [-3, -3, -3, -3, -3]])

# x 방향 Prewitt 기본 커널 (5x5) 정의
kernelx = np.array([[1, 0, -1, 0, 1], 
                    [1, 0, -1, 0, 1],
                    [1, 0, -1, 0, 1],
                    [1, 0, -1, 0, 1],
                    [1, 0, -1, 0, 1]])

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break  # 비디오의 끝에 도달하면 종료

    # 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur 적용
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Prewitt 엣지 디텍션 수행 (강화된 y 방향 커널 사용)
    prewitt_x = cv2.filter2D(gray_frame, cv2.CV_64F, kernelx)
    prewitt_y = cv2.filter2D(gray_frame, cv2.CV_64F, kernely_strong)

    # 두 방향의 엣지 크기 합산
    prewitt_combined = cv2.magnitude(prewitt_x, prewitt_y)

    # 결과를 0-255로 정규화
    prewitt_combined = cv2.normalize(prewitt_combined, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # 결과 프레임을 비디오에 저장
    out.write(prewitt_combined)

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
