import cv2
import numpy as np

# 입력 및 출력 비디오 파일 경로
input_video_path = 'C:/Users/gram15/Desktop/Working_dir/d/video3_.mp4'  # 입력 비디오 파일 경로
output_video_path = 'C:/Users/gram15/Desktop/Working_dir/d/video3_hough.mp4'  # 출력 비디오 파일 경로

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(input_video_path)

# 비디오의 프레임 너비, 높이, FPS 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 비디오 라이터 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break  # 비디오의 끝에 도달하면 종료

    # 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Canny 엣지 디텍션 수행
    edges = cv2.Canny(gray_frame, 50, 150)

    # Hough Line 변환 적용
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    # 직선 그리기
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 결과 프레임을 비디오에 저장
    out.write(frame)

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
