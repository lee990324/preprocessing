import cv2

# 입력 및 출력 비디오 파일 경로
input_video_path = 'C:/Users/gram15/Desktop/Working_dir/d/video3_.mp4'  # 입력 비디오 파일 경로
output_video_path = 'C:/Users/gram15/Desktop/Working_dir/d/video3_FAST.mp4'  # 출력 비디오 파일 경로

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(input_video_path)

# 비디오의 프레임 너비, 높이, FPS 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 비디오 라이터 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)

# FAST 검출기 생성
fast = cv2.FastFeatureDetector_create()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break  # 비디오의 끝에 도달하면 종료

    # 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FAST를 사용한 특징점 검출
    keypoints = fast.detect(gray_frame, None)

    # 특징점 그리기
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(255, 0, 0))

    # 결과 프레임을 비디오에 저장
    out.write(frame_with_keypoints)

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
