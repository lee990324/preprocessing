import cv2

input_video_path = '/home/bcml1/WBC_project/data/video/video_20240829_204726_crop_sod.avi'  # 입력 비디오 파일 경로
output_video_path = '/home/bcml1/WBC_project/data/video/video_20240829_204726_crop_sod_canny.avi'  # 출력 비디오 파일 경로

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

    # 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Canny 엣지 디텍션 수행
    edges = cv2.Canny(gray_frame, 21, 23)

    # 결과 프레임을 비디오에 저장
    out.write(edges)
# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
