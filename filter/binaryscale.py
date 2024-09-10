import cv2

#영상 경로
cap = cv2.VideoCapture("C:/Users/gram15/Desktop/Working_dir/d/video3__prewitt.mp4")

assert cap.isOpened(), "Error reading video file"
#영상 fps및 frame 크기 가져오기
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 영상 저장
video_writer = cv2.VideoWriter("C:/Users/gram15/Desktop/Working_dir/d/video3__prewitt_binary50.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)

while cap.isOpened():
    success, color_img = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
    video_writer.write(binary_img)

cap.release()
video_writer.release()