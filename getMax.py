import cv2

video_path = 'C:/temp/KU Leuven/Master/Master Thesis/Database/ovad/SA1/left/1_00_0002/ueye_stereo_vid.mp4'
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"视频分辨率：{width} x {height}")
# 1936 x 1216
cap.release()

