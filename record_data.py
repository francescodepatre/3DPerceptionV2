import freenect
import cv2
import numpy as np
import os

def get_video():
    frame, _ = freenect.sync_get_video()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
    return rgb_frame

def get_depth():
    depth_frame, _ = freenect.sync_get_depth()
    depth_frame = depth_frame.astype(np.uint8)  
    return depth_frame

def save_yolo_annotation(filename, bbox, class_id=0):
    with open(filename, 'w') as f:
        x_center, y_center, width, height = bbox
        f.write(f"{class_id}, {x_center}, {y_center}, {width}, {height}\n")

rgb_save_folder = './rgb/'
depth_save_folder = './depth/'

frame_count = 0

while True:
    rgb_data = get_video()
    depth_data = get_depth()

    frame_count += 1

    rgb_filename = os.path.join(rgb_save_folder, f"rgb_frame_{frame_count}.png")
    depth_filename = os.path.join(depth_save_folder, f"depth_frame_{frame_count}.png")

    cv2.imwrite(rgb_filename, rgb_data)
    cv2.imwrite(depth_filename, depth_data)
    print(f"Frame {frame_count} salvato.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

freenect.sync_stop()
cv2.destroyAllWindows()