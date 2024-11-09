import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def apply_depth_colormap(frame, colormap=cv2.COLORMAP_BONE):
    normalized_depth = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    inverted_depth = cv2.bitwise_not(np.uint8(normalized_depth))
    colored_frame = cv2.applyColorMap(inverted_depth, colormap)
    return colored_frame

parser = argparse.ArgumentParser(description="Process a depth video with colormap.")
parser.add_argument("input_path", type=str, help="Path to the input depth video file")
parser.add_argument("output_path", type=str, help="Path to save the processed video file")

args = parser.parse_args()

cap = cv2.VideoCapture(args.input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    colored_frame = apply_depth_colormap(frame)

    out.write(colored_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video di profondità modificato!")
