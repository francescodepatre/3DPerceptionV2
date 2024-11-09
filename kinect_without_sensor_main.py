import cv2
import numpy as np
import folium
from folium import IFrame
from ultralytics import YOLO
from kalman_filter import KalmanFilter
import requests
from scipy.spatial.distance import cdist
import webbrowser
import datetime

date = datetime.datetime.now()
current_timestamp = date.timestamp()

model = YOLO("./last.pt")

cap_rgb = cv2.VideoCapture('./rgb/basketball_rgb.mp4')  
cap_depth = cv2.VideoCapture('./depth/basketball_depth.mp4') 

with open('./obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

kalman_filters = {}
next_id = 0  

def get_location(file_path):
    try:
        with open(file_path, 'r') as file:
            latitude = float(file.readline().strip())
            longitude = float(file.readline().strip())
        
        return latitude, longitude
    except (IOError, ValueError) as e:
        print(f"Errore nella lettura del file o nel parsing delle coordinate: {e}")
        return None, None

def update_map(kinect_lat, kinect_lon, face_positions):
    face_positions_sorted = sorted(face_positions, key=lambda x: x[0], reverse=True)

    m = folium.Map(location=[kinect_lat, kinect_lon], zoom_start=15)

    folium.Marker([kinect_lat, kinect_lon], popup='Kinect Location', icon=folium.Icon(color='blue')).add_to(m)

    for dist_meters, obj_id in face_positions_sorted:
        popup_text = f"Person ID: {obj_id}<br>Distance: {dist_meters:.2f} m"
        popup = folium.Popup(popup_text, max_width=250)

        folium.Circle(
            location=[kinect_lat, kinect_lon],
            radius=dist_meters * 10,  
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.9,
            popup=popup
        ).add_to(m)

    
    m.save('./kinect_map.html')

latitude, longitude = get_location('GPS_location_data.txt')

frame_width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap_rgb.get(cv2.CAP_PROP_FPS)

output_video = cv2.VideoWriter(f"./output_video_{current_timestamp}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
while True:
    ret_rgb, rgb_image = cap_rgb.read()
    ret_depth, depth_map = cap_depth.read()

    if not ret_rgb or not ret_depth:
        break

    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

    MAX_DEPTH = 5.5
    depth_map_meters = (depth_map/255.0)* MAX_DEPTH

    results = model(rgb_image)  
    face_positions = []
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            if conf > 0.5 and cls == 0: 
                person_center_x = x1 + (x2 - x1) // 2
                person_center_y = y1 + (y2 - y1) // 2
                detections.append((person_center_x, person_center_y, x1, y1, x2, y2))

    kalman_predictions = []
    print(f"[INFO] Numero di filtri di Kalman attivi: {len(kalman_filters)}")
    for obj_id, kalman in kalman_filters.items():
        u = np.zeros((kalman.B.shape[1], 1))  
        kalman.predict(u)  
        predicted_x, predicted_y = kalman.x[2, 0], kalman.x[3, 0]
        kalman_predictions.append((predicted_x, predicted_y, obj_id))

    if kalman_predictions and detections:
        pred_coords = np.array([(x, y) for x, y, _ in kalman_predictions])
        det_coords = np.array([(x, y) for x, y, _, _, _, _ in detections])

        distances = cdist(pred_coords, det_coords, 'euclidean')

        while distances.size > 0:
            min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)
            pred_idx, det_idx = min_dist_idx

            if distances[pred_idx, det_idx] < 100:  
                predicted_obj_id = kalman_predictions[pred_idx][2]
                det_center_x, det_center_y, x1, y1, x2, y2 = detections[det_idx]

                if latitude is not None and longitude is not None:
                    if 0 <= det_center_x < depth_map_meters.shape[1] and 0 <= det_center_y < depth_map_meters.shape[0]:
                        distance_meters = depth_map_meters[det_center_y, det_center_x]

                        z = np.array([[latitude], [longitude], [distance_meters]])
                        kalman_filters[predicted_obj_id].update(z)

                        distance_kalman = kalman_filters[predicted_obj_id].x[4, 0]
                        face_positions.append((distance_kalman, predicted_obj_id))

                        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(rgb_image, f"Human Face",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(rgb_image, f"Distance: {distance_kalman:.2f} m",
                                    (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                distances = np.delete(distances, pred_idx, axis=0)
                distances = np.delete(distances, det_idx, axis=1)
                kalman_predictions.pop(pred_idx)
                detections.pop(det_idx)
            else:
                break

    for (x, y, x1, y1, x2, y2) in detections:
        if latitude is not None and longitude is not None:
            if 0 <= y < depth_map_meters.shape[0] and 0 <= x < depth_map_meters.shape[1]:
                distance_meters = depth_map_meters[y, x]
                x0 = np.array([[latitude], [longitude], [x], [y], [distance_meters]])
                F = np.eye(5)
                H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]])
                Q = np.eye(5)
                R = np.eye(3) * 5
                P0 = np.eye(5)

                kalman_filters[next_id] = KalmanFilter(F, np.zeros((5, 1)), H, Q, R, x0, P0)

                face_positions.append((distance_meters, next_id))

                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(rgb_image, f"Human Face", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                next_id += 1

    if latitude is not None and longitude is not None:
        update_map(latitude, longitude, face_positions)

    output_video.write(rgb_image)
    cv2.imshow('Output', rgb_image)

    # Premere il tasto q per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_rgb.release()
cap_depth.release()
cv2.destroyAllWindows()

webbrowser.open(f"file:///home/francesco-de-patre/Scrivania/3DPerceptionV2/kinect_map.html")
