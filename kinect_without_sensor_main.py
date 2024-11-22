import cv2
import numpy as np
import folium
from ultralytics import YOLO
import requests
import webbrowser
import datetime
from shapely.geometry import Point
from pyproj import Geod
import math
import MultiModalModel
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

date = datetime.datetime.now()
current_timestamp = date.timestamp()

model_yolo = YOLO("last.pt")

cap_rgb = cv2.VideoCapture('5_rgb.mp4')  
cap_depth = cv2.VideoCapture('5_depth.mp4') 

with open('obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

next_id = 0
#fov camera 60° quindi -30°,+30°
def get_location(file_path):
    try:
        with open(file_path, 'r') as file:
            latitude = float(file.readline().strip())
            longitude = float(file.readline().strip())
        
        return latitude, longitude
    except (IOError, ValueError) as e:
        print(f"Errore nella lettura del file o nel parsing delle coordinate: {e}")
        return None, None

def calcola_angolo(x):
    width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    cx=width/2
    dx=x-cx
    angle=(dx/cx)*30
    return angle

def update_map(kinect_lat, kinect_lon, face_positions, angle):
    #face_positions_sorted = sorted(face_positions, key=lambda x: x[0], reverse=True)

    m = folium.Map(location=[kinect_lat, kinect_lon], zoom_start=15)

    folium.Marker([kinect_lat, kinect_lon], popup='Kinect Location', icon=folium.Icon(color='blue')).add_to(m)
    origin = (kinect_lat, kinect_lon)
    for dist_meters, angle_rel, obj_id in face_positions:
        angle_rad = math.radians(angle+angle_rel)
        # Definisci il geod per WGS84 (il sistema di coordinate GPS più comune)
        geod = Geod(ellps="WGS84")
    
        # Calcola la nuova posizione
        lon2, lat2, _ = geod.fwd(kinect_lon, kinect_lat, angle, dist_meters*10)


        popup_text = f"Person ID: {obj_id}<br>Distance: {dist_meters:.2f} m"
        popup = folium.Popup(popup_text, max_width=250)

        folium.Marker(
            location=[lat2, lon2],
            popup=popup,
            icon=folium.Icon(color='green', icon='user', prefix='fa')
        ).add_to(m)

    
    m.save('kinect_map.html')

latitude, longitude = get_location('GPS_location_data.txt')

angle=28.12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiModalModel.MultiModalLSTMModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))

# Imposta il modello in modalità valutazione
model.eval()
frame_width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap_rgb.get(cv2.CAP_PROP_FPS)
counter = 0
sequence_length = 10
while True:
    ret_rgb, rgb_image = cap_rgb.read()
    ret_depth, depth_map = cap_depth.read()
    rgb_sequence = []
    numeric_sequence = []
    counter += 1
    if not ret_rgb or not ret_depth:
        break

    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

    MAX_DEPTH = 5.5
    depth_map_meters = (depth_map/255.0)* MAX_DEPTH
    next_id=0

    results = model_yolo(rgb_image)  
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



    for (x, y, x1, y1, x2, y2) in detections:
        if latitude is not None and longitude is not None:
            if 0 <= y < depth_map_meters.shape[0] and 0 <= x < depth_map_meters.shape[1]:
                distance_meters = depth_map_meters[y, x]
                transform = transforms.Compose([
                transforms.ToPILImage(),  # Converte l'immagine OpenCV (numpy array) in PIL
                transforms.Resize((224, 224)),  # Ridimensiona l'immagine a 224x224
                transforms.ToTensor(),  # Converte l'immagine PIL in Tensor (con dimensioni [C, H, W])
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza come richiesto da ResNet
            ])

            # Ritaglia l'immagine e verifica che il ritaglio non sia vuoto
            cropped_face = rgb_image[y1:y2, x1:x2]

            # Controlla che il ritaglio non sia vuoto
            if cropped_face.size == 0:
                print(f"Warning: bounding box ({x1}, {y1}, {x2}, {y2}) produce un ritaglio vuoto.")
                continue
            if len(cropped_face.shape) == 2 or cropped_face.shape[2] == 1:
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_GRAY2BGR)
            # Applica le trasformazioni all'immagine ritagliata
            try:
                frame_rgb_tensor = transform(cropped_face)  # Risultato: [3, 224, 224]
                frame_rgb_tensor = frame_rgb_tensor.unsqueeze(0).to(device)  # Aggiungi una dimensione per il batch: [1, 3, 224, 224]
                rgb_sequence.append(frame_rgb_tensor)
            except Exception as e:
                print(f"Errore durante la trasformazione dell'immagine: {e}")
                continue

            # Prepara i dati numerici
            numeric_data = [distance_meters, latitude, longitude, angle]
            numeric_tensor = torch.tensor(numeric_data, dtype=torch.float32).unsqueeze(0).to(device)  # Deve avere dimensioni [1, 4]
            numeric_sequence.append(numeric_tensor)
            #if (counter % sequence_length) == 0:
            rgb_sequence_tensor = torch.stack(rgb_sequence).unsqueeze(0).to(device)  # Forma: [1, seq_length, 3, 224, 224]
            numeric_sequence_tensor = torch.stack(numeric_sequence).unsqueeze(0).to(device)  # Forma: [1, seq_length, 4]
            with torch.no_grad():
                prediction = model(frame_rgb_tensor, numeric_tensor)
                predicted_distance = prediction.item()*2
                print(f"Predicted distance: {predicted_distance:.2f} m; ID: {next_id}")
                angle_rel = calcola_angolo(x)
                face_positions.append([predicted_distance, angle_rel, next_id])
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rgb_image, f"Human Face ID:{next_id}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(rgb_image, f"Distance: {predicted_distance:.2f} m",
                            (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                next_id += 1

            if (counter % sequence_length) == 0:
                rgb_sequence.clear()
                numeric_sequence.clear()
            # Passa l'immagine e i dati numerici al modello
            '''else:
                with torch.no_grad():
                    prediction = model(frame_rgb_tensor, numeric_tensor)
                    predicted_distance = prediction.item()*2.5
                    print(f"Predicted distance: {predicted_distance:.2f} m; ID: {next_id}")
                    angle_rel = calcola_angolo(x)
                    face_positions.append([predicted_distance, angle_rel, next_id])
                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(rgb_image, f"Human Face ID:{next_id}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(rgb_image, f"Distance: {predicted_distance:.2f} m",
                                (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    next_id += 1'''

    if latitude is not None and longitude is not None:
        update_map(latitude, longitude, face_positions, angle)

    cv2.imshow('Output', rgb_image)

    # Premere il tasto q per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_rgb.release()
cap_depth.release()
cv2.destroyAllWindows()

webbrowser.open(f"kinect_map.html")
