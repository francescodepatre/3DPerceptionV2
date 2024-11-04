import cv2
import numpy as np
from ultralytics import YOLO
from kinect_sensor import Kinect
from kalman_filter import KalmanFilter
from errorMessages import ErrorManager
import requests

def get_location():
    try:
        
        response = requests.get('https://ipinfo.io/json')
        data = response.json()

        # Estrai la latitudine e la longitudine
        location = data['loc'].split(',')
        latitude = location[0]
        longitude = location[1]

        return latitude, longitude


    except requests.exceptions.RequestException as e:
        ErrorManager = ErrorManager()
        ErrorManager.messageError("Dati GPS non disponibili")

# Carica il modello YOLO addestrato
model = YOLO("/home/francesco-de-patre/Scrivania/3DPerceptionV2/last.pt")

Kinect = Kinect()

# Carica le classi dal file obj.names
with open('/home/francesco-de-patre/Scrivania/3DPerceptionV2/obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Stato iniziale per il filtro di Kalman: [latitudine, longitudine, x_pers, y_pers, dist_pers] //fra
x0 = np.array([[0], [0], [0], [0], [0]])
dt = 1 #tempo tra due misurazioni consecutive ad esempio il tempo trascorso tra i fotogrammi(presumo da modificare)
#Matrici del modello di Kalman
F = np.array([[ 1, 0, 0, 0, 0  ],
              [ 0, 1, 0, 0, 0  ],
              [ 0, 0, 1, 0, dt ],
              [ 0, 0, 0, 1, dt ],
              [ 0, 0, 0, 0, 1  ]])

H = np.array([[ 1, 0, 0, 0, 0 ],
              [ 0, 1, 0, 0, 0 ],
              [ 0, 0, 0, 0, 1 ],])

Q = np.eye(6) * 0.01    #Rumore di processo     (Presumo da Cambiare)
R = np.eye(3) * 0.1     #Rumore di misurazione  (Presumo da Cambiare)

P0 = np.eye(6)          #Covarianza Iniziale    (Presumo da Cambiare)

Kalman = KalmanFilter(F, np.zeros((5, 1)), H, Q, R, x0, P0)

#estrai i dati gps

while True:
    # Cattura i frame dal Kinect
    rgb_image = Kinect.get_realtime_video()
    depth_map = Kinect.get_realtime_depth()

    results = model(rgb_image)  # Passa il frame direttamente al modello
    latitude, longitude = get_location()
    # Parsing dei risultati
    for r in results:
        for box in r.boxes:  # Rilevazioni trovate
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinate del bounding box
            conf = box.conf[0]  # Confidenza
            cls = int(box.cls[0])  # Classe predetta (ID)

            if conf > 0.5 and cls == 0:  # Filtra per confidenza e classe
                label = f"{classes[cls]} {conf:.2f}"

                # Calcola il centro della persona e la distanza
                person_center_x = x1 + (x2 - x1) // 2
                person_center_y = y1 + (y2 - y1) // 2

                if 0 <= person_center_x < rgb_image.shape[1] and 0 <= person_center_y < rgb_image.shape[0]:
                    distance  = depth_map[person_center_y, person_center_x]
                    distance_meters = distance / 1000.0  # Converte in metri
                else:
                    distance_meters = None
                
                if distance_meters is not None:
                    z = np.array([[latitude], [longitude], [distance]])
                    Kalman.predict(np.array([[latitude], [longitude], [person_center_x], [person_center_y], [distance_meters]]))
                    Kalman.update(z)
                # Disegna il bounding box e la distanza
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if distance_meters is not None:
                    
                    cv2.putText(rgb_image, f"{label} distance: {distance_meters:.2f}m Lat: {x[0,0]:.6f} Lon: {x[1,0]:.6f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
    # Mostra il frame con le rilevazioni e le distanze
    cv2.imshow('Output', rgb_image)

    # Premi 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
