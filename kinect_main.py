import cv2
import numpy as np
import folium
from folium import IFrame
from ultralytics import YOLO
from kinect_sensor import Kinect
from kalman_filter import KalmanFilter
import requests
from scipy.spatial.distance import cdist

# Inizializzazione delle variabili
model = YOLO("/home/francesco-de-patre/Scrivania/3DPerceptionV2/last.pt")
Kinect = Kinect()

# Carica le classi
with open('/home/francesco-de-patre/Scrivania/3DPerceptionV2/obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Dizionario per memorizzare i filtri di Kalman per ogni persona e le loro posizioni
kalman_filters = {}
next_id = 0  # ID per assegnare nuovi oggetti

Q = np.eye(5) * 0.01    #Rumore di processo     (Presumo da Cambiare)
R = np.eye(3) * 0.1     #Rumore di misurazione  (Presumo da Cambiare)

P0 = np.eye(5)          #Covarianza Iniziale    (Presumo da Cambiare)
# Funzione per ottenere la posizione GPS
def get_location():
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        location = data['loc'].split(',')
        latitude = float(location[0])
        longitude = float(location[1])
        return latitude, longitude
    except requests.exceptions.RequestException:
        return None, None

# Funzione per creare e aggiornare la mappa
def update_map(kinect_lat, kinect_lon, face_positions):
    # Crea una mappa centrata sulla posizione del Kinect
    m = folium.Map(location=[kinect_lat, kinect_lon], zoom_start=15)

    #icon = folium.Icon(icon="cloud", color="blue", icon_size=(5, 5))
    print(kinect_lat,kinect_lon)
    # Aggiungi un marker per la posizione del Kinect
    folium.Marker([kinect_lat, kinect_lon], popup='Kinect Location', icon=folium.Icon(color='blue')).add_to(m)


    # Disegna una circonferenza per ogni volto rilevato
    for idx, (dist_meters, obj_id) in enumerate(face_positions):
        folium.Circle(
            location=[kinect_lat+dist_meters, kinect_lon],
            radius=dist_meters * 100,  # Convertiamo la distanza in metri in un raggio per la visualizzazione
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.2,
            popup=IFrame(f"Person ID: {obj_id}", width=100, height=50)
        ).add_to(m)

    # Salva la mappa in un file HTML temporaneo
    m.save('kinect_map.html')

latitude, longitude = 44.801472, 10.328000

while True:
    # Cattura i frame dal Kinect
    rgb_image = Kinect.get_realtime_video()
    depth_map = Kinect.get_realtime_depth()

    results = model(rgb_image)  # Passa il frame direttamente al modello
    

    # Lista delle nuove rilevazioni
    detections = []

    # Lista delle posizioni delle facce per la mappa
    face_positions = []

    # Parsing dei risultati YOLO per estrarre i centri delle bounding box
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            if conf > 0.5 and cls == 0:  # Filtra per la classe 'persona'
                person_center_x = x1 + (x2 - x1) // 2
                person_center_y = y1 + (y2 - y1) // 2

                # Aggiungi la rilevazione alla lista
                detections.append((person_center_x, person_center_y, x1, y1, x2, y2))

    # Lista per memorizzare gli oggetti da rimuovere (perché non più rilevati)
    ids_to_remove = []

    # Matrice per tenere traccia dei centri stimati dai filtri di Kalman esistenti
    kalman_predictions = []

    # Ottieni le predizioni dai filtri di Kalman esistenti
    for obj_id, kalman in kalman_filters.items():
        u = np.zeros((kalman.B.shape[1], 1))  # Vettore di controllo nullo
        kalman.predict(u)
        predicted_x, predicted_y = kalman.x[2, 0], kalman.x[3, 0]
        kalman_predictions.append((predicted_x, predicted_y, obj_id))

    # Abbina le nuove rilevazioni agli oggetti esistenti utilizzando la distanza euclidea
    if kalman_predictions and detections:
        pred_coords = np.array([(x, y) for x, y, _ in kalman_predictions])
        det_coords = np.array([(x, y) for x, y, _, _, _, _ in detections])

        distances = cdist(pred_coords, det_coords, 'euclidean')

        # Abbinamento delle predizioni con le nuove rilevazioni
        while distances.size > 0:
            min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)
            pred_idx, det_idx = min_dist_idx

            # Se la distanza è al di sotto di una certa soglia, abbina la rilevazione al filtro
            if distances[pred_idx, det_idx] < 100:  # Soglia di distanza, può essere modificata
                predicted_obj_id = kalman_predictions[pred_idx][2]
                det_center_x, det_center_y, x1, y1, x2, y2 = detections[det_idx]

                # Aggiorna il filtro di Kalman con la nuova rilevazione
                if latitude is not None and longitude is not None:
                    distance = depth_map[det_center_y, det_center_x]
                    distance_meters = distance / 1000.0
                    z = np.array([[latitude], [longitude], [distance_meters]])
                    kalman_filters[predicted_obj_id].update(z)

                    # Aggiungi alla lista per la mappa
                    face_positions.append((distance_meters, predicted_obj_id))

                    # Disegna il bounding box
                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(rgb_image, f"ID: {predicted_obj_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Rimuovi le corrispondenze già abbinate
                distances = np.delete(distances, pred_idx, axis=0)
                distances = np.delete(distances, det_idx, axis=1)
                kalman_predictions.pop(pred_idx)
                detections.pop(det_idx)
            else:
                break

    # Crea nuovi filtri di Kalman per le rilevazioni che non sono state abbinate e disegna il bounding box
    for (x, y, x1, y1, x2, y2) in detections:
        distance = depth_map[y, x]
        distance_meters = distance / 1000.0 if distance is not None else 0.0
        x0 = np.array([[latitude], [longitude], [x], [y], [distance_meters]])
        F = np.eye(5)
        H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]])
        #Q = np.eye(5) * 0.01
        #R = np.eye(3) * 0.1
        #P0 = np.eye(5)
        kalman_filters[next_id] = KalmanFilter(F,np.zeros((5, 1)), H, Q, R, x0, P0)

        # Aggiungi alla lista per la mappa
        face_positions.append((distance_meters, next_id))

        # Disegna il bounding box per la nuova rilevazione
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(rgb_image, f"ID: {next_id} (NEW)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        next_id += 1

    # Aggiorna la mappa con la posizione del Kinect e le posizioni rilevate
    if latitude is not None and longitude is not None:
        update_map(latitude, longitude, face_positions)

    # Mostra il frame con le rilevazioni e le distanze
    cv2.imshow('Output', rgb_image)

    # Premi 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
