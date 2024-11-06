import cv2
import numpy as np
import folium
from folium import IFrame
from ultralytics import YOLO
from kalman_filter import KalmanFilter
import requests
from scipy.spatial.distance import cdist

# Inizializzazione delle variabili
model = YOLO("last.pt")

# Inizializzazione dei video RGB e Depth
cap_rgb = cv2.VideoCapture('0_rgb.mp4')  # Sostituisci con il percorso del tuo video RGB
cap_depth = cv2.VideoCapture('0_dpth.mp4')  # Sostituisci con il percorso del tuo video Depth

# Carica le classi
with open('obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Dizionario per memorizzare i filtri di Kalman per ogni persona e le loro posizioni
kalman_filters = {}
next_id = 0  # ID per assegnare nuovi oggetti

# Funzione per ottenere la posizione GPS
def get_location(file_path):
    try:
        with open(file_path, 'r') as file:
            # Leggi le righe dal file
            latitude = float(file.readline().strip())
            longitude = float(file.readline().strip())
        
        # Restituisci la latitudine e la longitudine lette dal file
        return latitude, longitude
    except (IOError, ValueError) as e:
        # Stampa un messaggio di errore se non è possibile leggere il file o convertire i valori
        print(f"Errore nella lettura del file o nel parsing delle coordinate: {e}")
        return None, None
    #return float(44.779328), float(10.315408)

# Funzione per creare e aggiornare la mappa
def update_map(kinect_lat, kinect_lon, face_positions):
    # Ordina face_positions in base alla distanza in modo decrescente
    face_positions_sorted = sorted(face_positions, key=lambda x: x[0], reverse=True)

    # Crea una mappa centrata sulla posizione del Kinect
    m = folium.Map(location=[kinect_lat, kinect_lon], zoom_start=15)

    # Aggiungi un marker per la posizione del Kinect
    folium.Marker([kinect_lat, kinect_lon], popup='Kinect Location', icon=folium.Icon(color='blue')).add_to(m)

    # Disegna una circonferenza per ogni volto rilevato in ordine di distanza (dal più lontano al più vicino)
    for dist_meters, obj_id in face_positions_sorted:
        popup_text = f"Person ID: {obj_id}<br>Distance: {dist_meters:.2f} m"
        popup = folium.Popup(popup_text, max_width=250)

        folium.Circle(
            location=[kinect_lat, kinect_lon],
            radius=dist_meters ,  # Convertiamo la distanza in metri in un raggio per la visualizzazione
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.2,
            popup=popup
        ).add_to(m)

    # Salva la mappa in un file HTML temporaneo
    m.save('kinect_map.html')

latitude, longitude = get_location('0.txt')
while True:
    # Cattura i frame dai video RGB e Depth
    ret_rgb, rgb_image = cap_rgb.read()
    ret_depth, depth_map = cap_depth.read()

    # Verifica se i video sono terminati
    if not ret_rgb or not ret_depth:
        break

    # Converti il frame di depth in scala di grigi
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

    # Converti la depth_map da valori di 0-255 a metri
    MAX_DEPTH = 4.0  # La profondità massima in metri
    depth_map_meters = ((255 - depth_map) / 255.0) * MAX_DEPTH

    # Processa il frame RGB con YOLO per il rilevamento dei volti
    results = model(rgb_image)  # Passa il frame direttamente al modello
    #latitude, longitude = get_location()

    # Lista delle nuove rilevazioni per la mappa e disegni sul frame corrente
    face_positions = []

    # Parsing dei risultati YOLO per estrarre i centri delle bounding box
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            if conf > 0.5 and cls == 0:  # Filtra per la classe 'persona'
                person_center_x = x1 + (x2 - x1) // 2
                person_center_y = y1 + (y2 - y1) // 2
                detections.append((person_center_x, person_center_y, x1, y1, x2, y2))

    # Matrice per tenere traccia dei centri stimati dai filtri di Kalman esistenti
    kalman_predictions = []

    # Ottieni le predizioni dai filtri di Kalman esistenti
    print(f"[INFO] Numero di filtri di Kalman attivi: {len(kalman_filters)}")
    for obj_id, kalman in kalman_filters.items():
        u = np.zeros((kalman.B.shape[1], 1))  # Vettore di controllo nullo
        kalman.predict(u)  # Chiama predict passando u
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

            if distances[pred_idx, det_idx] < 100:  # Soglia di distanza, può essere modificata
                predicted_obj_id = kalman_predictions[pred_idx][2]
                det_center_x, det_center_y, x1, y1, x2, y2 = detections[det_idx]

                if latitude is not None and longitude is not None:
                    if 0 <= det_center_x < depth_map_meters.shape[1] and 0 <= det_center_y < depth_map_meters.shape[0]:
                        distance_meters = depth_map_meters[det_center_y, det_center_x]

                        z = np.array([[latitude], [longitude], [distance_meters]])
                        kalman_filters[predicted_obj_id].update(z)

                        # Aggiungi alla lista per la mappa l'ID e la distanza
                        distance_kalman = kalman_filters[predicted_obj_id].x[4, 0]
                        face_positions.append((distance_kalman, predicted_obj_id))

                        # Disegna il bounding box e aggiungi ID e distanza
                        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(rgb_image, f"ID: {predicted_obj_id}, Distance: {distance_kalman:.2f} m",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Rimuovi le corrispondenze già abbinate
                distances = np.delete(distances, pred_idx, axis=0)
                distances = np.delete(distances, det_idx, axis=1)
                kalman_predictions.pop(pred_idx)
                detections.pop(det_idx)
            else:
                break

    # Crea nuovi filtri di Kalman per le rilevazioni che non sono state abbinate e disegna il bounding box
    for (x, y, x1, y1, x2, y2) in detections:
        if latitude is not None and longitude is not None:
            if 0 <= y < depth_map_meters.shape[0] and 0 <= x < depth_map_meters.shape[1]:
                distance_meters = depth_map_meters[y, x]
                x0 = np.array([[latitude], [longitude], [x], [y], [distance_meters]])
                F = np.eye(5)
                H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]])
                Q = np.eye(5) * 0.01
                R = np.eye(3) * 0.1
                P0 = np.eye(5)

                kalman_filters[next_id] = KalmanFilter(F, np.zeros((5, 1)), H, Q, R, x0, P0)

                face_positions.append((distance_meters, next_id))

                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(rgb_image, f"ID: {next_id} (NEW)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                next_id += 1

    # Aggiorna la mappa con la posizione del Kinect e le posizioni rilevate nel frame corrente
    if latitude is not None and longitude is not None:
        update_map(latitude, longitude, face_positions)

    # Mostra il frame con le rilevazioni e le distanze
    cv2.imshow('Output', rgb_image)

    # Premi 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_rgb.release()
cap_depth.release()
cv2.destroyAllWindows()
