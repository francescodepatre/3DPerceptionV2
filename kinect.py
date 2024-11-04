import cv2
import numpy as np
from ultralytics import YOLO
from kinect_sensor import Kinect
from kalman_filter import KalmanFilter

'''
# Funzione asincrona per ottenere la posizione GPS su Windows
async def get_location():
    geolocator = Geolocator()
    try:
        position = await geolocator.get_geoposition_async()
        latitude = position.coordinate.point.position.latitude
        longitude = position.coordinate.point.position.longitude
        return latitude, longitude
    except Exception as e:
        print("Errore nel determinare la posizione:", e)
        return None, None
'''

# Carica il modello YOLO addestrato
model = YOLO("last.pt")

Kinect = Kinect()

# Carica le classi dal file obj.names
with open('obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]


# Inizializzazione del GPS Linux
'''
session = gps.gps()
session.stream(gps.WATCH_ENABLE | gps.WATCH_NEWSTYLE)
'''
# Stato iniziale per il filtro di Kalman: [latitudine, longitudine, distanza] //floppy
'''
floppy
x = np.array([[0], [0], [0]]) 
A = np.eye(3)  # Matrice di transizione dello stato
H = np.eye(3)  # Matrice delle misurazioni
P = np.eye(3)  # Covarianza iniziale dell'errore
Q = np.eye(3) * 0.1  # Covarianza del rumore di processo
R = np.eye(3) * 1.0  # Covarianza del rumore di misurazione
'''
# Stato iniziale per il filtro di Kalman: [distanza, vel_distantza, latitudine, vel_latitudine, longitudine, vel_longitudine] //fra
x0 = np.array([[0], [0], [0], [0], [0], [0]])
dt = 1 #tempo tra due misurazioni consecutive ad esempio il tempo trascorso tra i fotogrammi(presumo da modificare)
#Matrici del modello di Kalman
F = np.array([[ 1, dt, 0, 0, 0, 0 ],
              [ 0, 1, 0, 0, 0,  0 ],
              [ 0, 0, 1, dt, 0, 0 ],
              [ 0, 0, 0, 1, 0, 0  ],
              [ 0, 0, 0, 0, 1, dt ],
              [ 0, 0, 0, 0, 0, 1  ]])

H = np.array([[ 1, 0, 0, 0, 0, 0 ],
              [ 0, 0, 1, 0, 0, 0 ],
              [ 0, 0, 0, 0, 1, 0 ],])

Q = np.eye(6) * 0.01    #Rumore di processo     (Presumo da Cambiare)
R = np.eye(3) * 0.1     #Rumore di misurazione  (Presumo da Cambiare)

P0 = np.eye(6)          #Covarianza Iniziale    (Presumo da Cambiare)

Kalman = KalmanFilter(F, np.zeros((6, 1)), H, Q, R, x0, P0)

#estrai i dati gps
latitude, longitude = get_location()
while True:
    # Cattura i frame dal Kinect
    rgb_image = Kinect.get_realtime_video()
    depth_map = Kinect.get_realtime_depth()

    results = model(rgb_image)  # Passa il frame direttamente al modello

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
                # Forse la posizione GPS può essere eseguita una volta sola, visto che la posizione non cambia
                # Estrai i dati GPS
                ''' Per Linux
                try:
                    report = session.next()
                    if report['class'] == 'TPV':
                        lat = getattr(report, 'lat', 0.0)
                        lon = getattr(report, 'lon', 0.0)
                    else:
                        lat, lon = 0.0, 0.0
                except KeyError:
                    lat, lon = 0.0, 0.0
                '''
                 # Ottieni la posizione GPS su Windows
                #lat, lon = asyncio.run(get_location())

                '''
                # Aggiorna lo stato usando il filtro di Kalman
                if distance_meters is not None:
                    z = np.array([[lat], [lon], [distance_meters]])
                    x, P = kalman_update(x, P, z, H, R)
                '''
                if distance_meters is not None:
                    z = np
                # Disegna il bounding box e la distanza
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if distance_meters is not None:
                    '''
                    cv2.putText(rgb_image, f"{label} distance: {distance_meters:.2f}m Lat: {x[0,0]:.6f} Lon: {x[1,0]:.6f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    '''
                    cv2.putText(rgb_image, f"{label} distance: {distance_meters:.2f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Mostra il frame con le rilevazioni e le distanze
    cv2.imshow('Output', rgb_image)

    # Premi 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
