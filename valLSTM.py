import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import itertools
import MultiModalModel
import torchvision.transforms as transforms


'''class MultiModalLSTMModel(nn.Module):
    def __init__(self):
        super(MultiModalLSTMModel, self).__init__()
        # Feature extractor per RGB
        self.cnn_rgb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn_rgb.fc = nn.Identity()  # Rimuovi l'ultimo layer per ottenere le feature [batch_size, 512]

        # Rete per i dati numerici (depth, GPS, orientamento)
        self.fc_numeric = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # LSTM per catturare il contesto temporale
        self.lstm = nn.LSTM(input_size=512 + 32, hidden_size=256, num_layers=1, batch_first=True)

        # Layer finale per la predizione
        self.fc_final = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Predizione della distanza
        )

    def forward(self, rgb_feature, numeric_feature):
        # Concatenazione delle feature per ogni bounding box
        combined_features = torch.cat((rgb_feature, numeric_feature), dim=1)  # [batch_size, 544]

        # Aggiungi una dimensione per la sequenza (richiesta da LSTM)
        combined_features = combined_features.unsqueeze(1)  # [batch_size, seq_length=1, 544]

        # Passa la sequenza alla LSTM
        lstm_out, _ = self.lstm(combined_features)  # lstm_out: [batch_size, seq_length=1, hidden_size]

        # Prendi l'output dell'ultimo elemento della sequenza
        final_out = lstm_out[:, -1, :]  # Forma: [batch_size, hidden_size]

        # Passa l'output al layer fully connected per la predizione finale
        output = self.fc_final(final_out)  # Forma: [batch_size, 1]

        return output
'''
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica il modello salvato e spostalo sulla GPU (se disponibile)
    model = MultiModalModel.MultiModalLSTMModel().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()  # Modalità valutazione
    yolo_model = YOLO("last.pt")

    # Apertura dei video RGB e di profondità per la valutazione
    video_rgb = cv2.VideoCapture("5_rgb.mp4")
    video_depth = cv2.VideoCapture("5_depth.mp4")

    # Carica il file delle distanze ground truth
    ground_truth_file = "depth_5.txt"
    ground_truth_data = {}

    with open(ground_truth_file, "r") as file:
        for line in file:
            # Dividi la riga usando ':' come delimitatore
            parts = line.split(':')
            
            # Estrai frame index e distanze delle bounding box
            frame_idx = int(parts[0].strip())
            distances = parts[-1].strip().split(',')  # Prendi tutte le distanze e dividile sulla virgola
            distances = [float(d.strip()) for d in distances]  # Converte ogni distanza in float

            # Salva le distanze in un dizionario con la chiave del frame
            ground_truth_data[frame_idx] = distances

    # Variabili per tracciare l'errore
    total_error = 0.0
    total_predictions = 0
    rgb_sequence = []
    numeric_sequence = []
    counter = 0
    sequence_length = 10
    # Loop attraverso i frame del video
    frame_idx = 0
    with tqdm(total=int(video_rgb.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Evaluating frames") as pbar:
        while video_rgb.isOpened() and video_depth.isOpened():
            ret_rgb, frame_rgb = video_rgb.read()
            ret_depth, frame_depth = video_depth.read()

            if not ret_rgb or not ret_depth:
                break

            # Prepara YOLO per rilevare le bounding box
            results = yolo_model.predict(frame_rgb)
            frame_distances = ground_truth_data.get(frame_idx, [])

            # Raccogli tutte le feature del frame corrente
            predictions = []

            for r in results:
                for i, box in enumerate(r.boxes):
                    if i >= len(frame_distances):
                        # Se ci sono più bounding box di quelle annotate, ignorale
                        break
                    transform = transforms.Compose([
                        transforms.ToPILImage(),  # Converte l'immagine OpenCV (numpy array) in PIL
                        transforms.Resize((224, 224)),  # Ridimensiona l'immagine a 224x224
                        transforms.ToTensor(),  # Converte l'immagine PIL in Tensor (con dimensioni [C, H, W])
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza come richiesto da ResNet
                    ])
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

                    # Ritaglio dell'immagine della bounding box
                    cropped_face = frame_rgb[y_min:y_max, x_min:x_max]

                    # Salta bounding box non valide
                    if cropped_face.size == 0:
                        continue

                    # Resize e preparazione per il modello
                    #cropped_face = cv2.resize(cropped_face, (224, 224))
                    #cropped_face_tensor = torch.tensor(cropped_face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                    frame_rgb_tensor = transform(cropped_face)  # Risultato: [3, 224, 224]
                    frame_rgb_tensor = frame_rgb_tensor.unsqueeze(0).to(device)  # Aggiungi una dimensione per il batch: [1, 3, 224, 224]
                    rgb_sequence.append(frame_rgb_tensor)
                    #rgb_sequence.append(cropped_face_tensor)
                    # Estrazione delle feature RGB
                    #rgb_feat = model.cnn_rgb(cropped_face_tensor)

                    # Ottieni i dati GPS simulati e orientamento della bussola (esempio)
                    gps_data = [45.464203, 9.189982]
                    compass_orientation = 45.0  # Orientamento simulato della bussola

                    # Prepara i dati numerici
                    depth_value = frame_distances[i]
                    numeric_data = [depth_value, gps_data[0], gps_data[1], compass_orientation]
                    numeric_tensor = torch.tensor(numeric_data, dtype=torch.float32).unsqueeze(0).to(device)
                    numeric_sequence.append(numeric_tensor)
                    # Estrazione delle feature numeriche
                    #num_feat = model.fc_numeric(numeric_tensor)

                    # Passa le feature al modello per ottenere la predizione
                    #prediction = model(rgb_feat, num_feat)
                    prediction = model(rgb_sequence, numeric_sequence)
                    # Calcola l'errore per ogni bounding box
                    prediction_value = prediction.item()*2.5  # Converti il tensor in un singolo valore
                    error = abs(prediction_value - depth_value)
                    total_error += error
                    total_predictions += 1
                    counter +=1
                    if (counter % sequence_length) == 0:
                        rgb_sequence.clear()
                        numeric_sequence.clear() 
                    # Aggiungi la predizione alla lista
                    predictions.append(prediction_value)

            frame_idx += 1
            pbar.update(1)

    # Calcola l'errore medio
    mean_error = total_error / total_predictions if total_predictions > 0 else float('inf')
    print(f"Errore medio sulle predizioni: {mean_error:.4f} metri")

    # Chiudi il video
    video_rgb.release()
    video_depth.release()