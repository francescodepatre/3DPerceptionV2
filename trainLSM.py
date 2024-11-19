import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import queue
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import os

class MultiModalLSTMModel(nn.Module):
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

    def forward(self, rgb_sequence, numeric_sequence):
        batch_size, seq_length, _, _, _ = rgb_sequence.size()

        # Estrazione delle feature per ogni frame nella sequenza
        rgb_features = []
        for t in range(seq_length):
            rgb_frame = rgb_sequence[:, t, :, :, :]  # Prendi il t-esimo frame di tutta la sequenza
            rgb_feat = self.cnn_rgb(rgb_frame)  # Estrai le feature dal frame
            rgb_features.append(rgb_feat)

        # Estrazione delle feature numeriche per ogni frame nella sequenza
        numeric_features = []
        for t in range(seq_length):
            numeric_data = numeric_sequence[:, t, :]  # Prendi il t-esimo set di dati numerici
            num_feat = self.fc_numeric(numeric_data)  # Estrai le feature dai dati numerici
            numeric_features.append(num_feat)

        # Concatenazione delle feature per ogni frame
        combined_features = [torch.cat((rgb_features[t], numeric_features[t]), dim=1) for t in range(seq_length)]
        combined_features = torch.stack(combined_features, dim=1)  # Forma: [batch_size, seq_length, 544]

        # Passa la sequenza alla LSTM
        lstm_out, _ = self.lstm(combined_features)  # lstm_out: [batch_size, seq_length, hidden_size]

        # Prendi l'output dell'ultimo frame della sequenza
        final_out = lstm_out[:, -1, :]  # Forma: [batch_size, hidden_size]

        # Passa l'output al layer fully connected per la predizione finale
        output = self.fc_final(final_out)  # Forma: [batch_size, 1]

        return output

class VideoDepthDataset(Dataset):
    def __init__(self, rgb_path, depth_path, sequence_length=10):
        self.rgb_files = sorted(glob.glob(rgb_path + "/*.mp4"))
        self.depth_files = sorted(glob.glob(depth_path + "/*.mp4"))
        self.sequence_length = sequence_length
        self.yolo_model = YOLO("./last.pt")  # Inizializza il modello YOLO per la rilevazione

        # Assicurati che il numero di video RGB e di profondità corrispondano
        assert len(self.rgb_files) == len(self.depth_files), "Il numero di video RGB e di profondità deve essere uguale"

    def __len__(self):
        # La lunghezza del dataset è approssimata dal numero di video, ma in realtà dipenderà dal numero totale di frame disponibili
        total_frames = 0
        for rgb_file in self.rgb_files:
            video_rgb = cv2.VideoCapture(rgb_file)
            total_frames += int(video_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        return total_frames // self.sequence_length

    def __getitem__(self, idx):
        video_index = idx // (self.__len__() // len(self.rgb_files))
        start_frame_index = (idx % (self.__len__() // len(self.rgb_files))) * self.sequence_length

        # Carica i video RGB e di profondità
        video_rgb = cv2.VideoCapture(self.rgb_files[video_index])
        video_depth = cv2.VideoCapture(self.depth_files[video_index])

        rgb_sequence = []
        numeric_sequence = []
        depth_values = []

        # Salta i frame fino al punto di inizio desiderato
        video_rgb.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        video_depth.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

        frame_count = 0
        while frame_count < self.sequence_length:
            ret_rgb, frame_rgb = video_rgb.read()
            ret_depth, frame_depth = video_depth.read()

            if not ret_rgb or not ret_depth:
                break

            # Rileva le bounding box con YOLO
            results = self.yolo_model.predict(frame_rgb)

            # Itera sulle rilevazioni e processa le bounding box
            box_found = False  # Aggiungi questo flag per verificare se è stata trovata almeno una bounding box
            for r in results:
                for box in r.boxes:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

                    # Ritaglia il volto e la profondità corrispondente
                    cropped_face = frame_rgb[y_min:y_max, x_min:x_max]
                    depth_cropped = frame_depth[y_min:y_max, x_min:x_max]

                    # Normalizzazione della profondità
                    depth_normalized = ((255 - depth_cropped) / 255.0) * 5.5
                    depth_value = np.mean(depth_normalized)

                    # Verifica se il valore di profondità è valido
                    if np.isnan(depth_value) or np.isinf(depth_value):
                        depth_value = 0.0

                    # Simula i dati GPS e l'orientamento della bussola
                    gps_data = [45.464203, 9.189982]
                    compass_orientation = np.random.uniform(0, 360)

                    # Prepara l'immagine e i dati numerici
                    cropped_face = cv2.resize(cropped_face, (224, 224))
                    cropped_face_tensor = torch.tensor(cropped_face, dtype=torch.float32).permute(2, 0, 1)

                    numeric_data = [depth_value, gps_data[0], gps_data[1], compass_orientation]
                    numeric_tensor = torch.tensor(numeric_data, dtype=torch.float32)

                    # Aggiungi alla sequenza
                    rgb_sequence.append(cropped_face_tensor)
                    numeric_sequence.append(numeric_tensor)
                    depth_values.append(depth_value) # Aggiungi il target di profondità
                    print(f'Depth_value: {depth_value}')

                    frame_count += 1
                    box_found = True  # Segnala che una bounding box è stata trovata

                    if frame_count >= self.sequence_length:
                        break

            # Se nessuna bounding box è stata trovata, aggiungi un valore di riempimento
            if not box_found:
                rgb_sequence.append(torch.zeros(3, 224, 224))
                numeric_sequence.append(torch.zeros(4))
                depth_values.append(0.0)
                frame_count += 1

        # Converte le sequenze in tensor PyTorch
        if len(rgb_sequence) < self.sequence_length:
            padding_length = self.sequence_length - len(rgb_sequence)
            rgb_sequence.extend([torch.zeros(3, 224, 224)] * padding_length)
            numeric_sequence.extend([torch.zeros(4)] * padding_length)
            depth_values.extend([0.0] * padding_length)  # Riempie con 0 per la profondità

        rgb_sequence_tensor = torch.stack(rgb_sequence)
        numeric_sequence_tensor = torch.stack(numeric_sequence)
        depth_values_tensor = torch.tensor(depth_values, dtype=torch.float32)

        return rgb_sequence_tensor, numeric_sequence_tensor, depth_values_tensor

if __name__ == "__main__":

    dataset = VideoDepthDataset("rgb", "depth", sequence_length=10)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Preparazione del modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalLSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_function = nn.MSELoss()

    if os.path.exists("modello_multimodale.pth"):
        model.load_state_dict(torch.load("modello_multimodale.pth"))

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for batch_idx, (rgb_sequence, numeric_sequence, depth_values) in enumerate(data_loader):
                rgb_sequence = rgb_sequence.to(device)
                numeric_sequence = numeric_sequence.to(device)
                depth_values = depth_values.to(device)  # Sposta i target sulla GPU

                # Target simulato (sostituiscilo con il tuo target reale)
                targets = depth_values.unsqueeze(1)
                
                optimizer.zero_grad()
                predictions = model(rgb_sequence, numeric_sequence)
                print(f'Predictions: {predictions}')
                print(f'Targets: {targets}')
                loss = loss_function(predictions, targets)
                if torch.isnan(loss):
                    print("Loss è nan, saltando questo batch")
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)

        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_epoch_loss:.4f}")

    # Salva il modello addestrato
    torch.save(model.state_dict(), "modello_multimodale.pth")