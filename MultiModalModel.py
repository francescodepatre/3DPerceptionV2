import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2
import numpy as np

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
        batch_size, seq_length, _, _= rgb_sequence.size()
        
        # Estrazione delle feature per ogni frame nella sequenza
        rgb_features = []
        for t in range(seq_length):
            rgb_frame = rgb_sequence[:, t, :, :]  # Prendi il t-esimo frame di tutta la sequenza
            rgb_frame = rgb_frame.unsqueeze(1)  # Aggiunge il canale per trasformare da [batch_size, altezza, larghezza] a [batch_size, 1, altezza, larghezza]
            rgb_frame = rgb_frame.repeat(1, 3, 1, 1)  # Duplica il canale per ottenere [batch_size, 3, altezza, larghezza]
            rgb_feat = self.cnn_rgb(rgb_frame)  # Estrai le feature dal frame
            rgb_features.append(rgb_feat)

        # Estrazione delle feature numeriche per ogni frame nella sequenza
        numeric_features = []
        for t in range(seq_length):
            numeric_data = numeric_sequence#[:, t, :]  # Prendi il t-esimo set di dati numerici
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
