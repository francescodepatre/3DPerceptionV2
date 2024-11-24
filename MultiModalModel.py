import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2
import numpy as np

class MultiModalLSTMModel(nn.Module):
    def __init__(self):
        super(MultiModalLSTMModel, self).__init__()
        # Feature extractor RGB
        self.cnn_rgb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn_rgb.fc = nn.Identity()  

        # Network for numeric data
        self.fc_numeric = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=512 + 32, hidden_size=256, num_layers=1, batch_first=True)

        # Layer for prediction
        self.fc_final = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Prediction
        )

    def forward(self, rgb_sequence, numeric_sequence):
        batch_size, seq_length, _, _= rgb_sequence.size()
        
        # Feature extractor
        rgb_features = []
        for t in range(seq_length):
            rgb_frame = rgb_sequence[:, t, :, :]  
            rgb_frame = rgb_frame.unsqueeze(1)  
            rgb_frame = rgb_frame.repeat(1, 3, 1, 1)  
            rgb_feat = self.cnn_rgb(rgb_frame)  # Feature extractor for frame
            rgb_features.append(rgb_feat)

        # Feature extractor for numeric
        numeric_features = []
        for t in range(seq_length):
            numeric_data = numeric_sequence#[:, t, :] 
            num_feat = self.fc_numeric(numeric_data)  # Feature extracor for frame
            numeric_features.append(num_feat)

        # Concat
        combined_features = [torch.cat((rgb_features[t], numeric_features[t]), dim=1) for t in range(seq_length)]
        combined_features = torch.stack(combined_features, dim=1)  

        #LSTM
        lstm_out, _ = self.lstm(combined_features)  

        # Output last frame of sequence
        final_out = lstm_out[:, -1, :]  

        # Output to layer fully connected for final prediciton
        output = self.fc_final(final_out)  

        return output
