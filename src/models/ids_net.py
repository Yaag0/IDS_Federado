import torch.nn as nn
import torch.nn.functional as F

class IDSNetBinario(nn.Module):
    """
    Red neuronal profunda optimizada.
    
    Arquitectura:
    - Capas lineales densas con reducción progresiva (256 -> 128 -> 64 -> 1).
    - Batch Normalization para estabilizar el entrenamiento.
    - Dropout (0.3) para prevenir el sobreajuste.
    - Salida tipo logit (sin activación final) para compatibilidad con BCEWithLogitsLoss.
    """
    def __init__(self, input_dim: int):
        super(IDSNetBinario, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x