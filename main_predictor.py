import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# --- Параметры ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AE_MODEL_PATH = "unicode_ae_8bit.pth"
TEXT_FILE = "input.txt" # Подложи сюда любой текстовый файл
MAX_DIST = 5
BATCH_SIZE = 4096 # Огромный батч для скорости CUDA

# --- Архитектуры (должны совпадать с оригиналом) ---
class UnicodeAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 8), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 32), nn.Sigmoid())
    def get_embedding(self, x): return self.encoder(x)

class TripletPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17, 256),
            nn.LayerNorm(256), # Стабилизирует обучение
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# --- Подготовка данных ---
class NgramDataset(Dataset):
    def __init__(self, text, ae_model):
        self.inputs = []
        self.targets = []
        
        print("Кэширование эмбеддингов...")
        char_cache = {}
        unique_chars = set(text)
        ae_model.eval()
        with torch.no_grad():
            for char in unique_chars:
                bits = torch.FloatTensor([int(b) for b in bin(ord(char))[2:].zfill(32)]).to(DEVICE)
                char_cache[char] = ae_model.get_embedding(bits.unsqueeze(0)).squeeze(0)

        print(f"Сборка триплетов (длина текста: {len(text)})...")
        for i in range(1, len(text) - 1):
            v_curr = char_cache[text[i]]
            v_target = char_cache[text[i+1]]
            for d in range(1, MAX_DIST + 1):
                if i - d >= 0:
                    v_past = char_cache[text[i-d]]
                    dist_val = torch.tensor([d / MAX_DIST]).to(DEVICE)
                    self.inputs.append(torch.cat((v_curr, v_past, dist_val)))
                    self.targets.append(v_target)
                    
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

def train():
    # 1. Загрузка AE
    ae = UnicodeAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load(AE_MODEL_PATH))
    
    # 2. Чтение текста
    if not os.path.exists(TEXT_FILE):
        with open(TEXT_FILE, "w", encoding="utf-8") as f:
            f.write("ПРЕЗИДЕНТ ПРЕКРАСНО ПРЕДСКАЗАЛ ПРЕПЯТСТВИЕ " * 1000)
    
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # 3. DataLoader
    dataset = NgramDataset(text, ae)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Обучение
    model = TripletPredictor().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"Обучение на {len(dataset)} примерах...")
    for epoch in range(50): # 50 эпох хватит для большого текста
        total_loss = 0
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), "predictor_weights.pth")
    print("Веса предиктора сохранены.")

if __name__ == "__main__":
    train()
