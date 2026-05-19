import torch
import torch.nn as nn
import torch.optim as optim
import os

# Настройки
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "unicode_ae_8bit.pth"
EMBEDDING_DIM = 8

class UnicodeAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, 8), nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.GELU(),
            nn.Linear(16, 32), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_embedding(self, x):
        return self.encoder(x)

def generate_full_unicode_sample():
    """Генерирует репрезентативную выборку из всего пространства Unicode."""
    codes = set()
    # 1. Основные алфавиты (0 - 3000)
    for i in range(0, 3000): codes.add(i)
    # 2. Иероглифы CJK (0x4E00 - 0x9FFF)
    for i in range(0, 1000): codes.add(0x4E00 + i)
    # 3. Эмодзи (0x1F600 - 0x1F64F)
    for i in range(0x1F600, 0x1F64F): codes.add(i)
    # 4. Случайный шум по всему 32-битному пространству (10к точек)
    # for _ in range(10000):
    #     codes.add(torch.randint(0, 0x10FFFF, (1,)).item()) 
    
    # Конвертация в биты
    bits_list = []
    for code in codes:
        bits = [int(b) for b in bin(code)[2:].zfill(32)]
        bits_list.append(bits)
    
    return torch.FloatTensor(bits_list).to(DEVICE)

def train():
    model = UnicodeAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    data = generate_full_unicode_sample()
    batch_size = 512
    epochs = 5000
    
    print(f"Запуск обучения на {len(data)} уникальных символах...")
    
    for epoch in range(epochs):
        # Перемешивание
        indices = torch.randperm(data.size(0))
        epoch_loss = 0
        
        for i in range(0, data.size(0), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = data[batch_indices]
            
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 500 == 0:
            avg_loss = epoch_loss / (len(data) / batch_size)
            print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.8f}")

    # Сохранение всей модели (архитектура + веса)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")

if __name__ == "__main__":
    train()
