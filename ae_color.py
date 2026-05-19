import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Проверяем доступность CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

# ==========================================
# 1. АРХИТЕКТУРА (4 -> 128 -> 2)
# ==========================================
class ColorAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Увеличим ширину слоев для обработки миллионов комбинаций
        self.encoder = nn.Sequential(
            nn.Linear(4, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 2) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 3), 
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = ColorAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ==========================================
# 2. ОБУЧЕНИЕ НА ГЕНЕРИРУЕМЫХ ДАННЫХ
# ==========================================

epochs = 5000
batch_size = 65536  # Огромный батч для CUDA

print("Начало масштабного обучения на RGB пространстве...")
for epoch in range(epochs):
    # Генерируем случайные RGB прямо на GPU
    rgb = torch.rand(batch_size, 3, device=device)
    
    # Считаем яркость (векторизованно на CUDA)
    brightness = (rgb[:, 0]*0.299 + rgb[:, 1]*0.587 + rgb[:, 2]*0.114).view(-1, 1)
    
    # Собираем вход (RGB + Brightness) и нормализуем в [-1, 1]
    x_input = torch.cat([rgb, brightness], dim=1) * 2 - 1
    
    optimizer.zero_grad()
    outputs = model(x_input)
    
    # Цель - исходные RGB (нормализованные в [-1, 1])
    loss = criterion(outputs, x_input[:, :3])
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.8f}")

# ==========================================
# 3. ВИЗУАЛИЗАЦИЯ (тестовая выборка)
# ==========================================
model.eval()
test_samples = 50000 # Возьмем 50к точек для отрисовки, чтобы не завис график
with torch.no_grad():
    rgb_test = torch.rand(test_samples, 3, device=device)
    bright_test = (rgb_test[:, 0]*0.299 + rgb_test[:, 1]*0.587 + rgb_test[:, 2]*0.114).view(-1, 1)
    x_test = torch.cat([rgb_test, bright_test], dim=1) * 2 - 1
    
    embeddings = model.encoder(x_test).cpu().numpy()
    rgb_colors = rgb_test.cpu().numpy()

plt.figure(figsize=(12, 10))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=rgb_colors, s=1, alpha=0.5)
plt.title("Эммерджентное RGB пространство (обучено на всем спектре)")
plt.xlabel("Latent 1")
plt.ylabel("Latent 2")
plt.show()
