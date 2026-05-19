import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 1. Архитектура "Бутылочное горлышко" 33-64-16-8
class UnicodeAutoencoder(nn.Module):
    def __init__(self, bits=32, emb_dim=8):
        super().__init__()
        # Энкодер: плавное сжатие
        self.encoder = nn.Sequential(
            nn.Linear(36, 64), # 32 бита + 1 sin
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Linear(16, emb_dim)
        )
        # Декодер: восстановление исходных 32 бит
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, bits),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_embedding(self, x):
        return self.encoder(x)

# 2. Генерация данных (Цифры, Латиница, Кириллица)
def prepare_unicode_dataset():
    ranges = [range(48, 58), range(65, 91), range(1040, 1104)] 
    x_input, labels, colors = [], [], []
    
    for i, r in enumerate(ranges):
        for code in r:
            # 1. Биты в диапазоне [-1, 1]
            bits = [int(b) * 2 - 1 for b in format(code, '032b')]
            
            # 2. Несколько частот для sin/cos (Positional Encoding)
            # Чтобы соседние коды (A=65, B=66) давали заметную разницу
            freq = code / 100.0 
            extra = [np.sin(freq), np.cos(freq), np.sin(freq/10), np.cos(freq/10)]
            
            x_input.append(bits + extra)
            labels.append(chr(code))
            colors.append(i)
            
    return torch.FloatTensor(x_input), labels, colors

# 3. Цикл обучения
data, labels, colors = prepare_unicode_dataset()
model = UnicodeAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

print("Начало обучения...")
for epoch in range(1001):
    optimizer.zero_grad()
    outputs = model(data)
    # Обучаем восстанавливать только биты (первые 32 колонки)
    loss = criterion(outputs, data[:, :32])
    loss.backward()
    optimizer.step()
    if epoch % 250 == 0:
        print(f"Эпоха {epoch}, Ошибка: {loss.item():.6f}")

# 4. Визуализация t-SNE
model.eval()
with torch.no_grad():
    embeddings = model.get_embedding(data).numpy()

# Сжимаем 8D в 2D для графика
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
vis = tsne.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(vis[:, 0], vis[:, 1], c=colors, cmap='rainbow', edgecolors='k')
for i, txt in enumerate(labels):
    plt.annotate(txt, (vis[i, 0], vis[i, 1]), xytext=(3,3), textcoords='offset points')

plt.title("Эммерджентная кластеризация Unicode (Эмбеддинги 8D)")
plt.show()
