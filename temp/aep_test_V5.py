import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 0. НАСТРОЙКА УСТРОЙСТВА (CUDA)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# ==========================================
# 1. АРХИТЕКТУРА ПЕРВОГО АВТОЭНКОДЕРА (ЮНИКОД)
# ==========================================
class UnicodeAutoencoder(nn.Module):
    def __init__(self, bits=32, emb_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(36, 64), nn.GELU(),
            nn.Linear(64, 16), nn.GELU(),
            nn.Linear(16, emb_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 16), nn.GELU(),
            nn.Linear(16, 64), nn.GELU(),
            nn.Linear(64, bits),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==========================================
# 2. ПОДГОТОВКА ДАННЫХ ДЛЯ ЮНИКОД-АВТОЭНКОДЕРА
# ==========================================
def get_char_vector(char):
    code = ord(char)
    bits = [int(b) * 2 - 1 for b in format(code, '032b')]
    freq = code / 100.0
    extra = [np.sin(freq), np.cos(freq), np.sin(freq/10), np.cos(freq/10)]
    return torch.FloatTensor(bits + extra)

def generate_full_unicode_sample():
    """Генерирует репрезентативную выборку из всего пространства Unicode."""
    codes = set()
    for i in range(0, 3000):
        codes.add(i)
    for i in range(0, 1000):
        codes.add(0x4E00 + i)
    for i in range(0x1F600, 0x1F64F):
        codes.add(i)
    return codes

unicode_codes = generate_full_unicode_sample()
alphabet = [chr(c) for c in sorted(unicode_codes)]
all_chars_vecs = torch.stack([get_char_vector(c) for c in alphabet]).to(device)

# ==========================================
# 3. ОБУЧЕНИЕ / ЗАГРУЗКА ПЕРВОГО АВТОЭНКОДЕРА
# ==========================================
ae_model = UnicodeAutoencoder().to(device)
ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)
ae_criterion = nn.MSELoss()

try:
    ae_model.load_state_dict(torch.load("unicode_autoencoder.pth", map_location=device))
    ae_model.eval()
    print("Загружена предобученная модель UnicodeAutoencoder")
except FileNotFoundError:
    print("--- Обучение UnicodeAutoencoder ---")
    for epoch in range(5001):
        ae_model.train()
        ae_optimizer.zero_grad()
        outputs = ae_model(all_chars_vecs)
        loss = ae_criterion(outputs, all_chars_vecs[:, :32])
        loss.backward()
        ae_optimizer.step()
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, AE Loss: {loss.item():.8f}")
    ae_model.eval()
    torch.save(ae_model.state_dict(), "unicode_autoencoder.pth")
    print("Модель UnicodeAutoencoder сохранена")

# ==========================================
# 4. ПОЛУЧЕНИЕ КЭША ЭМБЕДДИНГОВ (8-мерные векторы)
# ==========================================
print("Создание кэша эмбеддингов для всех символов...")
with torch.no_grad():
    # all_chars_vecs уже [N, 36]
    emb_all = ae_model.encoder(all_chars_vecs)  # [N, 8]

# ==========================================
# 5. АВТОЭНКОДЕР СВЯЗЕЙ (ПАРЫ СИМВОЛОВ)
# ==========================================
class PairAutoencoder(nn.Module):
    def __init__(self, input_dim=16, bottleneck_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==========================================
# 6. СОЗДАНИЕ ДАТАСЕТА ВСЕХ УПОРЯДОЧЕННЫХ ПАР
# ==========================================
print("Генерация всех упорядоченных пар символов...")
N = len(alphabet)
# emb_all shape [N, 8]
# Создаем пары через broadcasting: [N, N, 16]
emb_i = emb_all.unsqueeze(1).expand(-1, N, -1)   # [N, N, 8]
emb_j = emb_all.unsqueeze(0).expand(N, -1, -1)   # [N, N, 8]
pair_vectors = torch.cat([emb_i, emb_j], dim=-1)  # [N, N, 16]
pair_vectors = pair_vectors.reshape(-1, 16)        # [N*N, 16]

dataset = TensorDataset(pair_vectors, pair_vectors)  # вход = цель для автоэнкодера
dataloader = DataLoader(dataset, batch_size=65536, shuffle=True)

# ==========================================
# 7. ОБУЧЕНИЕ АВТОЭНКОДЕРА СВЯЗЕЙ
# ==========================================
pair_model = PairAutoencoder().to(device)
pair_optimizer = torch.optim.Adam(pair_model.parameters(), lr=0.001)
pair_criterion = nn.MSELoss()

print(f"\n--- Обучение PairAutoencoder на {N*N} парах ---")
NUM_EPOCHS = 20
for epoch in range(NUM_EPOCHS):
    pair_model.train()
    total_loss = 0.0
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)
        pair_optimizer.zero_grad()
        reconstructed = pair_model(batch_x)
        loss = pair_criterion(reconstructed, batch_x)
        loss.backward()
        pair_optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    avg_loss = total_loss / len(dataset)
    if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
        print(f"Pair AE Epoch {epoch:2d}, Loss: {avg_loss:.8f}")

pair_model.eval()
torch.save(pair_model.state_dict(), "pair_autoencoder.pth")
print("Модель PairAutoencoder сохранена в pair_autoencoder.pth")