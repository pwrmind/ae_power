import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# ==========================================
# 1. ЮНИКОД-АВТОЭНКОДЕР (как раньше)
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

def get_char_vector(char):
    code = ord(char)
    bits = [int(b) * 2 - 1 for b in format(code, '032b')]
    freq = code / 100.0
    extra = [np.sin(freq), np.cos(freq), np.sin(freq/10), np.cos(freq/10)]
    return torch.FloatTensor(bits + extra)

def generate_full_unicode_sample():
    codes = set()
    for i in range(0, 3000):
        codes.add(i)
    for i in range(0, 1000):
        codes.add(0x4E00 + i)
    for i in range(0x1F600, 0x1F64F):
        codes.add(i)
    return codes

# ==========================================
# 2. ЗАГРУЗКА / ОБУЧЕНИЕ ЮНИКОД-АВТОЭНКОДЕРА
# ==========================================
ae_model = UnicodeAutoencoder().to(device)
try:
    ae_model.load_state_dict(torch.load("unicode_autoencoder.pth", map_location=device))
    ae_model.eval()
    print("Загружена предобученная модель UnicodeAutoencoder")
except FileNotFoundError:
    unicode_codes = generate_full_unicode_sample()
    alphabet = [chr(c) for c in sorted(unicode_codes)]
    all_chars_vecs = torch.stack([get_char_vector(c) for c in alphabet]).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.002)
    ae_criterion = nn.MSELoss()
    print("--- Обучение UnicodeAutoencoder ---")
    for epoch in range(8001):
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
# 3. ПОЛУЧЕНИЕ И НОРМАЛИЗАЦИЯ ЭМБЕДДИНГОВ
# ==========================================
with torch.no_grad():
    # Генерируем все символы для кэша (алфавит тот же, что при обучении AE)
    unicode_codes = generate_full_unicode_sample()
    alphabet = [chr(c) for c in sorted(unicode_codes)]
    all_chars_vecs = torch.stack([get_char_vector(c) for c in alphabet]).to(device)
    emb_raw = ae_model.encoder(all_chars_vecs)  # [N, 8]

# Переносим на CPU для sklearn
emb_np = emb_raw.cpu().numpy()
scaler = StandardScaler()
emb_norm_np = scaler.fit_transform(emb_np)
# Обратно на GPU
emb_all = torch.tensor(emb_norm_np, dtype=torch.float32, device=device)
N = emb_all.shape[0]
print(f"Кэш эмбеддингов: {N} символов, среднее 0, std 1")

# ==========================================
# 4. АРХИТЕКТУРА PAIR AUTOENCODER (УСИЛЕННАЯ)
# ==========================================
class PairAutoencoder(nn.Module):
    def __init__(self, input_dim=16, bottleneck_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

pair_model = PairAutoencoder().to(device)
pair_optimizer = torch.optim.Adam(pair_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pair_optimizer, mode='min', factor=0.5, patience=3)
pair_criterion = nn.MSELoss()

# ==========================================
# 5. ОБУЧЕНИЕ С ДИНАМИЧЕСКОЙ ГЕНЕРАЦИЕЙ ПАР
# ==========================================
BATCH_SIZE = 65536
EPOCHS = 50
PAIRS_PER_EPOCH = 2_000_000  # 2M случайных пар за эпоху

print(f"\n--- Обучение PairAutoencoder (нормализованные эмбеддинги) ---")
print(f"Пар за эпоху: {PAIRS_PER_EPOCH}, batch_size: {BATCH_SIZE}")

for epoch in range(EPOCHS):
    pair_model.train()
    total_loss = 0.0
    total_samples = 0
    
    for start in range(0, PAIRS_PER_EPOCH, BATCH_SIZE):
        current_bs = min(BATCH_SIZE, PAIRS_PER_EPOCH - total_samples)
        idx_i = torch.randint(0, N, (current_bs,), device=device)
        idx_j = torch.randint(0, N, (current_bs,), device=device)
        
        emb_i = emb_all[idx_i]
        emb_j = emb_all[idx_j]
        pair_vecs = torch.cat([emb_i, emb_j], dim=-1)
        
        pair_optimizer.zero_grad()
        reconstructed = pair_model(pair_vecs)
        loss = pair_criterion(reconstructed, pair_vecs)
        loss.backward()
        pair_optimizer.step()
        
        total_loss += loss.item() * current_bs
        total_samples += current_bs
    
    avg_loss = total_loss / total_samples
    scheduler.step(avg_loss)
    
    if epoch % 3 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:2d}, Loss: {avg_loss:.6f}")

pair_model.eval()
torch.save(pair_model.state_dict(), "pair_autoencoder.pth")
print("Модель PairAutoencoder сохранена в pair_autoencoder.pth")