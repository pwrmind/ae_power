import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. АРХИТЕКТУРА АВТОЭНКОДЕРА
# ==========================================

class UnicodeAutoencoder(nn.Module):
    def __init__(self, bits=32, emb_dim=8):
        super().__init__()
        # 32 бита + 4 гармоники sin/cos = 36 входов
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
# 2. ПОДГОТОВКА ДАННЫХ (ВЫБОРКА ИЗ ВСЕГО UNICODE)
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
    # 1. Основные алфавиты (0 - 3000)
    for i in range(0, 3000):
        codes.add(i)
    # 2. Иероглифы CJK (0x4E00 - 0x9FFF)
    for i in range(0, 1000):
        codes.add(0x4E00 + i)
    # 3. Эмодзи (0x1F600 - 0x1F64F)
    for i in range(0x1F600, 0x1F64F):
        codes.add(i)
    return codes

# Получаем символы и их векторы
unicode_codes = generate_full_unicode_sample()
alphabet = [chr(c) for c in sorted(unicode_codes)]
all_chars_vecs = torch.stack([get_char_vector(c) for c in alphabet])

# ==========================================
# 3. ОБУЧЕНИЕ АВТОЭНКОДЕРА
# ==========================================

ae_model = UnicodeAutoencoder()
ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.002)
ae_criterion = nn.MSELoss()

print("--- Обучение UnicodeAutoencoder на выборке Unicode ---")
print(f"Количество символов в обучающей выборке: {len(alphabet)}")

for epoch in range(5001):
    ae_optimizer.zero_grad()
    outputs = ae_model(all_chars_vecs)
    loss = ae_criterion(outputs, all_chars_vecs[:, :32])  # восстанавливаем только 32 бита
    loss.backward()
    ae_optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, AE Loss: {loss.item():.8f}")

ae_model.eval()

# ==========================================
# 4. СОХРАНЕНИЕ МОДЕЛИ
# ==========================================

torch.save(ae_model.state_dict(), "unicode_autoencoder.pth")
print("Модель сохранена в unicode_autoencoder.pth")