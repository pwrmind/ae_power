import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. АРХИТЕКТУРЫ
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
    def forward(self, x): return self.decoder(self.encoder(x))

class MorphPredictor(nn.Module):
    def __init__(self, emb_dim=8, num_classes=33):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, prev_emb, curr_emb, dist):
        x = torch.cat([prev_emb, curr_emb, dist], dim=-1)
        return self.net(x)

# ==========================================
# 2. ПОДГОТОВКА И КЭШ
# ==========================================

def get_char_vector(char):
    code = ord(char)
    bits = [int(b) * 2 - 1 for b in format(code, '032b')]
    freq = code / 100.0
    extra = [np.sin(freq), np.cos(freq), np.sin(freq/10), np.cos(freq/10)]
    return torch.FloatTensor(bits + extra)

text = "проклятый старый дом "
alphabet = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
idx_to_char = {i: ch for i, ch in enumerate(alphabet)}
all_chars_vecs = torch.stack([get_char_vector(c) for c in alphabet])

# ==========================================
# 3. ЭТАП 1: ОБУЧЕНИЕ AE С ВЫВОДОМ
# ==========================================

ae_model = UnicodeAutoencoder()
ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)
ae_criterion = nn.MSELoss()

print("--- Этап 1: Обучение Автоэнкодера ---")
for epoch in range(5001):
    ae_optimizer.zero_grad()
    outputs = ae_model(all_chars_vecs)
    loss = ae_criterion(outputs, all_chars_vecs[:, :32])
    loss.backward()
    ae_optimizer.step()
    if epoch % 200 == 0:
        print(f"AE Epoch {epoch}, Loss: {loss.item():.8f}")

ae_model.eval()

# СОЗДАНИЕ КЭША ЭМБЕДДИНГОВ
char_emb_cache = {}
with torch.no_grad():
    for char in alphabet:
        vec = get_char_vector(char).unsqueeze(0)
        char_emb_cache[char] = ae_model.encoder(vec)

# ==========================================
# 4. ЭТАП 2: ОБУЧЕНИЕ ПРЕДИКТОРA (С КЭШЕМ)
# ==========================================

predictor = MorphPredictor(num_classes=len(alphabet))
p_optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
p_criterion = nn.CrossEntropyLoss()

print("\n--- Этап 2: Обучение Предиктора (Используется кэш эмбеддингов) ---")

for epoch in range(2001):
    total_loss = 0
    for i in range(5, len(text)):
        target_char = text[i]
        curr_char = text[i-1]
        dist = np.random.randint(2, 5) 
        prev_char = text[i - dist]
        
        # Берём из кэша вместо вызова модели
        emb_curr = char_emb_cache[curr_char]
        emb_prev = char_emb_cache[prev_char]
        dist_tensor = torch.FloatTensor([[dist / 10.0]])

        p_optimizer.zero_grad()
        logits = predictor(emb_prev, emb_curr, dist_tensor)
        loss = p_criterion(logits, torch.LongTensor([char_to_idx[target_char]]))
        loss.backward()
        p_optimizer.step()
        total_loss += loss.item()
        
    if epoch % 500 == 0:
        print(f"Pred Epoch {epoch}, Avg Loss: {total_loss/(len(text)-5):.4f}")

# ==========================================
# 5. ТЕСТИРОВАНИЕ
# ==========================================

def test_logic(char_prev, char_curr, dist_val):
    predictor.eval()
    with torch.no_grad():
        # Тест тоже может брать из кэша (если буква есть в алфавите)
        e_p = char_emb_cache.get(char_prev, ae_model.encoder(get_char_vector(char_prev).unsqueeze(0)))
        e_c = char_emb_cache.get(char_curr, ae_model.encoder(get_char_vector(char_curr).unsqueeze(0)))
        d = torch.FloatTensor([[dist_val / 10.0]])
        logits = predictor(e_p, e_c, d)
        return idx_to_char[torch.argmax(logits).item()]

print("\n--- Проверка логики ---")
print(f"Цель 'й': предш='я', тек='ы', dist=3 -> '{test_logic('я', 'ы', 3)}'")
print(f"Цель 'ы': предш='л', тек='т', dist=3 -> '{test_logic('л', 'т', 3)}'")
