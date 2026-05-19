import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. АРХИТЕКТУРЫ СЕТЕЙ
# ==========================================

class UnicodeAutoencoder(nn.Module):
    def __init__(self, bits=32, emb_dim=8):
        super().__init__()
        # 32 бита + 4 гармоники = 36 входов
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

class MorphPredictor(nn.Module):
    def __init__(self, emb_dim=8, num_classes=33):
        super().__init__()
        # Вход: 8 (пред.) + 8 (тек.) + 1 (дист.) = 17
        self.net = nn.Sequential(
            nn.Linear(17, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, num_classes) # Выход — вероятность каждой буквы
        )

    def forward(self, prev_emb, curr_emb, dist):
        x = torch.cat([prev_emb, curr_emb, dist], dim=-1)
        return self.net(x)

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def get_char_vector(char):
    code = ord(char)
    bits = [int(b) * 2 - 1 for b in format(code, '032b')]
    freq = code / 100.0
    extra = [np.sin(freq), np.cos(freq), np.sin(freq/10), np.cos(freq/10)]
    return torch.FloatTensor(bits + extra)

# Данные для обучения
text = "проклятый старый дом "
alphabet = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
idx_to_char = {i: ch for i, ch in enumerate(alphabet)}

# Векторы всех уникальных символов
all_chars_vecs = torch.stack([get_char_vector(c) for c in alphabet])

# ==========================================
# 3. ЭТАП 1: ОБУЧЕНИЕ АВТОЭНКОДЕРА
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
    if epoch % 1000 == 0:
        print(f"AE Loss: {loss.item():.8f}")

ae_model.eval()

# ==========================================
# 4. ЭТАП 2: ПОДГОТОВКА ДАННЫХ И ОБУЧЕНИЕ ПРЕДИКТОРА
# ==========================================

def prepare_predictor_data():
    inputs, targets = [], []
    for i in range(2, len(text)):
        with torch.no_grad():
            # Получаем эмбеддинги из энкодера
            e_p = ae_model.encoder(get_char_vector(text[i-2]).unsqueeze(0))
            e_c = ae_model.encoder(get_char_vector(text[i-1]).unsqueeze(0))
        
        # Дистанция: позиция в текущем слове (от последнего пробела)
        last_space = text[:i].rfind(' ')
        dist_val = (i - last_space) / 10.0
        
        inputs.append(torch.cat([e_p.flatten(), e_c.flatten(), torch.tensor([dist_val])]))
        targets.append(char_to_idx[text[i]])
    return torch.stack(inputs), torch.LongTensor(targets)

p_inputs, p_targets = prepare_predictor_data()

predictor = MorphPredictor(num_classes=len(alphabet))
p_optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
p_criterion = nn.CrossEntropyLoss()

print("\n--- Этап 2: Обучение Предиктора (Классификация) ---")
for epoch in range(2001):
    predictor.train()
    p_optimizer.zero_grad()
    
    # Подаем данные частями: prev_emb (0:8), curr_emb (8:16), dist (16:17)
    logits = predictor(p_inputs[:, :8], p_inputs[:, 8:16], p_inputs[:, 16:])
    loss = p_criterion(logits, p_targets)
    
    loss.backward()
    p_optimizer.step()
    
    if epoch % 500 == 0:
        # Считаем точность (accuracy)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == p_targets).float().mean()
        print(f"Pred Loss: {loss.item():.4f}, Acc: {acc.item():.2f}")

# ==========================================
# 5. ТЕСТИРОВАНИЕ
# ==========================================

def test(char1, char2, pos_in_word):
    predictor.eval()
    with torch.no_grad():
        e1 = ae_model.encoder(get_char_vector(char1).unsqueeze(0))
        e2 = ae_model.encoder(get_char_vector(char2).unsqueeze(0))
        d = torch.FloatTensor([[pos_in_word / 10.0]])
        
        logits = predictor(e1, e2, d)
        char_idx = torch.argmax(logits, dim=1).item()
        return idx_to_char[char_idx]

print("\n--- Результаты ---")
print(f"Контекст 'пр', поз 2 -> '{test('п', 'р', 2)}' (ожидаем 'о')")
print(f"Контекст 'ты', поз 7 -> '{test('т', 'ы', 7)}' (ожидаем 'й')")
print(f"Контекст 'до', поз 2 -> '{test('д', 'о', 2)}' (ожидаем 'м')")
