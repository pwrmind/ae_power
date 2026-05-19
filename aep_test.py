import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

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
    unicode_codes = generate_full_unicode_sample()
    alphabet = [chr(c) for c in sorted(unicode_codes)]
    all_chars_vecs = torch.stack([get_char_vector(c) for c in alphabet]).to(device)
    emb_raw = ae_model.encoder(all_chars_vecs)  # [N, 8]

emb_np = emb_raw.cpu().numpy()
scaler = StandardScaler()
emb_norm_np = scaler.fit_transform(emb_np)
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==========================================
# 5. ОБУЧЕНИЕ PAIR AUTOENCODER
# ==========================================
pair_model = PairAutoencoder().to(device)
pair_optimizer = torch.optim.Adam(pair_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pair_optimizer, mode='min', factor=0.5, patience=3)
pair_criterion = nn.MSELoss()

BATCH_SIZE = 65536
EPOCHS = 50   # как в вашем последнем запуске
PAIRS_PER_EPOCH = 2_000_000

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

# ==========================================
# 6. АНАЛИЗ ЛАТЕНТНОГО ПРОСТРАНСТВА И ЭМЕРДЖЕНТНОСТИ
# ==========================================
def get_char_type(char):
    code = ord(char)
    if 0x1F600 <= code <= 0x1F64F:
        return "emoji"
    elif 0x4E00 <= code <= 0x9FFF:
        return "CJK"
    elif 0 <= code < 3000:
        # уточним внутри базового диапазона
        if 48 <= code <= 57: return "digit"
        elif 65 <= code <= 90: return "latin_upper"
        elif 97 <= code <= 122: return "latin_lower"
        elif 1040 <= code <= 1071: return "cyrillic_upper"
        elif 1072 <= code <= 1103: return "cyrillic_lower"
        else: return "basic_other"
    else:
        return "other"

# Списки символов и их типов
char_types = [get_char_type(c) for c in alphabet]
char_codes = [ord(c) for c in alphabet]

def sample_pairs(num_pairs):
    """Генерирует случайные пары и возвращает тензоры, индексы и метаданные."""
    idx_i = torch.randint(0, N, (num_pairs,), device=device)
    idx_j = torch.randint(0, N, (num_pairs,), device=device)
    emb_i = emb_all[idx_i]
    emb_j = emb_all[idx_j]
    pair_vecs = torch.cat([emb_i, emb_j], dim=-1)  # [num_pairs, 16]
    with torch.no_grad():
        latent = pair_model.encoder(pair_vecs)  # [num_pairs, 8]
    return latent.cpu().numpy(), idx_i.cpu().numpy(), idx_j.cpu().numpy()

# --- Кластеризация (t-SNE) ---
print("\n--- Кластеризация латентного пространства (t-SNE) ---")
SAMPLE_SIZE = 4000
latent_np, idx_i, idx_j = sample_pairs(SAMPLE_SIZE)

# Создаем метки для пар на основе типов символов
pair_type_labels = []
for k in range(SAMPLE_SIZE):
    t1 = char_types[idx_i[k]]
    t2 = char_types[idx_j[k]]
    pair_type_labels.append(f"{t1} + {t2}")

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
latent_2d = tsne.fit_transform(latent_np)

plt.figure(figsize=(12, 8))
# Раскрасим по основному типу первого символа для наглядности
unique_types = list(set([t.split(' + ')[0] for t in pair_type_labels]))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
type_to_color = {t: colors[i] for i, t in enumerate(unique_types)}

for i, label in enumerate(pair_type_labels):
    main_type = label.split(' + ')[0]
    plt.scatter(latent_2d[i, 0], latent_2d[i, 1], color=type_to_color[main_type], alpha=0.6, s=10)

# Легенда
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=type_to_color[t], markersize=8, label=t) for t in unique_types]
plt.legend(handles=handles, title="Тип первого символа")
plt.title("t-SNE латентных представлений пар (PairAutoencoder bottleneck=8)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# --- Эмерджентный анализ нейронов горлышка ---
print("\n--- Эмерджентный анализ: корреляция нейронов bottleneck со свойствами пар ---")
# Сгенерируем большую выборку для корреляций
ANALYSIS_SAMPLES = 10000
latent_np, idx_i, idx_j = sample_pairs(ANALYSIS_SAMPLES)

# Свойства пар:
properties = {
    "char1_code": np.array([char_codes[i] for i in idx_i]),
    "char2_code": np.array([char_codes[j] for j in idx_j]),
    "char1_is_digit": np.array([1 if char_types[i] == "digit" else 0 for i in idx_i]),
    "char2_is_digit": np.array([1 if char_types[j] == "digit" else 0 for j in idx_j]),
    "char1_is_emoji": np.array([1 if char_types[i] == "emoji" else 0 for i in idx_i]),
    "char2_is_emoji": np.array([1 if char_types[j] == "emoji" else 0 for j in idx_j]),
    "char1_is_CJK": np.array([1 if char_types[i] == "CJK" else 0 for i in idx_i]),
    "char2_is_CJK": np.array([1 if char_types[j] == "CJK" else 0 for j in idx_j]),
    "both_same_type": np.array([1 if char_types[i] == char_types[j] else 0 for i, j in zip(idx_i, idx_j)]),
    "code_distance": np.array([abs(char_codes[i] - char_codes[j]) for i, j in zip(idx_i, idx_j)]),
    # можно добавить регистр для латиницы/кириллицы и т.д.
}

# Собираем DataFrame
df_latent = pd.DataFrame(latent_np, columns=[f"neuron_{i}" for i in range(8)])
for prop_name, prop_vals in properties.items():
    df_latent[prop_name] = prop_vals

# Корреляция
corr_matrix = df_latent.corr()
latent_cols = [f"neuron_{i}" for i in range(8)]
prop_cols = list(properties.keys())
subset_corr = corr_matrix.loc[latent_cols, prop_cols]

plt.figure(figsize=(12, 6))
sns.heatmap(subset_corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Корреляция нейронов PairAutoencoder bottleneck со свойствами пар")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Вывод самого значимого нейрона для каждого свойства
for prop in prop_cols:
    best_neuron = subset_corr[prop].abs().idxmax()
    corr_value = subset_corr.loc[best_neuron, prop]
    print(f"Свойство '{prop}' лучше всего закодировано в {best_neuron} (r = {corr_value:.2f})")