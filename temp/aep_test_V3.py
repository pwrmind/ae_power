import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import pandas as pd
import seaborn as sns


# ==========================================
# 1. АРХИТЕКТУРЫ СЕТЕЙ
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

class MorphPredictor(nn.Module):
    def __init__(self, emb_dim=8, num_classes=33):
        super().__init__()
        # Вход: emb(n-dist) + emb(n-1) + dist_value = 17 параметров
        self.net = nn.Sequential(
            nn.Linear(17, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, prev_emb, curr_emb, dist):
        x = torch.cat([prev_emb, curr_emb, dist], dim=-1)
        return self.net(x)

# ==========================================
# 2. ПОДГОТОВКА ДАННЫХ И КЭША
# ==========================================

def get_char_vector(char):
    code = ord(char)
    bits = [int(b) * 2 - 1 for b in format(code, '032b')]
    freq = code / 100.0
    extra = [np.sin(freq), np.cos(freq), np.sin(freq/10), np.cos(freq/10)]
    return torch.FloatTensor(bits + extra)

# ВАШ ТЕКСТ ДЛЯ ОБУЧЕНИЯ
text = "Если вы подадите большой текст, вы увидите, как веса перераспределятся. На сложных текстах dist обычно становится выше, потому что без точного понимания расстояния до приставки или корня сеть начинает ошибаться в окончаниях."
alphabet = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
idx_to_char = {i: ch for i, ch in enumerate(alphabet)}
all_chars_vecs = torch.stack([get_char_vector(c) for c in alphabet])

# ==========================================
# 3. ЭТАП 1: ОБУЧЕНИЕ АВТОЭНКОДЕРА (ЭМБЕДДИНГИ)
# ==========================================

ae_model = UnicodeAutoencoder()
ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.002)
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

# Создаем кэш, чтобы не вызывать энкодер в цикле
char_emb_cache = {}
with torch.no_grad():
    for char in alphabet:
        vec = get_char_vector(char).unsqueeze(0)
        char_emb_cache[char] = ae_model.encoder(vec)

# ==========================================
# 4. ЭТАП 2: БЫСТРОЕ ОБУЧЕНИЕ ПРЕДИКТОРA (БАТЧИ)
# ==========================================

def prepare_batched_data(text, cache, to_idx, samples_per_char=3):
    p_prev, p_curr, p_dist, p_targets = [], [], [], []
    for i in range(5, len(text)):
        for _ in range(samples_per_char):
            d = np.random.randint(2, 6) # Дистанция (смещение) от 2 до 5
            p_prev.append(cache[text[i-d]])
            p_curr.append(cache[text[i-1]])
            p_dist.append(torch.FloatTensor([[d / 10.0]]))
            p_targets.append(to_idx[text[i]])
    
    return (torch.cat(p_prev), torch.cat(p_curr), 
            torch.cat(p_dist), torch.LongTensor(p_targets))

# Подготовка данных один раз для ускорения
b_prev, b_curr, b_dist, b_targets = prepare_batched_data(text, char_emb_cache, char_to_idx)

predictor = MorphPredictor(num_classes=len(alphabet))
p_optimizer = torch.optim.Adam(predictor.parameters(), lr=0.002)
p_criterion = nn.CrossEntropyLoss()

print(f"\n--- Этап 2: Обучение Предиктора (Батч из {len(b_targets)} примеров) ---")
for epoch in range(3001):
    predictor.train()
    p_optimizer.zero_grad()
    
    # Векторизованный проход без циклов Python
    logits = predictor(b_prev, b_curr, b_dist)
    loss = p_criterion(logits, b_targets)
    
    loss.backward()
    p_optimizer.step()
    
    if epoch % 500 == 0:
        preds = torch.argmax(logits, dim=1)
        acc = (preds == b_targets).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {acc.item():.2f}")

# ==========================================
# 5. ВИЗУАЛИЗАЦИЯ
# ==========================================

def visualize_results(predictor, cache, text):
    predictor.eval()
    
    # 1. Важность входных параметров
    first_layer_weights = predictor.net[0].weight.data.abs().mean(dim=0).cpu().numpy()
    plt.figure(figsize=(10, 4))
    labels = [f'prev_{i}' for i in range(8)] + [f'curr_{i}' for i in range(8)] + ['dist']
    plt.bar(labels, first_layer_weights, color='skyblue')
    plt.title("Важность входных параметров")
    plt.show()

    # 2. t-SNE активаций скрытого слоя
    activations = []
    point_labels = []
    with torch.no_grad():
        for i in range(5, len(text)):
            d_val = 3
            input_vec = torch.cat([cache[text[i-d_val]].flatten(), 
                                 cache[text[i-1]].flatten(), 
                                 torch.tensor([d_val/10.0])])
            # Получаем скрытое состояние (после первого Linear + GELU)
            hidden = torch.nn.functional.gelu(predictor.net[0](input_vec))
            activations.append(hidden.numpy())
            point_labels.append(f"{text[i-1]}→{text[i]}")

    tsne = TSNE(n_components=2, perplexity=max(5, len(activations)//5), random_state=42)
    vis = tsne.fit_transform(np.array(activations))

    plt.figure(figsize=(12, 10))
    plt.scatter(vis[:, 0], vis[:, 1], alpha=0.5, c='red')
    for i, txt in enumerate(point_labels):
        plt.annotate(txt, (vis[i, 0], vis[i, 1]), fontsize=8)
    plt.title("Кластеризация переходов (Скрытый слой)")
    plt.show()

print("\n--- Запуск визуализации ---")
visualize_results(predictor, char_emb_cache, text)

def inspect_emergent_properties(ae_model, char_emb_cache):
    ae_model.eval()
    
    # 1. Определяем "скрытые" свойства, которые мы НЕ подавали в явном виде
    vowels = set("аеёиоуыэюяaeiouy")
    digits = set("0123456789")
    
    data_list = []
    for char, emb in char_emb_cache.items():
        emb_array = emb.flatten().numpy()
        char_code = ord(char)
        
        # Собираем метаданные для проверки эмерджентности
        properties = {
            "char": char,
            "is_vowel": 1 if char.lower() in vowels else 0,
            "is_digit": 1 if char in digits else 0,
            "is_uppercase": 1 if char.isupper() else 0,
            "is_cyrillic": 1 if 1040 <= char_code <= 1103 else 0,
            "is_punctuation": 1 if char in ".,!?- " else 0,
            "code_val": char_code
        }
        
        # Добавляем значения 8 нейронов из горлышка
        for i in range(len(emb_array)):
            properties[f"neuron_{i}"] = emb_array[i]
            
        data_list.append(properties)
    
    df = pd.DataFrame(data_list)
    
    # 2. Считаем корреляцию нейронов с нашими скрытыми свойствами
    # Нас интересует: "догадался" ли какой-то нейрон о гласных или цифрах?
    latent_cols = [f"neuron_{i}" for i in range(8)]
    property_cols = ["is_vowel", "is_digit", "is_uppercase", "is_cyrillic", "is_punctuation"]
    
    corr_matrix = df[latent_cols + property_cols].corr()
    subset_corr = corr_matrix.loc[latent_cols, property_cols]
    
    # 3. Визуализация
    plt.figure(figsize=(10, 6))
    sns.heatmap(subset_corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Эмерджентный анализ: Корреляция нейронов горлышка со скрытыми свойствами")
    plt.show()
    
    # Вывод самого "умного" нейрона
    for prop in property_cols:
        best_neuron = subset_corr[prop].abs().idxmax()
        correlation = subset_corr.loc[best_neuron, prop]
        print(f"Свойство '{prop}' лучше всего закодировано в {best_neuron} (r = {correlation:.2f})")

# Запуск после обучения AE
inspect_emergent_properties(ae_model, char_emb_cache)
