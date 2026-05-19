import torch
import torch.nn as nn
import numpy as np
import os
import random

# ==================== НАСТРОЙКИ ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DIST = 5                     # максимальная дистанция для триплетов
EMB_DIM = 8                      # размер бутылочного горлышка автоэнкодера
BITS = 32
INPUT_DIM = BITS + 4             # 36
AE_MODEL_PATH = "ae_model.pth"
PRED_MODEL_PATH = "pred_model.pth"
EMB_CACHE_PATH = "char_embeddings.pt"
ALPHABET_PATH = "alphabet.pt"
TEXT_FILE = "input.txt"
LR_AE = 0.001
LR_PRED = 0.001
EPOCHS_AE = 5000
EPOCHS_PRED = 500                # предиктору хватит благодаря большому числу примеров
BATCH_SIZE = 256                 # для обучения AE можно использовать весь алфавит сразу

# ==================== МОДЕЛИ ====================
class UnicodeAutoencoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, emb_dim=EMB_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(),
            nn.Linear(64, 16), nn.GELU(),
            nn.Linear(16, emb_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 16), nn.GELU(),
            nn.Linear(16, 64), nn.GELU(),
            nn.Linear(64, BITS),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


class MorphPredictor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, prev_emb, curr_emb, dist):
        x = torch.cat([prev_emb, curr_emb, dist], dim=-1)
        return self.net(x)


# ==================== ПРЕОБРАЗОВАНИЕ СИМВОЛА ====================
def get_char_vector(char):
    """36-мерный вектор: 32 бита + 4 гармоники."""
    code = ord(char)
    bits = [int(b) * 2 - 1 for b in format(code, '032b')]   # {-1, 1}
    freq = code / 100.0
    extra = [np.sin(freq), np.cos(freq), np.sin(freq / 10), np.cos(freq / 10)]
    return torch.FloatTensor(bits + extra).to(DEVICE)


# ==================== ОБУЧЕНИЕ АВТОЭНКОДЕРА ====================
def train_autoencoder(alphabet):
    """Обучает автоэнкодер на всех символах алфавита (если модель ещё не сохранена)."""
    if os.path.exists(AE_MODEL_PATH):
        print(f"Автоэнкодер уже обучен ({AE_MODEL_PATH}). Пропуск.")
        return

    model = UnicodeAutoencoder(input_dim=INPUT_DIM, emb_dim=EMB_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_AE)
    criterion = nn.MSELoss()

    # Все символы как один батч
    all_vecs = torch.stack([get_char_vector(c) for c in alphabet])
    target = all_vecs[:, :BITS]      # восстанавливаем только битовую часть

    print(f"Обучение автоэнкодера на {len(alphabet)} символах...")
    for epoch in range(1, EPOCHS_AE + 1):
        optimizer.zero_grad()
        output = model(all_vecs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0 or epoch == 1:
            print(f"AE Epoch {epoch}/{EPOCHS_AE}, Loss: {loss.item():.8f}")

    torch.save(model.state_dict(), AE_MODEL_PATH)
    print(f"Автоэнкодер сохранён в {AE_MODEL_PATH}")


# ==================== ПОДГОТОВКА ЭМБЕДДИНГОВ ====================
def build_emb_cache(model, alphabet):
    """Строит кэш эмбеддингов для всех символов."""
    model.eval()
    cache = {}
    with torch.no_grad():
        for c in alphabet:
            vec = get_char_vector(c).unsqueeze(0)
            cache[c] = model.encode(vec).cpu()
    return cache


# ==================== ОБУЧЕНИЕ ПРЕДИКТОРА ====================
def train_predictor(alphabet, char_to_idx, emb_cache):
    """Обучает предиктор на всех триплетах с дистанциями 1..MAX_DIST (если модель ещё не сохранена)."""
    if os.path.exists(PRED_MODEL_PATH):
        print(f"Предиктор уже обучен ({PRED_MODEL_PATH}). Пропуск.")
        return

    # Читаем текст
    if not os.path.exists(TEXT_FILE):
        raise FileNotFoundError(f"Обучающий файл '{TEXT_FILE}' не найден.")
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Подготавливаем данные: для каждой позиции и каждой дистанции создаём пример
    inputs_prev = []
    inputs_curr = []
    inputs_dist = []
    targets = []

    predictor = MorphPredictor(num_classes=len(alphabet)).to(DEVICE)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=LR_PRED)
    criterion = nn.CrossEntropyLoss()

    print("Формирование обучающих примеров с дистанциями 1..MAX_DIST...")
    for i in range(1, len(text) - 1):
        target_char = text[i]
        curr_char = text[i - 1]
        for d in range(1, MAX_DIST + 1):
            prev_idx = i - d
            if prev_idx < 0:
                continue
            prev_char = text[prev_idx]
            # Все символы должны быть в алфавите (мы его заранее собрали)
            if prev_char not in emb_cache or curr_char not in emb_cache or target_char not in char_to_idx:
                continue
            inputs_prev.append(emb_cache[prev_char])
            inputs_curr.append(emb_cache[curr_char])
            inputs_dist.append(torch.tensor([d / MAX_DIST]))
            targets.append(char_to_idx[target_char])

    if len(inputs_prev) == 0:
        raise RuntimeError("Нет подходящих примеров. Проверьте текст и алфавит.")

    # Переводим в тензоры
    prev_tensor = torch.stack(inputs_prev).to(DEVICE)
    curr_tensor = torch.stack(inputs_curr).to(DEVICE)
    dist_tensor = torch.stack(inputs_dist).to(DEVICE)
    target_tensor = torch.LongTensor(targets).to(DEVICE)

    dataset_size = prev_tensor.size(0)
    print(f"Найдено примеров: {dataset_size}")

    # Перемешиваем и обучаем батчами
    perm = torch.randperm(dataset_size)
    for epoch in range(1, EPOCHS_PRED + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()
        for start in range(0, dataset_size, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            logits = predictor(prev_tensor[idx], curr_tensor[idx], dist_tensor[idx])
            loss = criterion(logits, target_tensor[idx])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item() * len(idx)
        avg_loss = epoch_loss / dataset_size
        if epoch % 100 == 0 or epoch == 1:
            print(f"Pred Epoch {epoch}/{EPOCHS_PRED}, Loss: {avg_loss:.6f}")

    torch.save(predictor.state_dict(), PRED_MODEL_PATH)
    print(f"Предиктор сохранён в {PRED_MODEL_PATH}")


# ==================== ГЕНЕРАЦИЯ С ГОЛОСОВАНИЕМ ====================
@torch.no_grad()
def generate(predictor, ae_model, seed_text, length=40):
    predictor.eval()
    ae_model.eval()

    # Загружаем кэш и алфавит
    if not (os.path.exists(EMB_CACHE_PATH) and os.path.exists(ALPHABET_PATH)):
        print("Сначала обучите модели.")
        return
    emb_cache = torch.load(EMB_CACHE_PATH)
    alphabet = torch.load(ALPHABET_PATH)
    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    idx_to_char = {i: ch for i, ch in enumerate(alphabet)}

    current_text = seed_text.upper()
    print(f"Seed: {current_text} | Result: ", end="", flush=True)

    for _ in range(length):
        last_char = current_text[-1]
        if last_char not in emb_cache:
            # Пытаемся добавить новый символ на лету
            if last_char not in alphabet:
                print(f"\nСимвол '{last_char}' отсутствует в алфавите.")
                break
            vec = get_char_vector(last_char).unsqueeze(0)
            emb = ae_model.encode(vec).cpu()
            emb_cache[last_char] = emb
        curr_emb = emb_cache[last_char].unsqueeze(0).to(DEVICE)

        votes = {}
        for d in range(1, MAX_DIST + 1):
            if len(current_text) <= d:
                continue
            past_char = current_text[-d - 1]
            if past_char not in emb_cache:
                continue
            past_emb = emb_cache[past_char].unsqueeze(0).to(DEVICE)
            dist_tensor = torch.tensor([[d / MAX_DIST]], device=DEVICE)
            logits = predictor(past_emb, curr_emb, dist_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
            best_idx = int(np.argmax(probs))
            best_char = idx_to_char[best_idx]
            weight = 1.0 / (d ** 1.2)          # эвристический вес
            votes[best_char] = votes.get(best_char, 0) + weight

        if not votes:
            break
        next_char = max(votes, key=votes.get)
        current_text += next_char
        print(next_char, end="", flush=True)
    print()
    return current_text


# ==================== ГЛАВНЫЙ ЗАПУСК ====================
def main():
    # 1. Чтение текста и сбор алфавита
    if not os.path.exists(TEXT_FILE):
        raise FileNotFoundError(f"Создайте файл '{TEXT_FILE}' с обучающим текстом (например, 'ПРЕЗИДЕНТ ПРЕКРАСНО ПРЕДСКАЗАЛ ПРЕПЯТСТВИЕ').")
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    alphabet = sorted(list(set(text)))
    print(f"Алфавит из {len(alphabet)} символов: {''.join(alphabet)}")

    # Сохраняем алфавит (если ещё нет)
    if not os.path.exists(ALPHABET_PATH):
        torch.save(alphabet, ALPHABET_PATH)

    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}

    # 2. Автоэнкодер (обучаем, если нужно)
    train_autoencoder(alphabet)

    # 3. Загружаем автоэнкодер и строим кэш эмбеддингов
    ae = UnicodeAutoencoder(input_dim=INPUT_DIM, emb_dim=EMB_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_MODEL_PATH, map_location=DEVICE))
    ae.eval()

    if not os.path.exists(EMB_CACHE_PATH):
        print("Построение кэша эмбеддингов...")
        emb_cache = build_emb_cache(ae, alphabet)
        torch.save(emb_cache, EMB_CACHE_PATH)
        print(f"Эмбеддинги сохранены в {EMB_CACHE_PATH}")
    else:
        emb_cache = torch.load(EMB_CACHE_PATH)

    # 4. Предиктор (обучаем, если нужно)
    train_predictor(alphabet, char_to_idx, emb_cache)

    # 5. Загружаем предиктор
    predictor = MorphPredictor(num_classes=len(alphabet)).to(DEVICE)
    predictor.load_state_dict(torch.load(PRED_MODEL_PATH, map_location=DEVICE))
    predictor.eval()

    # 6. Генерация
    print("\n--- Генерация ---")
    generate(predictor, ae, "ПРЕП", length=40)


if __name__ == "__main__":
    main()