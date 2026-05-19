import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import random

# ========== Настройки ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AE_MODEL_PATH = "unicode_ae_8bit.pth"
PREDICTOR_MODEL_PATH = "predictor_weights.pth"
EMBEDDING_CACHE_PATH = "char_embeddings.pt"   # теперь хранит только 8‑мерные эмбеддинги для входов
TEXT_FILE = "input.txt"
MAX_DIST = 5
BITS = 32
EMBEDDING_DIM = 8
AE_EPOCHS = 5000
PRED_EPOCHS = 30
BATCH_SIZE = 512
PRED_BATCH_SIZE = 4096
AE_LR = 0.002
PRED_LR = 0.001

# ==================== Архитектуры ====================
class UnicodeAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(BITS, 16), nn.GELU(),
            nn.Linear(16, EMBEDDING_DIM), nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 16), nn.GELU(),
            nn.Linear(16, BITS), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_embedding(self, x):
        return self.encoder(x)


class TripletPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, BITS), nn.Sigmoid()   # ВЫХОД: 32 бита!
        )

    def forward(self, x):
        return self.net(x)


# ==================== Вспомогательные функции ====================
def char_to_bits(char):
    code = ord(char)
    bits = [int(b) for b in bin(code)[2:].zfill(BITS)]
    return torch.tensor(bits, dtype=torch.float32, device=DEVICE)


def build_char_embeddings(ae_model, text):
    ae_model.eval()
    embeddings = {}
    unique = set(text)
    with torch.no_grad():
        for c in unique:
            bits = char_to_bits(c).unsqueeze(0)
            embeddings[c] = ae_model.get_embedding(bits).squeeze(0).cpu()
    return embeddings


# ==================== Обучение автоэнкодера ====================
def train_autoencoder():
    if os.path.exists(AE_MODEL_PATH):
        print(f"Автоэнкодер уже обучен ({AE_MODEL_PATH}). Пропуск обучения.")
        return

    if not os.path.exists(TEXT_FILE):
        raise FileNotFoundError(f"Файл '{TEXT_FILE}' не найден.")
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    alphabet = list(set(text))
    print(f"Обучение автоэнкодера на {len(alphabet)} уникальных символах...")

    model = UnicodeAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()

    data = []
    for c in alphabet:
        data.append(char_to_bits(c))
    data = torch.stack(data)
    random.shuffle(data)

    for epoch in range(AE_EPOCHS):
        indices = torch.randperm(data.size(0))
        epoch_loss = 0.0
        for i in range(0, data.size(0), BATCH_SIZE):
            idx = indices[i:i + BATCH_SIZE]
            batch = data[idx]
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 500 == 0:
            avg = epoch_loss / (data.size(0) // BATCH_SIZE + 1)
            print(f"Epoch [{epoch}/{AE_EPOCHS}] Loss: {avg:.8f}")

    torch.save(model.state_dict(), AE_MODEL_PATH)
    print(f"Автоэнкодер сохранён в {AE_MODEL_PATH}")


# ==================== Датасет триплетов (цель – 32 бита) ====================
class NgramDataset(Dataset):
    def __init__(self, text, char_embeddings):
        self.inputs = []
        self.targets = []
        emb = {c: e.to(DEVICE) for c, e in char_embeddings.items()}

        for i in range(1, len(text) - 1):
            curr_char = text[i]
            next_char = text[i + 1]
            if curr_char not in emb or next_char not in emb:
                continue
            v_curr = emb[curr_char]
            v_target = char_to_bits(next_char)      # полные 32 бита следующего символа
            for d in range(1, MAX_DIST + 1):
                past_idx = i - d
                if past_idx >= 0:
                    past_char = text[past_idx]
                    if past_char in emb:
                        v_past = emb[past_char]
                        dist_val = torch.tensor([d / MAX_DIST], device=DEVICE)
                        inp = torch.cat((v_curr, v_past, dist_val))
                        self.inputs.append(inp)
                        self.targets.append(v_target)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ==================== Обучение предиктора ====================
def train_predictor(ae_model):
    if os.path.exists(PREDICTOR_MODEL_PATH):
        print(f"Предиктор уже обучен ({PREDICTOR_MODEL_PATH}). Пропуск обучения.")
        return

    if not os.path.exists(TEXT_FILE):
        raise FileNotFoundError(f"Файл '{TEXT_FILE}' не найден.")
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    if os.path.exists(EMBEDDING_CACHE_PATH):
        print("Загрузка кэша эмбеддингов...")
        char_embeddings = torch.load(EMBEDDING_CACHE_PATH, map_location="cpu")
    else:
        print("Построение эмбеддингов...")
        char_embeddings = build_char_embeddings(ae_model, text)
        torch.save(char_embeddings, EMBEDDING_CACHE_PATH)

    dataset = NgramDataset(text, char_embeddings)
    loader = DataLoader(dataset, batch_size=PRED_BATCH_SIZE, shuffle=True)

    model = TripletPredictor().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=PRED_LR)
    criterion = nn.MSELoss()

    print(f"Обучение предиктора на {len(dataset)} примерах (выход 32 бита)...")
    for epoch in range(PRED_EPOCHS):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == PRED_EPOCHS - 1:
            print(f"Epoch {epoch+1}/{PRED_EPOCHS}, Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), PREDICTOR_MODEL_PATH)
    print(f"Предиктор сохранён в {PREDICTOR_MODEL_PATH}")


# ==================== Генерация ====================
@torch.no_grad()
def generate(ae_model, predictor_model, seed_text, length=40):
    ae_model.eval()
    predictor_model.eval()

    if not os.path.exists(EMBEDDING_CACHE_PATH):
        print("Нет кэша эмбеддингов. Запустите обучение предиктора.")
        return
    char_embeddings = torch.load(EMBEDDING_CACHE_PATH, map_location="cpu")   # 8‑мерные входы
    emb_map = {c: e.to(DEVICE) for c, e in char_embeddings.items()}

    # Все известные символы и их 32‑битные представления
    chars = list(emb_map.keys())
    bits_map = {c: char_to_bits(c) for c in chars}   # на устройстве
    # Тензоры для быстрого поиска
    bits_tensor = torch.stack([bits_map[c] for c in chars])  # [num_chars, 32]

    current_text = seed_text.upper()
    print(f"Seed: {current_text} | Result: ", end="", flush=True)

    for _ in range(length):
        last_char = current_text[-1]
        if last_char not in emb_map:
            break
        v_curr = emb_map[last_char].unsqueeze(0)

        votes = {}
        for d in range(1, MAX_DIST + 1):
            idx = len(current_text) - d
            if idx < 0:
                continue
            past_char = current_text[idx]
            if past_char not in emb_map:
                continue
            v_past = emb_map[past_char].unsqueeze(0)
            dist_val = torch.tensor([[d / MAX_DIST]], device=DEVICE)
            inp = torch.cat((v_curr, v_past, dist_val), dim=1)
            pred_vec = predictor_model(inp).squeeze(0)   # 32-мерный вектор

            # Ближайший сосед по 32 битам
            dists = torch.norm(bits_tensor - pred_vec, dim=1)   # [num_chars]
            best_idx = torch.argmin(dists).item()
            best_char = chars[best_idx]
            weight = 1.0 / (d ** 1.2)
            votes[best_char] = votes.get(best_char, 0) + weight

        if not votes:
            break
        next_char = max(votes, key=votes.get)
        current_text += next_char
        print(next_char, end="", flush=True)

    print()
    return current_text


# ==================== Главный запуск ====================
def main():
    if not os.path.exists(TEXT_FILE):
        raise FileNotFoundError(f"Файл '{TEXT_FILE}' отсутствует. Создайте его с обучающим текстом (например, 'ПРЕЗИДЕНТ ПРЕКРАСНО ПРЕДСКАЗАЛ ПРЕПЯТСТВИЕ').")

    train_autoencoder()

    ae = UnicodeAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load(AE_MODEL_PATH, map_location=DEVICE))
    ae.eval()

    train_predictor(ae)

    predictor = TripletPredictor().to(DEVICE)
    predictor.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=DEVICE))
    predictor.eval()

    print("\n--- Генерация ---")
    generate(ae, predictor, "ПРЕП", length=40)


if __name__ == "__main__":
    main()