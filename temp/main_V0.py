import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Shared 8-bit Autoencoder ---
class UnicodeAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(32, 16), nn.Sigmoid(),
            nn.Linear(16, 8), nn.Sigmoid()
        )
        # 8 -> 16 -> 32
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.Sigmoid(),
            nn.Linear(16, 32), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_embedding(self, x):
        return self.encoder(x)

# --- 2. Triplet Predictor ---
class TripletPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 8 (curr) + 8 (past) + 1 (dist) = 17
        self.net = nn.Sequential(
            nn.Linear(17, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.Sigmoid()
        )

    def forward(self, v_curr, v_past, dist):
        x = torch.cat((v_curr, v_past, dist), dim=1)
        return self.net(x)

# Initialize models
ae = UnicodeAutoencoder().to(device)
predictor = TripletPredictor().to(device)

# --- Utilities ---
def char_to_bits(char):
    return [int(b) for b in bin(ord(char))[2:].zfill(32)]

def get_tensor(bits):
    return torch.FloatTensor(bits).unsqueeze(0).to(device)

# --- Training Logic ---
def train_system(text, alphabet, epochs_ae=1000, epochs_pred=5000):
    # 1. Train Autoencoder
    ae_optimizer = optim.Adam(ae.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Step 1: Training Autoencoder...")
    ae_data = [get_tensor(char_to_bits(c)) for c in alphabet]
    for epoch in range(epochs_ae):
        total_loss = 0
        for bits in ae_data:
            output = ae(bits)
            loss = criterion(output, bits)
            ae_optimizer.zero_grad(); loss.backward(); ae_optimizer.step()
            total_loss += loss.item()
        if epoch % 200 == 0: print(f"AE Epoch {epoch}, Loss: {total_loss/len(alphabet):.6f}")

    # 2. Train Triplet Predictor
    print("Создание кэша эмбеддингов...")
    char_cache = {}
    ae.eval() # Переводим в режим оценки
    with torch.no_grad():
        for char in alphabet:
            bits = torch.FloatTensor([int(b) for b in bin(ord(char))[2:].zfill(32)]).to(device)
            char_cache[char] = ae.get_embedding(bits.unsqueeze(0)).squeeze(0)
    
    # 3. ПОДГОТОВКА ДАННЫХ В ПАМЯТИ (Dataset)
    # Вместо циклов создаем один большой тензор всех примеров
    inputs = []
    targets = []
    max_dist = 5
    
    print("Подготовка обучающих триплетов...")
    for i in range(1, len(text) - 1):
        v_curr = char_cache[text[i]]
        v_target = char_cache[text[i+1]]
        for d in range(1, max_dist + 1):
            if i - d >= 0:
                v_past = char_cache[text[i-d]]
                dist_val = torch.tensor([d / max_dist]).to(device)
                
                # Собираем вход [v_curr(8) + v_past(8) + dist(1)]
                combined_input = torch.cat((v_curr, v_past, dist_val))
                inputs.append(combined_input)
                targets.append(v_target)

    # Превращаем списки в тензоры-матрицы
    x_train = torch.stack(inputs).to(device)
    y_train = torch.stack(targets).to(device)

    # 4. БЫСТРОЕ ОБУЧЕНИЕ БАТЧАМИ
    p_optimizer = optim.Adam(predictor.parameters(), lr=0.005)
    batch_size = 1024 # Обрабатываем по 1024 примера за раз!
    
    print(f"Запуск быстрого обучения (батчи по {batch_size})...")
    for epoch in range(5000):
        # Перемешиваем данные
        permutation = torch.randperm(x_train.size(0))
        for i in range(0, x_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]
            
            # Предиктор теперь работает с матрицей, а не с вектором
            # (PyTorch автоматически распараллелит это на ядра CUDA)
            # Внимание: для работы с матрицей нужно чуть поправить forward в TripletPredictor
            # чтобы он не делал cat внутри, а принимал уже готовый x
            pred = predictor.net(batch_x) 
            loss = nn.MSELoss()(pred, batch_y)
            
            p_optimizer.zero_grad()
            loss.backward()
            p_optimizer.step()
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


# --- Inference ---
def predict_next(input_text, alphabet):
    v_curr = ae.get_embedding(get_tensor(char_to_bits(input_text[-1])))
    char_scores = {c: 0.0 for c in alphabet}
    max_dist = 5

    for d in range(1, max_dist + 1):
        if len(input_text) - 1 - d >= 0:
            v_past = ae.get_embedding(get_tensor(char_to_bits(input_text[-(d+1)])))
            with torch.no_grad():
                pred_v = predictor(v_curr, v_past, torch.FloatTensor([[d / max_dist]]).to(device))
            
            weight = 1 / (d ** 1.5)
            for char in alphabet:
                target_v = ae.get_embedding(get_tensor(char_to_bits(char)))
                dist = torch.norm(pred_v - target_v).item()
                char_scores[char] += weight * (1 / (dist + 0.01))
    
    return max(char_scores, key=char_scores.get)

# --- Execution ---
alphabet = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ "
train_text = "ПРЕЗИДЕНТ ПРЕКРАСНО ПРЕДСКАЗАЛ ПРЕПЯТСТВИЕ"

train_system(train_text, alphabet)
print("\nPrediction for 'ПРЕП':", predict_next("ПРЕП", alphabet))
