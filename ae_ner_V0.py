import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Tuple

# ==========================================
# 1. ВАШИ НАРАБОТКИ (Генерация фичей и PMG)
# ==========================================

def get_char_vector(char: str) -> torch.FloatTensor:
    """Генерация 36-мерного вектора для символа Unicode."""
    code = ord(char)
    bits = [int(b) * 2 - 1 for b in format(code, '032b')]
    freq = code / 100.0
    extra = [np.sin(freq), np.cos(freq), np.sin(freq / 10), np.cos(freq / 10)]
    return torch.FloatTensor(bits + extra)


class ParametricMemoryGate(nn.Module):
    """Кастомная параметрическая функция активации PMG."""
    def __init__(self, initial_base: float = 4.0, initial_shift: float = -1.0):
        super().__init__()
        if initial_base <= 1.0:
            raise ValueError("initial_base must be strictly greater than 1.0")
        raw_base_init = np.log(initial_base - 1.0)
        self.raw_base = nn.Parameter(torch.tensor([raw_base_init], dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor([initial_shift], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = 1.0 + torch.exp(self.raw_base)
        power = torch.clamp(x + self.shift, -20.0, 20.0)
        gate = (base ** power) / (1.0 + (base ** power))
        eps = 1e-7
        return torch.clamp(gate, eps, 1.0 - eps)


class UnicodeAutoencoder(nn.Module):
    """Ваш автоэнкодер для сжатия 36-мерного вектора в emb_dim (8)."""
    def __init__(self, bits=32, emb_dim=8):
        super().__init__()
        self.pmg1 = ParametricMemoryGate()
        self.pmg2 = ParametricMemoryGate()
        self.pmg3 = ParametricMemoryGate()
        self.pmg4 = ParametricMemoryGate()

        self.encoder = nn.Sequential(
            nn.Linear(36, 64),
            nn.BatchNorm1d(64),
            self.pmg1,
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            self.pmg2,
            nn.Linear(32, emb_dim)  # Латентное пространство
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.BatchNorm1d(32),
            self.pmg3,
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            self.pmg4,
            nn.Linear(64, bits),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ==========================================
# 2. КАСТОМНАЯ PMG-LSTM АРХИТЕКТУРА ДЛЯ NER
# ==========================================

class CustomPMGCell(nn.Module):
    """Ячейка памяти LSTM, управляемая ParametricMemoryGate вместо Sigmoid."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, hidden_size * 4)
        self.h2h = nn.Linear(hidden_size, hidden_size * 4)
        
        # Индивидуальные PMG для ворот памяти
        self.forget_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-0.5)
        self.input_gate  = ParametricMemoryGate(initial_base=4.0, initial_shift=-0.5)
        self.output_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-0.5)
        self.c_gate = nn.Tanh()

    def forward(self, x: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = hx
        gates = self.x2h(x) + self.h2h(h_prev)
        i_gate, f_gate, g_gate, o_gate = list(gates.chunk(4, dim=-1))
        
        f = self.forget_gate(f_gate)
        i = self.input_gate(i_gate)
        o = self.output_gate(o_gate)
        g = self.c_gate(g_gate)
        
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class CharacterPMGNetwork(nn.Module):
    """
    Улучшенная двунаправленная (Bidirectional) NER сеть.
    Смотрит на текст с двух сторон, чтобы идеально определять границы дат и имен.
    """
    def __init__(self, pre_trained_autoencoder: UnicodeAutoencoder, hidden_size: int, num_classes: int):
        super().__init__()
        # Замораживаем веса вашего энкодера
        self.char_encoder = pre_trained_autoencoder.encoder
        for param in self.char_encoder.parameters():
            param.requires_grad = True
            
        emb_dim = self.char_encoder[-1].out_features
        self.hidden_size = hidden_size
        
        # Две ячейки: для прямого хода (forward) и обратного (backward)
        self.rnn_cell_forward = CustomPMGCell(input_size=emb_dim, hidden_size=hidden_size)
        self.rnn_cell_backward = CustomPMGCell(input_size=emb_dim, hidden_size=hidden_size)
        
        # Классификатор принимает объединенный вектор из двух направлений (hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # Сжатие через ваш энкодер (36 -> 8)
        x_reshaped = x.view(batch_size * seq_len, 36)
        latent_vectors = self.char_encoder(x_reshaped).view(batch_size, seq_len, -1)
        
        # --- ПРЯМОЙ ХОД (Слева направо) ---
        h_f = torch.zeros(batch_size, self.hidden_size, device=device)
        c_f = torch.zeros(batch_size, self.hidden_size, device=device)
        outputs_f = []
        for t in range(seq_len):
            h_f, c_f = self.rnn_cell_forward(latent_vectors[:, t, :], (h_f, c_f))
            outputs_f.append(h_f)
            
        # --- ОБРАТНЫЙ ХОД (Справа налево) ---
        h_b = torch.zeros(batch_size, self.hidden_size, device=device)
        c_b = torch.zeros(batch_size, self.hidden_size, device=device)
        outputs_b = [None] * seq_len
        for t in reversed(range(seq_len)):
            h_b, c_b = self.rnn_cell_backward(latent_vectors[:, t, :], (h_b, c_b))
            outputs_b[t] = h_b
            
        # Конкатенируем выходы обоих направлений для каждого шага
        final_outputs = []
        for t in range(seq_len):
            combined = torch.cat([outputs_f[t], outputs_b[t]], dim=-1) # [batch_size, hidden_size * 2]
            final_outputs.append(self.classifier(combined))
            
        return torch.stack(final_outputs, dim=1)




# ==========================================
# 3. ГЕНЕРАТОР СИНТЕТИЧЕСКИХ ДАННЫХ ДЛЯ NER
# ==========================================

# Определяем теги (классы) для каждого символа
TAGS = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-DATE': 3, 'I-DATE': 4}
INV_TAGS = {v: k for k, v in TAGS.items()}

NAMES = ["Иван", "Анна", "Петр", "Мария", "Олег", "Елена", "Дмитрий"]
NOUNS = ["встретил", "увидел", "позвонил", "написал", "приехал", "ушел"]
DATES = ["15.05.2024", "01.01.2000", "23.11.1995", "09.05.1945", "31.12.2025"]

def generate_sentence_data():
    """Создает случайную строку и карту разметки (тегов) для каждого символа."""
    name = random.choice(NAMES)
    action = random.choice(NOUNS)
    date = random.choice(DATES)
    
    # Собираем предложение типа: "Иван приехал 15.05.2024"
    sentence = f"{name} {action} {date}"
    labels = []
    
    # Разметка для Имени
    for i, char in enumerate(name):
        labels.append(TAGS['B-PER'] if i == 0 else TAGS['I-PER'])
    labels.append(TAGS['O']) # пробел
    
    # Разметка для обычного слова
    for char in action:
        labels.append(TAGS['O'])
    labels.append(TAGS['O']) # пробел
    
    # Разметка для Даты
    for i, char in enumerate(date):
        labels.append(TAGS['B-DATE'] if i == 0 else TAGS['I-DATE'])
        
    return sentence, labels

def make_batch(batch_size=16, seq_len=35):
    """Формирует тренировочный батч с фиксированной длиной последовательности."""
    batch_x, batch_y = [], []
    for _ in range(batch_size):
        text, labels = generate_sentence_data()
        
        # Дополняем пробелами или обрезаем до фиксированной длины seq_len
        if len(text) < seq_len:
            text = text.ljust(seq_len, ' ')
            labels = labels + [TAGS['O']] * (seq_len - len(labels))
        else:
            text = text[:seq_len]
            labels = labels[:seq_len]
            
        # Конвертируем строку в 36-мерные фичи
        x_vectors = torch.stack([get_char_vector(c) for c in text])
        batch_x.append(x_vectors)
        batch_y.append(torch.tensor(labels, dtype=torch.long))
        
    return torch.stack(batch_x), torch.stack(batch_y)


# ==========================================
# 4. ДЕМОНСТРАЦИЯ И ОБУЧЕНИЕ МОДЕЛИ
# ==========================================

if __name__ == "__main__":
    print("Инициализация весов и моделей (Bi-LSTM)...")
    autoencoder = UnicodeAutoencoder(bits=32, emb_dim=8)
    model = CharacterPMGNetwork(pre_trained_autoencoder=autoencoder, hidden_size=64, num_classes=len(TAGS))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("\nСтарт симуляции обучения...")
    model.train()
    for epoch in range(1, 101):
        x_train, y_train = make_batch(batch_size=32, seq_len=40)
        
        optimizer.zero_grad()
        predictions = model(x_train)
        loss = criterion(predictions.view(-1, len(TAGS)), y_train.view(-1))
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Эпоха {epoch:3d}/100 | Loss: {loss.item():.4f}")
            
    print("\nПроверка обученной модели на тесте:")
    model.eval()
    with torch.no_grad():
        test_text = "Мария ушел 01.01.2000"
        test_x = torch.stack([get_char_vector(c) for c in test_text]).unsqueeze(0)
        
        pred_out = model(test_x)
        pred_classes = torch.argmax(pred_out, dim=-1).squeeze(0).tolist()
        
        print(f"{'Символ':<8} | {'Предсказанный тег':<10}")
        print("-" * 25)
        for char, class_id in zip(test_text, pred_classes):
            print(f"   '{char}'   | {INV_TAGS[class_id]}")
            
    print("\nВывод выученных параметров PMG гейтов (Прямой ход):")
    with torch.no_grad():
        # Извлекаем параметры напрямую через формулу из вашего класса ParametricMemoryGate
        f_gate = model.rnn_cell_forward.forget_gate
        learned_base = 1.0 + torch.exp(f_gate.raw_base).item()
        learned_shift = f_gate.shift.item()
        print(f"Forget Gate -> Выученная база (base): {learned_base:.4f}, Сдвиг (shift): {learned_shift:.4f}")
