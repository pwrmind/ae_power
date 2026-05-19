import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

# ==========================================
# 1. КАСТОМНАЯ АКТИВАЦИЯ: ParametricMemoryGate
# ==========================================
class ParametricMemoryGate(nn.Module):
    def __init__(self, initial_base: float = 4.0, initial_shift: float = 0.5):
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

    def get_parameters(self) -> Tuple[float, float]:
        with torch.no_grad():
            actual_base = 1.0 + torch.exp(self.raw_base).item()
            actual_shift = self.shift.item()
            return actual_base, actual_shift

    def __repr__(self) -> str:
        try:
            base, shift = self.get_parameters()
            return f"ParametricMemoryGate(learned_base={base:.4f}, learned_shift={shift:.4f})"
        except Exception:
            return "ParametricMemoryGate()"


# ==========================================
# 2. ГЕНЕРАЦИЯ ВЕКТОРОВ И НАБОР ДАННЫХ
# ==========================================
def get_char_vector(char: str) -> torch.Tensor:
    """Генерирует вектор признаков символа размера 36."""
    code = ord(char)
    # Превращаем код символа в 32 бита (из 0 и 1 делаем -1 и 1)
    bits = [int(b) * 2 - 1 for b in format(code, '032b')]
    
    # Добавляем тригонометрическое расширение (4 признака)
    freq = code / 100.0
    extra = [np.sin(freq), np.cos(freq), np.sin(freq / 10), np.cos(freq / 10)]
    
    return torch.FloatTensor(bits + extra)


def generate_full_unicode_sample():
    """Генерирует репрезентативный набор кодов символов."""
    codes = set()
    for i in range(0, 3000):
        codes.add(i)
    for i in range(0, 1000):
        codes.add(0x4E00 + i)
    for i in range(0x1F600, 0x1F64F):
        codes.add(i)
    return list(codes)


class UnicodeDataset(Dataset):
    """Датасет PyTorch для обучения автоэнкодера."""
    def __init__(self):
        self.codes = generate_full_unicode_sample()
        self.vectors = []
        for code in self.codes:
            char = chr(code)
            self.vectors.append(get_char_vector(char))
        self.vectors = torch.stack(self.vectors)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx], self.vectors[idx]


# ==========================================
# 3. АРХИТЕКТУРА АВТОЭНКОДЕРА СИМВОЛОВ
# ==========================================
class UnicodeAutoencoder(nn.Module):
    def __init__(self, bits=32, emb_dim=8): # Изначально зафиксировано на 8
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
            nn.Linear(32, emb_dim)  # Латентное узкое горлышко = 8
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

    def get_pmg_layers(self):
        return [self.pmg1, self.pmg2, self.pmg3, self.pmg4]


# ==========================================
# 4. МОДЕЛЬ СВЕРХЛЕГКОГО ЭМБЕДДИНГА СЛОВ
# ==========================================
class SuperLightWordEmbedder(nn.Module):
    def __init__(self, trained_unicode_autoencoder: nn.Module):
        super().__init__()
        # Замораживаем обученный энкодер символов
        self.char_encoder = trained_unicode_autoencoder.encoder
        for param in self.char_encoder.parameters():
            param.requires_grad = False
            
        # 8 чисел Mean + 8 чисел Max = 16. Сжимаем обратно в итоговый 8-мерный вектор слова
        self.word_projection = nn.Linear(16, 8, bias=False)
        
    def _get_word_matrix(self, word: str) -> torch.Tensor:
        """Собирает из строки матрицу признаков символов [длина_слова, 36]."""
        char_vectors = [get_char_vector(char) for char in word]
        return torch.stack(char_vectors)

    def forward(self, word: str) -> torch.Tensor:
        # 1. Строим матрицу признаков для слова
        x_char = self._get_word_matrix(word) # [L, 36]
        
        self.char_encoder.eval()
        with torch.no_grad():
            # Получаем латентные представления символов -> [L, 8]
            char_embeddings = self.char_encoder(x_char)
            
        # ========================================================
        # СИНУСОИДАЛЬНОЕ ПОЗИЦИОННОЕ КОДИРОВАНИЕ (PE)
        # ========================================================
        L = char_embeddings.size(0)       # Длина слова
        emb_dim = char_embeddings.size(1) # Успешно подтягивает размер 8 из латента
        
        # Создаем тензор позиций: [0, 1, 2, ..., L-1]
        positions = torch.arange(L, dtype=torch.float32, device=char_embeddings.device).unsqueeze(1)
        
        # Расчет частотных делителей для размерности 8
        div_term = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float32) * -(np.log(10000.0) / emb_dim))
        
        # Заполняем матрицу PE
        pe = torch.zeros(L, emb_dim, device=char_embeddings.device)
        pe[:, 0::2] = torch.sin(positions * div_term)  # Четные индексы — синус
        pe[:, 1::2] = torch.cos(positions * div_term)  # Нечетные индексы — косинус
        
        # Применяем позиционное кодирование через сложение
        char_embeddings = char_embeddings + pe
        # ========================================================

        # 2. Математический пулинг (Mean + Max) вдоль оси символов (dim=0)
        mean_pool = torch.mean(char_embeddings, dim=0)
        max_pool = torch.max(char_embeddings, dim=0).values  
        
        # Соединяем пулинги вместе -> вектор размера 16 (8 + 8)
        combined = torch.cat([mean_pool, max_pool], dim=0)
        
        # 3. Финальная проекция и L2-нормализация -> итоговый вектор размера 8
        word_vector = self.word_projection(combined)
        word_vector = nn.functional.normalize(word_vector, p=2, dim=0)
        
        return word_vector


# ==========================================
# 5. СКРИПТ ОБУЧЕНИЯ И ПРОВЕРКИ
# ==========================================
if __name__ == "__main__":
    print("Инициализация данных...")
    dataset = UnicodeDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    print("Создание модели автоэнкодера...")
    autoencoder = UnicodeAutoencoder(bits=32, emb_dim=8) # Итоговый латент равен 8
    
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    # --- Шаг 1: Обучение автоэнкодера символов ---
    print("\nНачало обучения автоэнкодера символов...")
    autoencoder.train()
    epochs = 150
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, _ in dataloader:
            optimizer.zero_grad()
            target_bits = batch_x[:, :32] 
            predicted_bits = autoencoder(batch_x)
            
            loss = criterion(predicted_bits, target_bits)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Эпоха {epoch+1}/{epochs} | Средняя ошибка (MSE): {epoch_loss/len(dataloader):.5f}")

    print("\nОбучение завершено!")
    print("Итоговые параметры PMG слоев:")
    for i, pmg in enumerate(autoencoder.get_pmg_layers()):
        base, shift = pmg.get_parameters()
        print(f"  PMG {i+1}: learned_base = {base:.4f}, learned_shift = {shift:.4f}")

    print("\nСоздание эмбеддера слов...")
    word_embedder = SuperLightWordEmbedder(autoencoder)

    # --- Шаг 2: Контрастивное обучение проекции слов ---
    print("\nОбучение эмбеддера слов (устранение коллапса пространства через Contrastive Loss)...")
    word_embedder.word_projection.train()
    word_optimizer = optim.Adam(word_embedder.word_projection.parameters(), lr=0.01)
    
    for step in range(500):  
        word_optimizer.zero_grad()
        
        v_privet = word_embedder("привет")
        v_privetik = word_embedder("приветик")
        v_processor = word_embedder("процессор")
        v_pracesor = word_embedder("працесор")
        v_kot = word_embedder("кот")
        v_tok = word_embedder("ток")
        v_smaylik = word_embedder("смайлик")
        
        loss = 0.0
        # Положительные пары (к 1.0)
        loss += (1.0 - torch.dot(v_privet, v_privetik)) ** 2
        loss += (1.0 - torch.dot(v_processor, v_pracesor)) ** 2
        
        # Отрицательные пары (к 0.0)
        loss += (0.0 - torch.dot(v_privet, v_smaylik)) ** 2
        loss += (0.0 - torch.dot(v_processor, v_smaylik)) ** 2
        loss += (0.0 - torch.dot(v_kot, v_smaylik)) ** 2
        loss += (0.0 - torch.dot(v_tok, v_smaylik)) ** 2
        
        # Анаграммы (к 0.4)
        loss += (0.4 - torch.dot(v_kot, v_tok)) ** 2
        
        loss.backward()
        word_optimizer.step()

    # Переводим эмбеддер в финальный режим оценки
    word_embedder.word_projection.eval()

    # --- Шаг 3: Тестирование итоговых метрик ---
    print("\nТестирование косинусного сходства слов:")

    def get_similarity(w1: str, w2: str) -> float:
        v1 = word_embedder(w1)
        v2 = word_embedder(w2)
        return torch.dot(v1, v2).item()

    print(f"Размерность итогового вектора слова: {word_embedder('тест').shape[0]}")
    print(f"Сходство 'привет' и 'приветик': {get_similarity('привет', 'приветик'):.4f}")
    print(f"Сходство 'процессор' и 'працесор' (опечатки): {get_similarity('процессор', 'працесор'):.4f}")
    print(f"Сходство 'кот' и 'ток' (анаграммы): {get_similarity('кот', 'ток'):.4f}")
    print(f"Сходство разных слов ('привет' / 'смайлик'): {get_similarity('привет', 'смайлик'):.4f}")
