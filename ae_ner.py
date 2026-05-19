import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Tuple

# ==========================================
# 1. СТРУКТУРА ВАШИХ МОДЕЛЕЙ (Фичи и PMG)
# ==========================================

def get_char_vector(char: str) -> torch.FloatTensor:
    """Превращает символ в ваш кастомный 36-мерный вектор."""
    code = ord(char)
    bits = [int(b) * 2 - 1 for b in format(code, '032b')]
    freq = code / 100.0
    extra = [np.sin(freq), np.cos(freq), np.sin(freq / 10), np.cos(freq / 10)]
    return torch.FloatTensor(bits + extra)


class ParametricMemoryGate(nn.Module):
    """Ваша кастомная параметрическая активация PMG."""
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
    """Ваш автоэнкодер сжатия пространства символов до 8 измерений."""
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
            nn.Linear(32, emb_dim)
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


class CustomPMGCell(nn.Module):
    """LSTM ячейка памяти, управляемая вашими PMG-активациями."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, hidden_size * 4)
        self.h2h = nn.Linear(hidden_size, hidden_size * 4)
        
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
    """Двунаправленная NER сеть с размороженным сквозным автоэнкодером."""
    def __init__(self, pre_trained_autoencoder: UnicodeAutoencoder, hidden_size: int, num_classes: int):
        super().__init__()
        # Разрешаем обучение автоэнкодера в процессе (Fine-tuning)
        self.char_encoder = pre_trained_autoencoder.encoder
        for param in self.char_encoder.parameters():
            param.requires_grad = True
            
        emb_dim = self.char_encoder[-1].out_features
        self.hidden_size = hidden_size
        
        self.rnn_cell_forward = CustomPMGCell(input_size=emb_dim, hidden_size=hidden_size)
        self.rnn_cell_backward = CustomPMGCell(input_size=emb_dim, hidden_size=hidden_size)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # Сжатие 36 -> 8
        x_reshaped = x.view(batch_size * seq_len, 36)
        latent_vectors = self.char_encoder(x_reshaped).view(batch_size, seq_len, -1)
        
        # Прямой ход
        h_f = torch.zeros(batch_size, self.hidden_size, device=device)
        c_f = torch.zeros(batch_size, self.hidden_size, device=device)
        outputs_f = []
        for t in range(seq_len):
            h_f, c_f = self.rnn_cell_forward(latent_vectors[:, t, :], (h_f, c_f))
            outputs_f.append(h_f)
            
        # Обратный ход
        h_b = torch.zeros(batch_size, self.hidden_size, device=device)
        c_b = torch.zeros(batch_size, self.hidden_size, device=device)
        outputs_b = [None] * seq_len
        for t in reversed(range(seq_len)):
            h_b, c_b = self.rnn_cell_backward(latent_vectors[:, t, :], (h_b, c_b))
            outputs_b[t] = h_b
            
        # Объединение контекстов
        final_outputs = []
        for t in range(seq_len):
            combined = torch.cat([outputs_f[t], outputs_b[t]], dim=-1)
            final_outputs.append(self.classifier(combined))
            
        return torch.stack(final_outputs, dim=1)


# ==========================================
# 2. ПОДГОТОВКА РЕАЛЬНЫХ МУЛЬТИЯЗЫЧНЫХ ДАННЫХ
# ==========================================

# Справочник тегов WikiANN (0: O, 1: B-PER, 2: I-PER, 3: B-ORG, 4: I-ORG, 5: B-LOC, 6: I-LOC)
WIKIANN_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

class MultilingualCharDataset(Dataset):
    """Загрузчик реального датасета с трансформацией слов в посимвольные цепочки."""
    def __init__(self, languages=["ru", "en"], split="train", max_samples=500, seq_len=64):
        self.seq_len = seq_len
        self.samples = []
        
        print(f"Загрузка и конвертация данных WikiANN ({split})...")
        for lang in languages:
            # Загружаем конкретную языковую конфигурацию напрямую из HuggingFace
            raw_data = load_dataset("wikiann", lang, split=split, trust_remote_code=True)
            
            # Ограничиваем выборку для демонстрационного примера
            num_to_take = min(max_samples, len(raw_data))
            for i in range(num_to_take):
                tokens = raw_data[i]["tokens"]
                word_tags = raw_data[i]["ner_tags"]
                
                char_sequence = ""
                char_labels = []
                
                # Конвертируем пословную разметку в посимвольную
                for word, tag_id in zip(tokens, word_tags):
                    # Если это не первое слово в предложении, добавляем пробел
                    if len(char_sequence) > 0:
                        char_sequence += " "
                        char_labels.append(0) # Пробел размечаем тегом 'O'
                        
                    # Нам нужно проверить, является ли тег "начальным" (B-теги: 1, 3, 5)
                    # 1: B-PER, 3: B-ORG, 5: B-LOC
                    is_b_tag = tag_id in [1, 3, 5]
                    
                    # Посимвольно разбираем слово
                    for c_idx, char in enumerate(word):
                        char_sequence += char
                        
                        if tag_id == 0:
                            # Обычный текст сохраняет тег 'O'
                            char_labels.append(0)
                        else:
                            if c_idx == 0 and is_b_tag:
                                # Самый первый символ слова получает оригинальный B-тег
                                char_labels.append(tag_id)
                            else:
                                # Все остальные символы сущности (или если исходный тег уже I-)
                                # получают соответствующий I-тег (B-тег + 1)
                                if is_b_tag:
                                    char_labels.append(tag_id + 1)
                                else:
                                    char_labels.append(tag_id)
                                    
                # Дополнительная жесткая проверка: длины строк и меток должны идеально совпадать
                assert len(char_sequence) == len(char_labels), f"Критический рассинхрон длин: {len(char_sequence)} vs {len(char_labels)}"
                self.samples.append((char_sequence, char_labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, labels = self.samples[idx]
        
        # Обрезаем или дополняем (Padding) до фиксированного размера seq_len
        if len(text) < self.seq_len:
            text_pad = text.ljust(self.seq_len, '\x00') # Технический символ заполнения
            labels_pad = labels + [-100] * (self.seq_len - len(labels)) # Маскируем паддинг
        else:
            text_pad = text[:self.seq_len]
            labels_pad = labels[:self.seq_len]
            
        # Строим 36-мерные фичи для всей строки
        x_tensor = torch.stack([get_char_vector(c) for c in text_pad])
        y_tensor = torch.tensor(labels_pad, dtype=torch.long)
        
        return x_tensor, y_tensor

# ==========================================
# 3. ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    
    # 1. Готовим мультиязычный датасет (Смешиваем русский и английский)
    train_dataset = MultilingualCharDataset(languages=["ru", "en"], split="train", max_samples=600, seq_len=50)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 2. Инициализируем компоненты
    autoencoder = UnicodeAutoencoder(bits=32, emb_dim=8).to(device)
    model = CharacterPMGNetwork(pre_trained_autoencoder=autoencoder, hidden_size=64, num_classes=len(WIKIANN_TAGS)).to(device)
    
    # ignore_index=-100 приказывает лоссу игнорировать символы дополнения (Padding)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    # 3. Обучение
    print("\nСтарт обучения на реальных мультиязычных текстах WikiANN...")
    model.train()
    for epoch in range(1, 11): # Обучим на 10 эпохах для демонстрации
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            
            loss = criterion(predictions.view(-1, len(WIKIANN_TAGS)), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Эпоха {epoch:2d}/10 | Средний Loss: {total_loss / len(train_loader):.4f}")
        
    # ==========================================
    # 4. ТЕСТИРОВАНИЕ НА РАЗНЫХ ЯЗЫКАХ
    # ==========================================
    model.eval()
    print("\n=== Проверка мультиязычности ===")
    
    # Тестируем две фразы на разных языках, которых не было в таком виде в шаблонах
    test_phrases = [
        "Путин приехал в город Москва", 
        "Elon Musk visited London today"
    ]
    
    with torch.no_grad():
        for phrase in test_phrases:
            print(f"\nАнализ строки: \"{phrase}\"")
            # Обрезаем или дополняем фразу до 50 символов для консистентности прохода
            fixed_phrase = phrase[:50].ljust(50, '\x00')
            
            test_x = torch.stack([get_char_vector(c) for c in fixed_phrase]).unsqueeze(0).to(device)
            pred_out = model(test_x)
            pred_classes = torch.argmax(pred_out, dim=-1).squeeze(0).tolist()
            
            # Выводим только значащие символы фразы (без паддинга)
            print(f"  {'Символ':<8} | {'Тег':<8}")
            print("  -------------------")
            for char, class_id in zip(phrase, pred_classes[:len(phrase)]):
                tag_name = WIKIANN_TAGS[class_id]
                # Подсвечиваем только найденные сущности, чтобы лог был читаемым
                if tag_name != "O":
                    print(f"   '{char}'   | \033[92m{tag_name}\033[0m")
                else:
                    print(f"   '{char}'   | {tag_name}")
                    
    print("\nГеометрия гейта Forget Gate после реального текста:")
    f_gate = model.rnn_cell_forward.forget_gate
    print(f"  Base: {1.0 + torch.exp(f_gate.raw_base).item():.4f} | Shift: {f_gate.shift.item():.4f}")
