import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

# ==========================================
# 1. ОПРЕДЕЛЕНИЕ ВАШИХ КАСТОМНЫХ КЛАССОВ
# ==========================================

class MultiHeadParametricMemoryAttention(nn.Module):
    """Кастомное многоголовое внимание на базе вашей формулы PMG вместо Softmax."""
    def __init__(self, embed_dim: int, num_heads: int, initial_base: float = 4.0, initial_shift: float = -1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Проекции (для SmolLM2 используем bfloat16 или float16 для скорости)
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=torch.float16)
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=torch.float16)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=torch.float16)
        self.out_proj = nn.Linear(embed_dim, embed_dim, dtype=torch.float16)
        
        # Обучаемые параметры инициализируем в float32 для стабильности градиентов
        raw_base_init = np.log(initial_base - 1.0)
        self.raw_base = nn.Parameter(torch.full((1, num_heads, 1, 1), raw_base_init, dtype=torch.float32))
        self.shift = nn.Parameter(torch.full((1, num_heads, 1, 1), initial_shift, dtype=torch.float32))

    def _parametric_gate(self, x: torch.Tensor) -> torch.Tensor:
        # Приводим параметры к типу float16 (типу входных данных) прямо в процессе
        base = 1.0 + torch.exp(self.raw_base.to(x.dtype))
        power = torch.clamp(x + self.shift.to(x.dtype), -20.0, 20.0)
        gate = (base ** power) / (1.0 + (base ** power))
        return torch.clamp(gate, 1e-7, 1.0 - 1e-7)

    def forward(self, x: torch.Tensor, attention_mask=None):
        # x имеет форму: [Batch, Seq_Len, Embed_Dim]
        B, L, D = x.shape
        
        # Проекции и разбиение на головы
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Скалярное произведение матриц
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Применение вашей формулы активации
        attention_matrix = self._parametric_gate(scores)
        
        # Сборка контекста обратно
        context = torch.matmul(attention_matrix, V)
        context = context.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(context)


class HybridAttentionWrapper(nn.Module):
    """Обертка, плавно подмешивающая вашу математику к оригинальному Softmax для архитектуры Llama."""
    def __init__(self, original_attention, embed_dim, num_heads):
        super().__init__()
        self.original_attn = original_attention
        self.pmg_attn = MultiHeadParametricMemoryAttention(embed_dim, num_heads)
        # 5% влияния вашей математики на старте.
        self.mix_coef = nn.Parameter(torch.tensor([0.05], dtype=torch.float16))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Получаем выход оригинального внимания Llama (возвращает кортеж: (output, weights, past_key_value))
        orig_output = self.original_attn(hidden_states, attention_mask=attention_mask, **kwargs)
        
        # Считаем параллельный поток по вашей формуле
        pmg_output = self.pmg_attn(hidden_states)
        
        if isinstance(orig_output, tuple):
            combined_hidden = orig_output[0] + self.mix_coef * pmg_output
            return (combined_hidden,) + orig_output[1:]
        
        return orig_output + self.mix_coef * pmg_output

# ==========================================
# 2. ЗАГРУЗКА И НАСТРОЙКА МОДЕЛИ SMOL-LM2
# ==========================================

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
print(f"Запуск инициализации... Загрузка {model_name} (займет ~700 МБ VRAM)")

tokenizer = AutoTokenizer.from_pretrained(model_name)
# SmolLM2 требует настройки padding токена
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="cuda"
)

# Модификация слоев внимания архитектуры Llama
embed_dim = model.config.hidden_size
num_heads = model.config.num_attention_heads

print("Интеграция Parametric Memory Gate в слои SmolLM...")
for i in range(len(model.model.layers)):
    orig_layer = model.model.layers[i].self_attn
    model.model.layers[i].self_attn = HybridAttentionWrapper(orig_layer, embed_dim, num_heads)

# Настройка градиентов: замораживаем базу, обучаем ТОЛЬКО вашу математику
print("Настройка параметров обучения...")
for param in model.parameters():
    param.requires_grad = False

# Размораживаем все новые добавленные слои
for i in range(len(model.model.layers)):
    for param in model.model.layers[i].self_attn.pmg_attn.parameters():
        param.requires_grad = True
    model.model.layers[i].self_attn.mix_coef.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"-> Успешно! Число обучаемых параметров вашей математики: {trainable_params:,}")

# ==========================================
# 3. ПОДГОТОВКА ДАННЫХ И ТРЕНИРОВКА
# ==========================================

# Небольшой текстовый датасет для проверки сжатия информации
train_texts = [
    "Природа — это сложная система, где всё взаимодействует со всем одновременно.",
    "Математика капли воды скрывает в себе фрактальные деревья и цепные реакции.",
    "Искусственный интеллект меняет коэффициенты внимания, обучаясь логике связей.",
    "Если сложить вероятности бинарного дерева до бесконечности, мы получим единицу.",
] * 15  # 60 примеров для плотного мини-обучения

inputs = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
input_ids = inputs["input_ids"].cuda()
labels = input_ids.clone()

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

print("\n=== Старт обучения на вашей RTX 4070 (SmolLM-360M) ===")
model.train()

for epoch in range(5):
    t0 = time.time()
    optimizer.zero_grad()
    
    # Прямой проход
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    
    # Обратный проход
    loss.backward()
    optimizer.step()
    
    # Сбор статистики параметров по слоям
    all_bases = []
    all_shifts = []
    all_coefs = []
    
    with torch.no_grad():
        for layer in model.model.layers:
            actual_base = 1.0 + torch.exp(layer.self_attn.pmg_attn.raw_base)
            all_bases.append(actual_base.mean().item())
            all_shifts.append(layer.self_attn.pmg_attn.shift.mean().item())
            all_coefs.append(layer.self_attn.mix_coef.mean().item())
            
    avg_base = np.mean(all_bases)
    avg_shift = np.mean(all_shifts)
    avg_coef = np.mean(all_coefs)
    epoch_time = time.time() - t0
    
    print(f"Эпоха {epoch+1}/5 | Loss: {loss.item():.4f} | Время: {epoch_time:.2f}с")
    print(f"   [Текущие средние PMG] Base: {avg_base:.4f} | Shift: {avg_shift:.4f} | Вес в модели (mix_coef): {avg_coef:.4f}")

print("\n=== Обучение успешно завершено! ===")
