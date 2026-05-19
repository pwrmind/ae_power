import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

# ==========================================
# 1. ЗАЩИЩЕННЫЕ КЛАССЫ ВЫЧИСЛЕНИЙ
# ==========================================

class MultiHeadParametricMemoryAttention(nn.Module):
    """Применяет вашу формулу к оценкам взаимодействия в безопасном float32."""
    def __init__(self, num_heads: int, initial_base: float = 4.0, initial_shift: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        
        raw_base_init = np.log(initial_base - 1.0)
        self.raw_base = nn.Parameter(torch.full((1, num_heads, 1, 1), raw_base_init, dtype=torch.float32))
        self.shift = nn.Parameter(torch.full((1, num_heads, 1, 1), initial_shift, dtype=torch.float32))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        # Принудительно работаем в float32 для защиты от переполнения степеней
        scores_f32 = scores.to(torch.float32)
        base = 1.0 + torch.exp(self.raw_base)
        power = torch.clamp(scores_f32 + self.shift, -15.0, 15.0) # Чуть сузили диапазон для безопасности
        
        gate = (base ** power) / (1.0 + (base ** power))
        return torch.clamp(gate, 1e-6, 1.0 - 1e-6)


class LlamaAttentionWrapper(nn.Module):
    """Безопасная обертка с вычислениями в float32 пространстве."""
    def __init__(self, original_attention, num_heads):
        super().__init__()
        self.original_attn = original_attention
        self.pmg_gate = MultiHeadParametricMemoryAttention(num_heads)
        # Инициализируем mix_coef в float32 для стабильности обновления градиентов
        self.mix_coef = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))
        self.num_heads = num_heads

    def _dummy_scores(self, h, heads):
        B, L, D = h.shape
        h_heads = h.view(B, L, heads, D // heads).transpose(1, 2)
        return torch.matmul(h_heads, h_heads.transpose(-2, -1)) / np.sqrt(D // heads)

    def forward(self, hidden_states, *args, **kwargs):
        orig_outputs = self.original_attn(hidden_states, *args, **kwargs)
        
        if isinstance(orig_outputs, tuple):
            attn_output = orig_outputs[0]
        else:
            attn_output = orig_outputs
            
        # Считаем маску PMG
        pmg_mod = self.pmg_gate(self._dummy_scores(hidden_states, self.num_heads))
        scale = pmg_mod.mean(dim=-1).mean(dim=1).unsqueeze(-1)
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Переводим всё в float32 перед умножением!
        attn_output_f32 = attn_output.to(torch.float32)
        scale_f32 = scale.to(torch.float32)
        
        # Выполняем адаптивное масштабирование
        modified_output_f32 = attn_output_f32 * (1.0 + self.mix_coef * scale_f32)
        
        # Возвращаем в оригинальный float16 для модели
        modified_output = modified_output_f32.to(attn_output.dtype)
        
        if isinstance(orig_outputs, tuple):
            return (modified_output,) + orig_outputs[1:]
        return modified_output

# ==========================================
# 2. ЗАГРУЗКА И НАСТРОЙКА МОДЕЛИ
# ==========================================

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Загрузка {model_name} на {device}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
num_heads = model.config.num_attention_heads

print("Интеграция чистых PMG-активаций через безопасные обертки...")
for i in range(len(model.model.layers)):
    orig_layer = model.model.layers[i].self_attn
    wrapped_layer = LlamaAttentionWrapper(orig_layer, num_heads).to(device)
    model.model.layers[i].self_attn = wrapped_layer

# Замораживаем веса базовой модели
for param in model.parameters():
    param.requires_grad = False

# Размораживаем строго только ваши параметры управления
for layer in model.model.layers:
    layer.self_attn.pmg_gate.raw_base.requires_grad = True
    layer.self_attn.pmg_gate.shift.requires_grad = True
    layer.self_attn.mix_coef.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"-> Успешно! Число обучаемых параметров вашей математики: {trainable_params}")

# ==========================================
# 3. ЦИКЛ ОБУЧЕНИЯ С ПОНИЖЕННЫМ LR
# ==========================================

train_texts = [
    "Природа — это сложная система, где всё взаимодействует со всем одновременно.",
    "Математика капли воды скрывает в себе фрактальные деревья и цепные реакции.",
    "Искусственный интеллект меняет коэффициенты внимания, обучаясь логике связей.",
    "Если сложить вероятности бинарного дерева до бесконечности, мы получим единицу.",
] * 15

inputs = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
input_ids = inputs["input_ids"].to(device)
labels = input_ids.clone()

# Поставили более аккуратный lr=1e-4 для исключения рывков градиента
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

print("\n=== Старт стабильного обучения (SmolLM-360M) ===")
model.train()

for epoch in range(5):
    t0 = time.time()
    optimizer.zero_grad()
    
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    
    loss.backward()
    
    # Жесткий клиппинг градиентов для полной стабильности
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=0.5)
    
    optimizer.step()
    
    # Сбор чистой статистики
    all_bases, all_shifts, all_coefs = [], [], []
    with torch.no_grad():
        for layer in model.model.layers:
            actual_base = 1.0 + torch.exp(layer.self_attn.pmg_gate.raw_base)
            all_bases.append(actual_base.mean().item())
            all_shifts.append(layer.self_attn.pmg_gate.shift.mean().item())
            all_coefs.append(layer.self_attn.mix_coef.mean().item())
            
    epoch_time = time.time() - t0
    print(f"Эпоха {epoch+1}/5 | Loss: {loss.item():.4f} | Время: {epoch_time:.2f}с")
    print(f"   [PMG параметры] Base: {np.mean(all_bases):.4f} | Shift: {np.mean(all_shifts):.4f} | Вес коэффициента (mix_coef): {np.mean(all_coefs):.4f}")

print("\n=== Обучение завершено без ошибок! ===")
