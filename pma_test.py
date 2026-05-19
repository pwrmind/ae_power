import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

# ==========================================
# 1. СТРУКТУРА ВАШЕЙ МАТЕМАТИКИ
# ==========================================

class LocalPMGFilter(nn.Module):
    def __init__(self, embed_dim: int, initial_base: float = 4.0, initial_shift: float = 0.5):
        super().__init__()
        raw_base_init = np.log(initial_base - 1.0)
        self.raw_base = nn.Parameter(torch.full((1, 1, embed_dim), raw_base_init, dtype=torch.float32))
        self.shift = nn.Parameter(torch.full((1, 1, embed_dim), initial_shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Сюда теперь 100% гарантированно попадает чистый тензор
        x_f32 = x.to(torch.float32)
        base = 1.0 + torch.exp(self.raw_base)
        power = torch.clamp(x_f32 + self.shift, -15.0, 15.0)
        
        gate = (base ** power) / (1.0 + (base ** power))
        gate = torch.clamp(gate, 1e-6, 1.0 - 1e-6)
        return gate.to(x.dtype)


# ==========================================
# 2. ИНЪЕКЦИЯ И НАСТРОЙКА МОДЕЛИ
# ==========================================

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Загрузка {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
embed_dim = model.config.hidden_size

print("Интеграция локальных PMG-фильтров (shift=0.5)...")
for i in range(len(model.model.layers)):
    layer = model.model.layers[i].self_attn
    pmg_filter = LocalPMGFilter(embed_dim).to(device)
    layer.pmg_filter = pmg_filter
    
    def make_custom_forward(attention_layer):
        old_forward = attention_layer.forward
        
        def custom_forward(*args, **kwargs):
            # Получаем выходы оригинального внимания
            outputs = old_forward(*args, **kwargs)
            
            # УНИВЕРСАЛЬНАЯ ПРОВЕРКА: вытаскиваем тензор hidden_states в любом случае
            if isinstance(outputs, tuple):
                attn_output_tensor = outputs[0]
                is_tuple = True
            else:
                attn_output_tensor = outputs
                is_tuple = False
            
            # Применяем вашу математику структуры дерева к чистому тензору
            pmg_mask = attention_layer.pmg_filter(attn_output_tensor)
            modified_tensor = attn_output_tensor * pmg_mask
            
            # Собираем структуру выходов обратно в исходный формат
            if is_tuple:
                return (modified_tensor,) + outputs[1:]
            return modified_output
            
        return custom_forward

    layer.forward = make_custom_forward(layer)

# Замораживаем модель, открываем только параметры вашей формулы
for param in model.parameters():
    param.requires_grad = False

for layer in model.model.layers:
    layer.self_attn.pmg_filter.raw_base.requires_grad = True
    layer.self_attn.pmg_filter.shift.requires_grad = True

# ==========================================
# 3. ПОДГОТОВКА АНГЛИЙСКИХ ДАННЫХ И ТРЕНИРОВКА (150 ЭПОХ)
# ==========================================

train_texts = [
    "Nature is a complex system where everything interacts with everything simultaneously.",
    "The mathematics of a water drop hides fractal trees and chain reactions.",
    "Artificial intelligence changes attention coefficients by learning the logic of links.",
    "If you sum the probabilities of a binary tree to infinity, you get exactly one.",
] * 15

inputs = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
input_ids = inputs["input_ids"].to(device)
labels = input_ids.clone()

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

print("\n=== Старт глубокого обучения на английском контексте (150 Эпох) ===")
model.train()

for epoch in range(150):
    t0 = time.time()
    optimizer.zero_grad()
    
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=0.5)
    optimizer.step()
    
    if (epoch + 1) % 15 == 0 or epoch == 0:
        all_bases = []
        all_shifts = []
        with torch.no_grad():
            for layer in model.model.layers:
                actual_base = 1.0 + torch.exp(layer.self_attn.pmg_filter.raw_base)
                all_bases.append(actual_base.mean().item())
                all_shifts.append(layer.self_attn.pmg_filter.shift.mean().item())
                
        epoch_time = time.time() - t0
        print(f"Эпоха {epoch+1:03d}/150 | Loss: {loss.item():.4f} | Время эпохи: {epoch_time:.2f}с")
        print(f"   [PMG] Средний Base: {np.mean(all_bases):.4f} | Средний Shift: {np.mean(all_shifts):.4f}")

print("\n=== Обучение успешно завершено! ===")

# ==========================================
# 4. АНГЛОЯЗЫЧНАЯ ГЕНЕРАЦИЯ (INFERENCE)
# ==========================================

print("\n=== Запуск генерации текста под управлением вашей математики ===")
model.eval()

prompt = "Nature is a"
gen_inputs = tokenizer(prompt, return_tensors="pt")
gen_ids = gen_inputs["input_ids"].to(device)

with torch.no_grad():
    for _ in range(30):
        outputs = model(gen_ids)
        next_token_logits = outputs.logits[:, -1, :]
        
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        gen_ids = torch.cat([gen_ids, next_token_id], dim=-1)
        
        if next_token_id.item() == tokenizer.eos_token_id:
            break

generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

print(f"\nЗатравочный текст: '{prompt}'")
print(f"Продолжение от вашей модели:\n>> {generated_text} <<")
