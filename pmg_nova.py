import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np
import time

# ==========================================
# 1. ЧИСТЫЙ ФУНДАМЕНТ: ПРЯМЫЕ ПАРАМЕТРЫ PMG
# ==========================================

class ParametricMemoryGate(nn.Module):
    """Ваша формула на прямых, защищенных параметрах без экспонент."""
    def __init__(self, channels: int, initial_base: float = 4.0, initial_shift: float = 0.5):
        super().__init__()
        # Инициализируем параметры напрямую как чистые числа float32
        self.base = nn.Parameter(torch.full((1, 1, channels), initial_base, dtype=torch.float32))
        self.shift = nn.Parameter(torch.full((1, 1, channels), initial_shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.to(torch.float32)
        # Гарантируем, что база всегда строго больше 1, без использования exp()
        safe_base = torch.clamp(self.base, min=1.01)
        power = x_f32 + self.shift
        
        # Безопасный расчет b^p через стабильный метод PyTorch
        log_base = torch.log(safe_base)
        logits = power * log_base
        
        gate = torch.sigmoid(logits)
        return torch.clamp(gate, 1e-6, 1.0 - 1e-6).to(x.dtype)


class PMGAttentionGate2D(nn.Module):
    """2D версия гейта на прямых параметрах для матриц внимания."""
    def __init__(self, num_heads: int, initial_base: float = 4.0, initial_shift: float = 0.5):
        super().__init__()
        self.base = nn.Parameter(torch.full((1, num_heads, 1, 1), initial_base, dtype=torch.float32))
        self.shift = nn.Parameter(torch.full((1, num_heads, 1, 1), initial_shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.to(torch.float32)
        safe_base = torch.clamp(self.base, min=1.01)
        power = x_f32 + self.shift
        
        log_base = torch.log(safe_base)
        logits = power * log_base
        
        gate = torch.sigmoid(logits)
        return torch.clamp(gate, 1e-6, 1.0 - 1e-6)

# ==========================================
# 2. АРХИТЕКТУРА ТРАНСФОРМЕРА
# ==========================================

class PMGCausalAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, dtype=torch.float16)
        self.out_proj = nn.Linear(n_embed, n_embed, dtype=torch.float16)
        self.pmg_matrix = PMGAttentionGate2D(n_head)
        self.scores_norm = nn.LayerNorm(self.head_dim, dtype=torch.float32)

    def forward(self, x, causal_mask):
        B, L, D = x.shape
        q, k, v = self.qkv_proj(x).split(D, dim=-1)
        
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        
        q_f32, k_f32, v_f32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
        
        q_f32 = self.scores_norm(q_f32)
        k_f32 = self.scores_norm(k_f32)
        
        scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = scores + causal_mask[:L, :L].to(torch.float32)
        
        attn_weights = self.pmg_matrix(scores)
        
        out_f32 = torch.matmul(attn_weights, v_f32)
        out = out_f32.transpose(1, 2).contiguous().view(B, L, D).to(torch.float16)
        return self.out_proj(out)


class PMGBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed, dtype=torch.float16)
        self.attn = PMGCausalAttention(n_embed, n_head)
        self.ln2 = nn.LayerNorm(n_embed, dtype=torch.float16)
        
        self.mlp_fc = nn.Linear(n_embed, 4 * n_embed, dtype=torch.float16)
        self.mlp_act = ParametricMemoryGate(4 * n_embed)
        self.mlp_proj = nn.Linear(4 * n_embed, n_embed, dtype=torch.float16)

    def forward(self, x, causal_mask):
        x = x + self.attn(self.ln1(x), causal_mask)
        mlp_out = self.mlp_proj(self.mlp_act(self.mlp_fc(self.ln2(x))))
        x = x + mlp_out
        return x


class PurePMGTransformer(nn.Module):
    def __init__(self, vocab_size, n_embed=256, n_head=8, n_layer=6, max_seq_len=64):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, n_embed, dtype=torch.float16)
        self.pos_emb = nn.Embedding(max_seq_len, n_embed, dtype=torch.float16)
        
        self.blocks = nn.ModuleList([PMGBlock(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed, dtype=torch.float16)
        self.head = nn.Linear(n_embed, vocab_size, bias=False, dtype=torch.float16)
        
        mask = torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, idx, targets=None):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, self.causal_mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

# ==========================================
# 3. ДАННЫЕ В ПАМЯТИ И ТРЕНИРОВКА
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Инициализация... Устройство: {device}")

base_phrases = [
    "Nature is a complex system where everything interacts with everything simultaneously.",
    "The mathematics of a water drop hides fractal trees and chain reactions.",
    "Artificial intelligence changes attention coefficients by learning the logic of links.",
    "If you sum the probabilities of a binary tree to infinity, you get exactly one.",
    "To be, or not to be, that is the question of nature and mathematical reality.",
    "Every drop of water contains more connections than human history can process.",
    "Fractal structures emerge when equations start interacting with their own outputs."
]
text = " ".join(base_phrases * 200)

enc = tiktoken.get_encoding("cl100k_base")
data_tokens = enc.encode_ordinary(text)
vocab_size = enc.n_vocab
print(f"Массив готов! Всего токенов в памяти: {len(data_tokens):,}, Размер словаря: {vocab_size:,}")

def get_batch(data, batch_size, seq_len):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+seq_len], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+seq_len+1], dtype=torch.long) for i in ix])
    return x.to(device), y.to(device)

model = PurePMGTransformer(vocab_size=vocab_size, n_embed=256, n_head=8, n_layer=6, max_seq_len=64).to(device)
print(f"Создана чистая PMG модель. Общее число весов: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

batch_size = 32
seq_len = 64
steps = 600

print("\n=== НАЧАЛО ЖЕЛЕЗОБЕТОННОГО PRE-TRAINING С НУЛЯ ===")
model.train()
t_start = time.time()

for step in range(steps):
    t0 = time.time()
    xb, yb = get_batch(data_tokens, batch_size, seq_len)
    
    optimizer.zero_grad()
    logits, loss = model(xb, yb)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    
    if step % 50 == 0 or step == steps - 1:
        attn_bases, attn_shifts = [], []
        for block in model.blocks:
            # Читаем параметры напрямую БЕЗ exp()
            attn_bases.append(torch.clamp(block.attn.pmg_matrix.base, min=1.01).mean().item())
            attn_shifts.append(block.attn.pmg_matrix.shift.mean().item())
            
        step_time = time.time() - t0
        print(f"Шаг {step:03d}/{steps} | Loss: {loss.item():.4f} | Время шага: {step_time:.2f}с")
        print(f"   [Внимание PMG] Средний Base: {np.mean(attn_bases):.4f} | Средний Shift: {np.mean(attn_shifts):.4f}")

print(f"\n=== Обучение успешно завершено за {time.time() - t_start:.2f}с! ===")

# ==========================================
# 4. ГЕНЕРАЦИЯ
# ==========================================

print("\n=== ЗАПУСК ГЕНЕРАЦИИ ===")
model.eval()
prompt = "To be, or not to be, that is the"
gen_tokens = enc.encode_ordinary(prompt)
x_gen = torch.tensor(gen_tokens, dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    for _ in range(40):
        x_cond = x_gen[:, -seq_len:]
        logits, _ = model(x_cond)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        x_gen = torch.cat([x_gen, next_token_id], dim=-1)

print(f"\n>> {enc.decode(x_gen.tolist())} <<")
