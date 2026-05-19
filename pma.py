import torch
import torch.nn as nn
import numpy as np

# Ваш класс оставляем без изменений, он идеален
class ParametricMemoryGate(nn.Module):
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


class ParametricMemoryAttention(nn.Module):
    """Кастомный слой внимания на основе вашего PMG."""
    def __init__(self, embed_dim: int, initial_base: float = 4.0, initial_shift: float = -1.0):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Проекции для Запросов, Ключей и Значений
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Инициализируем ваш кастомный затвор
        self.pmg = ParametricMemoryGate(initial_base=initial_base, initial_shift=initial_shift)
        
        # Финальная проекция на выходе
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x имеет размерность: [Batch_Size, Sequence_Length, Embed_Dim]
        B, L, D = x.shape
        
        # 1. Получаем матрицы Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 2. Вычисляем сырые оценки взаимодействия (матричное умножение)
        # Масштабируем на sqrt(D) для стабильности, как в классическом внимании
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(D)
        
        # 3. Применяем ВАШУ формулу вместо Softmax!
        # Каждая связь теперь рассчитывается через параметрическое дерево
        attention_matrix = self.pmg(scores)
        
        # 4. Умножаем веса внимания на матрицу Значений (V)
        context = torch.matmul(attention_matrix, V)
        
        # 5. Пропускаем через финальный линейный слой
        return self.out_proj(context)

class MultiHeadParametricMemoryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, initial_base: float = 4.0, initial_shift: float = -1.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Проекции для всех голов сразу
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Индивидуальные обучаемые параметры base и shift для КАЖДОЙ головы
        if initial_base <= 1.0:
            raise ValueError("initial_base must be strictly greater than 1.0")
        raw_base_init = np.log(initial_base - 1.0)
        
        # Размерность [1, num_heads, 1, 1] для удобного вещания (broadcasting) на матрицы внимания
        self.raw_base = nn.Parameter(torch.full((1, num_heads, 1, 1), raw_base_init, dtype=torch.float32))
        self.shift = nn.Parameter(torch.full((1, num_heads, 1, 1), initial_shift, dtype=torch.float32))

    def _parametric_gate(self, x: torch.Tensor) -> torch.Tensor:
        # x имеет форму [Batch, num_heads, Seq_Len, Seq_Len]
        base = 1.0 + torch.exp(self.raw_base)  # Гарантируем base > 1 для каждой головы
        power = torch.clamp(x + self.shift, -20.0, 20.0)
        
        gate = (base ** power) / (1.0 + (base ** power))
        eps = 1e-7
        return torch.clamp(gate, eps, 1.0 - eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Входной тензор x: [Batch_Size, Seq_Len, Embed_Dim]
        B, L, D = x.shape
        
        # 1. Линейные проекции
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 2. Разделение на головы: [B, L, D] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Расчет сырых скалярных произведений для каждой головы
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # 4. Применение ВАШЕЙ формулы вместо Softmax
        # Благодаря размерности (1, num_heads, 1, 1) параметры применятся индивидуально к каждой голове
        attention_matrix = self._parametric_gate(scores)
        
        # 5. Взвешивание значений V
        context = torch.matmul(attention_matrix, V)
        
        # 6. Сборка голов обратно в единый вектор: [B, num_heads, L, head_dim] -> [B, L, D]
        context = context.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        
        # 7. Финальная проекция
        return self.out_proj(context)
