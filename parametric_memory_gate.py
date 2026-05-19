import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class ParametricMemoryGate(nn.Module):
    """
    Implementation of the Parametric Memory Gate (PMG) activation function.
    
    Formula:
        f(x) = (base^(x + shift)) / (1 + base^(x + shift))
        
    Mathematical Properties:
        - Strictly bounded interval: (0, 1), making it ideal for gating mechanisms, 
          attention filters, and binary classification outputs.
        - Asymmetric noise suppression: Rapidly decays to near-zero values on the 
          negative tail, filtering out background noise significantly faster than a 
          standard Sigmoid function.
        - Trainable geometry: Both 'base' and 'shift' are learned parameters, 
          allowing the network to dynamically adjust the slope and threshold of the gate.
          
    Security & Stability:
        - The 'base' parameter is optimized in log-space (raw_base) to mathematically 
          guarantee that base > 1.0 at all times, preventing division by zero.
        - Internal values are bounded via clamping to prevent Exploding Gradients 
          and OverflowErrors during rapid backpropagation steps.
    """
    def __init__(self, initial_base: float = 4.0, initial_shift: float = -1.0):
        """
        Args:
            initial_base (float): Starting base value for the exponent. Must be > 1.0. Defaults to 4.0.
            initial_shift (float): Starting horizontal shift value. Defaults to -1.0.
        """
        super().__init__()
        if initial_base <= 1.0:
            raise ValueError("initial_base must be strictly greater than 1.0")
            
        # Optimization in log-space ensures stability: base = 1.0 + exp(raw_base)
        raw_base_init = np.log(initial_base - 1.0)
        self.raw_base = nn.Parameter(torch.tensor([raw_base_init], dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor([initial_shift], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Parametric Memory Gate.
        """
        base = 1.0 + torch.exp(self.raw_base)
        
        # Guard against overflow/underflow under extreme input values
        power = torch.clamp(x + self.shift, -20.0, 20.0)
        
        # Core mathematical formulation
        gate = (base ** power) / (1.0 + (base ** power))
        
        # Защита от аппаратного округления до абсолютных 0.0 или 1.0
        eps = 1e-7
        gate = torch.clamp(gate, eps, 1.0 - eps)
        
        return gate

    def get_parameters(self) -> Tuple[float, float]:
        """
        Helper method to extract the human-readable learned parameters.
        
        Returns:
            Tuple[float, float]: (actual_base, actual_shift)
        """
        with torch.no_grad():
            actual_base = 1.0 + torch.exp(self.raw_base).item()
            actual_shift = self.shift.item()
            return actual_base, actual_shift

    def __repr__(self) -> str:
        """
        Official string representation for PyTorch model printing.
        """
        try:
            base, shift = self.get_parameters()
            return f"ParametricMemoryGate(learned_base={base:.4f}, learned_shift={shift:.4f})"
        except Exception:
            return "ParametricMemoryGate()"

class ParametricMemoryGateNP:
    def __init__(self, base: float = 5.0, shift: float = -0.1):
        # Эмулируем параметры, которые были обучены в PyTorch
        self.base = base
        self.shift = shift
        self.eps = 1e-7

    def forward_step(self, x: float) -> float:
        """
        Сверхбыстрый пошаговый инференс для реального времени (внутри цикла полетного контроллера)
        """
        # Ограничение экспоненты для предотвращения OverflowError в оригинальном стиле
        power = np.clip(x + self.shift, -20.0, 20.0)
        
        # Основная формула PMG
        base_pow = self.base ** power
        gate = base_pow / (1.0 + base_pow)
        
        # Защита от аппаратного округления
        return np.clip(gate, self.eps, 1.0 - self.eps)

class UnicodeAutoencoder(nn.Module):
    def __init__(self, bits=32, emb_dim=16):
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
            nn.Linear(32, emb_dim)      # латент без активации
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
# class DifferentialMemoryGate(nn.Module):
#     def __init__(self, initial_base: float = 4.0, initial_shift: float = -1.0):
#         super().__init__()
#         if initial_base <= 1.0:
#             raise ValueError("initial_base must be strictly greater than 1.0")
            
#         raw_base_init = np.log(initial_base - 1.0)
#         self.raw_base = nn.Parameter(torch.tensor([raw_base_init], dtype=torch.float32))
#         self.shift = nn.Parameter(torch.tensor([initial_shift], dtype=torch.float32))
        
#         # Внутреннее состояние памяти для хранения предыдущего шага
#         self.register_buffer('x_prev', torch.tensor([0.0], dtype=torch.float32))

#     def reset(self):
#         """Сброс памяти при старте новой сессии телеметрии"""
#         self.x_prev.fill_(0.0)

#     def forward(self, x_t: torch.Tensor) -> torch.Tensor:
#         """
#         Пошаговое или последовательное выполнение.
#         Автоматически рассчитывает дельту относительно x_prev сохраненного внутри.
#         """
#         base = 1.0 + torch.exp(self.raw_base)
        
#         # Встроенный расчет дифференциала скорости изменения
#         delta_x = x_t - self.x_prev
        
#         # Обновляем внутреннюю память текущим значением
#         # detach() критически важен, чтобы граф вычислений не рос бесконечно во времени
#         self.x_prev.copy_(x_t.detach())
        
#         power = torch.clamp(delta_x + self.shift, -20.0, 20.0)
#         gate = (base ** power) / (1.0 + (base ** power))
        
#         eps = 1e-7
#         return torch.clamp(gate, eps, 1.0 - eps)
