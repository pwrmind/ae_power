import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# ==========================================
# 1. СТАБИЛЬНЫЙ ФУНДАМЕНТ: ВАША МАТЕМАТИКА PMG
# ==========================================

class ParametricMemoryGate(nn.Module):
    def __init__(self, channels: int, initial_base: float = 4.0, initial_shift: float = 0.5):
        super().__init__()
        self.base = nn.Parameter(torch.full((1, channels, 1, 1), initial_base, dtype=torch.float32))
        self.shift = nn.Parameter(torch.full((1, channels, 1, 1), initial_shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.to(torch.float32)
        safe_base = torch.clamp(self.base, min=1.01)
        power = x_f32 + self.shift
        log_base = torch.log(safe_base)
        logits = power * log_base
        gate = torch.sigmoid(logits)
        return torch.clamp(gate, 1e-6, 1.0 - 1e-6).to(x.dtype)

# ==========================================
# 2. АРХИТЕКТУРА ВАРИАЦИОННОГО PMG-АВТОЭНКОДЕРА
# ==========================================

class PMG_VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # --- ЭНКОДЕР ---
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # -> [B, 32, 16, 16]
        self.enc_act1  = ParametricMemoryGate(32)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # -> [B, 64, 8, 8]
        self.enc_act2  = ParametricMemoryGate(64)
        
        # Проекция в пространство вероятностей
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        
        # --- ДЕКОДЕР ---
        self.decoder_input = nn.Linear(latent_dim, 64 * 8 * 8)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # -> [B, 32, 16, 16]
        self.dec_act1  = ParametricMemoryGate(32)
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) # -> [B, 16, 16, 16]
        self.dec_act2  = ParametricMemoryGate(16)
        
        self.dec_final = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def encode(self, x):
        x = self.enc_act1(self.enc_conv1(x))
        x = self.enc_act2(self.enc_conv2(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 64, 8, 8)
        x = self.dec_act1(self.dec_conv1(x))
        x = self.dec_act2(self.dec_conv2(x))
        return torch.sigmoid(self.dec_final(x))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ==========================================
# 3. ИСПРАВЛЕННАЯ ГЕНЕРАЦИЯ ШУМА ПЕРЛИНА В ОЗУ
# ==========================================

def generate_perlin_noise_2d(shape_tuple, res_tuple):
    """Исправленный генератор природного шума с явным разделением осей."""
    shape_h, shape_w = shape_tuple
    res_h, res_w = res_tuple
    
    def f(t): return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res_h / shape_h, res_w / shape_w)
    grid = torch.stack(torch.meshgrid(
        torch.arange(0, res_h, delta[0]), 
        torch.arange(0, res_w, delta[1]), 
        indexing='ij'
    ), dim=-1) % 1
    
    gradients = torch.randn((res_h + 1, res_w + 1, 2))
    gradients /= torch.norm(gradients, dim=-1, keepdim=True)
    
    # Исправленное зацикливание шагов интерполяции через целые числа
    step_h, step_w = int(shape_h / res_h), int(shape_w / res_w)
    g00 = gradients[0:-1, 0:-1].repeat_interleave(step_h, dim=0).repeat_interleave(step_w, dim=1)
    g10 = gradients[1:, 0:-1].repeat_interleave(step_h, dim=0).repeat_interleave(step_w, dim=1)
    g01 = gradients[0:-1, 1:].repeat_interleave(step_h, dim=0).repeat_interleave(step_w, dim=1)
    g11 = gradients[1:, 1:].repeat_interleave(step_h, dim=0).repeat_interleave(step_w, dim=1)
    
    n00 = torch.sum(g00 * grid, dim=-1)
    n10 = torch.sum(g10 * (grid - torch.tensor([1.0, 0.0])), dim=-1)
    n01 = torch.sum(g01 * (grid - torch.tensor([0.0, 1.0])), dim=-1)
    n11 = torch.sum(g11 * (grid - torch.tensor([1.0, 1.0])), dim=-1)
    
    t = f(grid)
    n0 = torch.lerp(n00, n10, t[:, :, 0])
    n1 = torch.lerp(n01, n11, t[:, :, 0])
    return torch.lerp(n0, n1, t[:, :, 1])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Инициализация... Устройство: {device}")

print("Сборка фрактального датасета (Имитация живых миров)...")
num_images = 512
images = np.zeros((num_images, 1, 32, 32), dtype=np.float32)

for i in range(num_images):
    # Корректно передаем пары значений
    noise1 = generate_perlin_noise_2d((32, 32), (4, 4))
    noise2 = generate_perlin_noise_2d((32, 32), (8, 8))
    combined_noise = noise1 * 0.7 + noise2 * 0.3
    combined_noise = (combined_noise - combined_noise.min()) / (combined_noise.max() - combined_noise.min())
    images[i, 0] = combined_noise.numpy()

dataset = torch.tensor(images).to(device)
print(f"Датасет готов! Форма: {dataset.shape}")

def get_batch(data, batch_size):
    ix = torch.randint(len(data), (batch_size,))
    return data[ix]

# ==========================================
# 4. ТРЕНИРОВКА ВАРИАЦИОННОЙ МОДЕЛИ (120 ЭПОХ)
# ==========================================

latent_dim = 16
model = PMG_VAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

batch_size = 32
epochs = 120

print("\n=== СТАРТ СИНТЕЗА ВЕРОЯТНОСТНЫХ ПОЛЕЙ ===")
model.train()

for epoch in range(epochs):
    t0 = time.time()
    num_batches = len(dataset) // batch_size
    epoch_loss = 0.0
    
    for _ in range(num_batches):
        batch_imgs = get_batch(dataset, batch_size)
        optimizer.zero_grad()
        
        reconstructed, mu, logvar = model(batch_imgs)
        
        # Считаем суммарную MSE по батчу пикселей
        recon_loss = F.mse_loss(reconstructed, batch_imgs, reduction='sum')
        # KLD регуляризатор
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = recon_loss + 0.1 * kld_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch + 1) % 15 == 0 or epoch == 0:
        all_bases = []
        for module in model.modules():
            if isinstance(module, ParametricMemoryGate):
                all_bases.append(module.base.mean().item())
        print(f"Эпоха {epoch+1:03d}/120 | VAE Total Loss: {epoch_loss/num_batches:.2f} | Время: {time.time()-t0:.2f}с")
        print(f"   [PMG Вселенная] Средний параметр Base: {np.mean(all_bases):.4f}")

print("\n=== Пространство вероятностей сформировано! ===")

# ==========================================
# 5. ГЕНЕРАЦИЯ СОВЕРШЕННО НОВОГО МИРА ИЗ ХАОСА
# ==========================================
model.eval()
with torch.no_grad():
    print("\n=== ЗАПУСК ГЕНЕРАЦИИ НОВОГО МИРА ИЗ ПУСТОТЫ ===")
    # 16 случайных нормально распределенных чисел (чистый хаос векторов)
    random_noise_vector = torch.randn(1, latent_dim).to(device)
    
    # Декодируем хаос через вашу PMG формулу
    generated_world = model.decode(random_noise_vector)
    
    print("\n=== МАТРИЦА СГЕНЕРИРОВАННОЙ ТЕКСТУРЫ (Срез 8x8 из никогда не существовавшей картинки) ===")
    print(np.round(generated_world[0, 0, 12:20, 12:20].cpu().numpy(), 2))
