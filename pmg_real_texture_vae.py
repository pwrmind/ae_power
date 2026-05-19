import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from PIL import Image

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
# 2. ГЛУБОКАЯ АРХИТЕКТУРА VAE ДЛЯ РЕАЛЬНЫХ ДАННЫХ (64x64)
# ==========================================

class PMG_Deep_VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # --- ЭНКОДЕР (Вход: [B, 3, 64, 64]) ---
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # -> [B, 32, 32, 32]
        self.enc_act1  = ParametricMemoryGate(32)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # -> [B, 64, 16, 16]
        self.enc_act2  = ParametricMemoryGate(64)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)# -> [B, 128, 8, 8]
        self.enc_act3  = ParametricMemoryGate(128)
        
        # Проекция в латентное пространство вероятностей
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # --- ДЕКОДЕР ---
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # -> [B, 64, 16, 16]
        self.dec_act1  = ParametricMemoryGate(64)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # -> [B, 32, 32, 32]
        self.dec_act2  = ParametricMemoryGate(32)
        self.dec_conv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # -> [B, 16, 64, 64]
        self.dec_act3  = ParametricMemoryGate(16)
        
        self.dec_final = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def encode(self, x):
        x = self.enc_act1(self.enc_conv1(x))
        x = self.enc_act2(self.enc_conv2(x))
        x = x.view(x.size(0), -1) if hasattr(self, 'fc_mu') and self.fc_mu.in_features == 64 * 8 * 8 else self._forward_encoder_flat(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def _forward_encoder_flat(self, x):
        # Вспомогательный метод для обхода уплощения при изменении архитектуры
        x = self.enc_act3(self.enc_conv3(x))
        return x.view(x.size(0), -1)

    def encode_full(self, x):
        x = self.enc_act1(self.enc_conv1(x))
        x = self.enc_act2(self.enc_conv2(x))
        x = self.enc_act3(self.enc_conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.dec_act1(self.dec_conv1(x))
        x = self.dec_act2(self.dec_conv2(x))
        x = self.dec_act3(self.dec_conv3(x))
        return torch.sigmoid(self.dec_final(x))

    def forward(self, x):
        mu, logvar = self.encode_full(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ==========================================
# 3. ГЕНЕРАТОР РЕАЛИСТИЧНЫХ ТЕКСТУР ПРИРОДЫ
# ==========================================

def generate_perlin_noise_2d(size, res):
    def f(t): return 6*t**5 - 15*t**4 + 10*t**3
    delta = res / size
    grid = torch.stack(torch.meshgrid(torch.arange(0, res, delta), torch.arange(0, res, delta), indexing='ij'), dim=-1) % 1
    gradients = torch.randn((res + 1, res + 1, 2))
    gradients /= torch.norm(gradients, dim=-1, keepdim=True)
    step = int(size / res)
    g00 = gradients[0:-1, 0:-1].repeat_interleave(step, dim=0).repeat_interleave(step, dim=1)
    g10 = gradients[1:, 0:-1].repeat_interleave(step, dim=0).repeat_interleave(step, dim=1)
    g01 = gradients[0:-1, 1:].repeat_interleave(step, dim=0).repeat_interleave(step, dim=1)
    g11 = gradients[1:, 1:].repeat_interleave(step, dim=0).repeat_interleave(step, dim=1)
    n00 = torch.sum(g00 * grid, dim=-1)
    n10 = torch.sum(g10 * (grid - torch.tensor([1.0, 0.0])), dim=-1)
    n01 = torch.sum(g01 * (grid - torch.tensor([0.0, 1.0])), dim=-1)
    n11 = torch.sum(g11 * (grid - torch.tensor([1.0, 1.0])), dim=-1)
    t = f(grid)
    n0 = torch.lerp(n00, n10, t[:, :, 0])
    n1 = torch.lerp(n01, n11, t[:, :, 0])
    return torch.lerp(n0, n1, t[:, :, 1])

def generate_wood_bark(size=64):
    """Имитирует глубокие продольные трещины древесной коры."""
    # Вытягиваем шум по вертикали, сжимая по горизонтали
    noise = generate_perlin_noise_2d(size, 8)
    # Создаем эффект волокон через синусоидальное искажение координат
    y, x = np.ogrid[0:size, 0:size]
    warp = noise.numpy() * 4.0
    wood = np.sin((x + warp) * 0.4)
    # Переводим в диапазон 0-1 и красим в благородные коричневые оттенки коры
    wood = (wood - wood.min()) / (wood.max() - wood.min())
    rgb = np.zeros((3, size, size), dtype=np.float32)
    rgb[0] = wood * 0.45 + 0.15  # R
    rgb[1] = wood * 0.30 + 0.10  # G
    rgb[2] = wood * 0.15 + 0.05  # B
    return torch.tensor(rgb)

def generate_stone_texture(size=64):
    """Имитирует кристаллическую сколотую структуру гранита/камня."""
    # Мелкий высокочастотный шум
    noise1 = generate_perlin_noise_2d(size, 16)
    # Крупные тектонические трещины
    noise2 = generate_perlin_noise_2d(size, 4)
    stone = torch.abs(noise2) * 0.6 + noise1 * 0.4
    stone = (stone - stone.min()) / (stone.max() - stone.min())
    # Каменная серо-зеленая или холодная палитра
    rgb = np.zeros((3, size, size), dtype=np.float32)
    rgb[0] = stone.numpy() * 0.35 + 0.2  # R
    rgb[1] = stone.numpy() * 0.40 + 0.2  # G
    rgb[2] = stone.numpy() * 0.45 + 0.25 # B
    return torch.tensor(rgb)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Инициализация... Устройство: {device}")

print("Синтез физического датасета (50% Кора дерева, 50% Камень, разрешение 64x64)...")
num_images = 512
images = np.zeros((num_images, 3, 64, 64), dtype=np.float32)

for i in range(num_images):
    if i % 2 == 0:
        images[i] = generate_wood_bark(64).numpy()
    else:
        images[i] = generate_stone_texture(64).numpy()

dataset = torch.tensor(images).to(device)
print(f"Датасет готов! Всего высокоточных текстур природы: {dataset.shape[0]}")

def get_batch(data, batch_size):
    ix = torch.randint(len(data), (batch_size,))
    return data[ix]

# ==========================================
# 4. ЦИКЛ ОБУЧЕНИЯ НА РЕАЛЬНЫХ ДАННЫХ (300 ЭПОХ)
# ==========================================

latent_dim = 64
model = PMG_Deep_VAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

batch_size = 32
epochs = 300

# Создаем папки для сохранения результатов
os.makedirs("original_samples", exist_ok=True)
os.makedirs("generated_real_worlds", exist_ok=True)

# Сохраним образцы оригиналов для сравнения
Image.fromarray((images[0].transpose(1,2,0)*255).astype(np.uint8)).save("original_samples/original_wood.png")
Image.fromarray((images[1].transpose(1,2,0)*255).astype(np.uint8)).save("original_samples/original_stone.png")

print("\n=== СТАРТ СИНТЕЗА РЕАЛЬНЫХ МАТЕРИАЛЬНЫХ СТРУКТУР (300 ЭПОХ) ===")
model.train()

for epoch in range(epochs):
    t0 = time.time()
    num_batches = len(dataset) // batch_size
    epoch_loss = 0.0
    
    for _ in range(num_batches):
        batch_imgs = get_batch(dataset, batch_size)
        optimizer.zero_grad()
        
        reconstructed, mu, logvar = model(batch_imgs)
        recon_loss = F.mse_loss(reconstructed, batch_imgs, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = recon_loss + 0.05 * kld_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch + 1) % 30 == 0 or epoch == 0 or epoch == epochs - 1:
        all_bases = []
        for module in model.modules():
            if isinstance(module, ParametricMemoryGate):
                all_bases.append(module.base.mean().item())
        print(f"Эпоха {epoch+1:03d}/{epochs} | Real VAE Loss: {epoch_loss/num_batches:.2f} | Время: {time.time()-t0:.2f}с")
        print(f"   [PMG Материя] Средний параметр Base: {np.mean(all_bases):.4f}")

print("\n=== Физическое пространство вероятностей сформировано! ===")

# ==========================================
# 5. МАТЕРИАЛИЗАЦИЯ И ФИЗИЧЕСКОЕ СОХРАНЕНИЕ
# ==========================================
model.eval()

with torch.no_grad():
    print("\n=== ГЕНЕРАЦИЯ ФОТОГРАФИЙ МИРОВ ИЗ ЧИСТОГО ХАОСА ===")
    
    random_noise_vectors = torch.randn(16, latent_dim).to(device)
    generated_worlds = model.decode(random_noise_vectors)
    
    for idx in range(16):
        img_tensor = generated_worlds[idx].cpu()
        img_array = img_tensor.permute(1, 2, 0).numpy()
        img_array = (img_array * 255).astype(np.uint8)
        
        img = Image.fromarray(img_array, mode='RGB')
        # Апскейлим до 256x256 для детального изучения волокон и сколов
        img = img.resize((256, 256), resample=Image.Resampling.NEAREST)
        
        file_path = f"generated_real_worlds/nature_world_{idx+1:02d}.png"
        img.save(file_path)
        
    print(f"-> Успешно! 16 фотографий природы сохранены в папку: {os.path.abspath('generated_real_worlds')}")
    print(f"-> Образцы оригиналов для сравнения лежат в: {os.path.abspath('original_samples')}")
