import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import cv2
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
# 2. АРХИТЕКТУРА ЦВЕТНОГО PMG-VAE (RGB)
# ==========================================

class PMG_RGB_VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.enc_act1  = ParametricMemoryGate(32)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_act2  = ParametricMemoryGate(64)
        
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, 64 * 8 * 8)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_act1  = ParametricMemoryGate(32)
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dec_act2  = ParametricMemoryGate(16)
        
        self.dec_final = nn.Conv2d(16, 3, kernel_size=3, padding=1)

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
# 3. МАТЕМАТИЧЕСКАЯ МОДЕЛЬ ТУРБУЛЕНТНОСТИ
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

def generate_fbm_turbulence(size):
    noise1 = generate_perlin_noise_2d(size, 4)
    noise2 = generate_perlin_noise_2d(size, 8)
    noise3 = generate_perlin_noise_2d(size, 16)
    fbm = noise1 * 0.55 + noise2 * 0.30 + noise3 * 0.15
    return (fbm - fbm.min()) / (fbm.max() - fbm.min())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Инициализация... Устройство: {device}")

print("Сборка хроматического датасета...")
num_images = 512
images = np.zeros((num_images, 3, 32, 32), dtype=np.float32)
for i in range(num_images):
    images[i, 0] = generate_fbm_turbulence(32).numpy()
    images[i, 1] = generate_fbm_turbulence(32).numpy()
    images[i, 2] = generate_fbm_turbulence(32).numpy()

dataset = torch.tensor(images).to(device)

def get_batch(data, batch_size):
    ix = torch.randint(len(data), (batch_size,))
    return data[ix]

# ==========================================
# 4. МЕГА-ТРЕНИРОВКА МОДЕЛИ (500 ЭПОХ)
# ==========================================

latent_dim = 64
model = PMG_RGB_VAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

batch_size = 32
epochs = 500  # Увеличили в 3.3 раза для глубокой детализации!

print("\n=== СТАРТ ГЛУБОКОГО ЦВЕТНОГО СИНТЕЗА (500 ЭПОХ) ===")
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
        
    if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == epochs - 1:
        all_bases = []
        for module in model.modules():
            if isinstance(module, ParametricMemoryGate):
                all_bases.append(module.base.mean().item())
        print(f"Эпоха {epoch+1:03d}/{epochs} | VAE RGB Loss: {epoch_loss/num_batches:.2f} | Время: {time.time()-t0:.2f}с")
        print(f"   [PMG] Средний параметр Base: {np.mean(all_bases):.4f}")

print("\n=== Идеальное цветовое пространство сформировано! ===")

# ==========================================
# 5. СИНТЕЗ «ЖИВОГО» ВИДЕО-МИРА НА ДИСК
# ==========================================
model.eval()
video_filename = "living_fbm_world.mp4"
print(f"\n=== РЕНДЕРИНГ ВИДЕО КИПЕНИЯ ВСЕЛЕННОЙ В ФАЙЛ '{video_filename}' ===")

# Настройки видео: 120 кадров, разрешение 256х256 пикселей, 30 кадров в секунду
num_frames = 120
frame_size = (256, 256)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_filename, fourcc, 30, frame_size)

# Выбираем два случайных латентных полюса (вектора А и Б) в пространстве вероятностей
z_start = torch.randn(1, latent_dim).to(device)
z_end = torch.randn(1, latent_dim).to(device)

with torch.no_grad():
    for f in range(num_frames):
        # Циклическая интерполяция по синусоиде для создания бесшовного и плавного покачивания
        alpha = (np.sin(np.pi * 2 * f / num_frames) + 1.0) / 2.0
        z_current = z_start * (1.0 - alpha) + z_end * alpha
        
        # Декодируем текущий миг времени через вашу PMG-активацию
        frame_tensor = model.decode(z_current)[0].cpu()
        frame_array = frame_tensor.permute(1, 2, 0).numpy()
        frame_array = (frame_array * 255).astype(np.uint8)
        
        # Апскейлим кадр до 256х256 пикселей
        img_pil = Image.fromarray(frame_array, mode='RGB')
        img_pil = img_pil.resize(frame_size, resample=Image.Resampling.NEAREST)
        
        # OpenCV работает в формате BGR, переводим цвета из RGB в BGR
        open_cv_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Записываем кадр в видеофайл
        video_writer.write(open_cv_frame)

video_writer.release()
print(f"-> Пиратский триумф! Видеофайл успешно собран и сохранен: {os.path.abspath(video_filename)}")
