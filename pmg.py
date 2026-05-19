import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from parametric_memory_gate import ParametricMemoryGate  # ваш модуль


# ---------- генерация векторов символов ----------
def get_char_vector(char):
    code = ord(char)
    bits = [int(b) * 2 - 1 for b in format(code, '032b')]
    freq = code / 100.0
    extra = [np.sin(freq), np.cos(freq), np.sin(freq / 10), np.cos(freq / 10)]
    return torch.FloatTensor(bits + extra)


def generate_full_unicode_sample():
    """Репрезентативная выборка из Unicode."""
    codes = set()
    for i in range(0, 3000):
        codes.add(i)
    for i in range(0, 1000):
        codes.add(0x4E00 + i)
    for i in range(0x1F600, 0x1F64F):
        codes.add(i)
    return codes


# ---------- автоэнкодер с BatchNorm и увеличенным латентом ----------
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


# ---------- калибровка shift PMG ----------
def calibrate_pmg_shift(pmg_layer, input_tensor, target_mean=0.5, lr=0.01, steps=200):
    """Оптимизирует shift, чтобы средний выход PMG был близок к target_mean."""
    pmg_layer.raw_base.requires_grad_(False)
    opt = torch.optim.Adam([pmg_layer.shift], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        out = pmg_layer(input_tensor)
        loss = (out.mean() - target_mean).pow(2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            pmg_layer.shift.clamp_(-10.0, 10.0)

    pmg_layer.raw_base.requires_grad_(True)
    return pmg_layer.shift.item()


# ---------- подготовка данных ----------
codes = generate_full_unicode_sample()
X_list = [get_char_vector(chr(c)).numpy() for c in codes]
X_full = torch.FloatTensor(np.array(X_list))          # (N, 36)
print(f"Датасет: {X_full.shape}")

model = UnicodeAutoencoder(bits=32, emb_dim=8)

# ---------- собираем входы для каждого PMG ----------
model.eval()
with torch.no_grad():
    x = X_full
    # энкодер: Linear -> BN -> PMG1 -> Linear -> BN -> PMG2 -> Linear -> emb
    x = model.encoder[0](x)          # Linear
    x = model.encoder[1](x)          # BN
    input_pmg1 = x.clone()
    x = model.encoder[2](x)          # PMG1
    x = model.encoder[3](x)          # Linear
    x = model.encoder[4](x)          # BN
    input_pmg2 = x.clone()
    x = model.encoder[5](x)          # PMG2
    emb = model.encoder[6](x)        # Linear -> латент

    # декодер: Linear -> BN -> PMG3 -> Linear -> BN -> PMG4 -> Linear -> Tanh
    x = model.decoder[0](emb)        # Linear
    x = model.decoder[1](x)          # BN
    input_pmg3 = x.clone()
    x = model.decoder[2](x)          # PMG3
    x = model.decoder[3](x)          # Linear
    x = model.decoder[4](x)          # BN
    input_pmg4 = x.clone()
    x = model.decoder[5](x)          # PMG4
    # дальше Linear + Tanh

pmg_layers = model.get_pmg_layers()
inputs = [input_pmg1, input_pmg2, input_pmg3, input_pmg4]

print("\nКалибровка shift:")
for i, (pmg, inp) in enumerate(zip(pmg_layers, inputs)):
    shift_before = pmg.shift.item()
    new_shift = calibrate_pmg_shift(pmg, inp, target_mean=0.5, lr=0.01, steps=200)
    print(f"  PMG {i+1}: shift {shift_before:.4f} -> {new_shift:.4f}")

# ---------- проверка после калибровки ----------
print("\nСредние выходы после калибровки (BN в eval):")
model.eval()
with torch.no_grad():
    x = X_full
    for i, layer in enumerate(model.encoder):
        x = layer(x)
        if isinstance(layer, ParametricMemoryGate):
            print(f"  PMG enc {i}: mean={x.mean().item():.4f}")
    for i, layer in enumerate(model.decoder):
        x = layer(x)
        if isinstance(layer, ParametricMemoryGate):
            print(f"  PMG dec {i}: mean={x.mean().item():.4f}")

# ---------- обучение ----------
BATCH_SIZE = 128
EPOCHS = 500
LEARNING_RATE = 0.0005

codes_list = list(codes)
y_full = X_full[:, :32]
dataset = TensorDataset(X_full, y_full)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

model.train()
loss_history = []

print("\nОбучение:")
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    epoch_loss /= len(dataset)
    loss_history.append(epoch_loss)

    scheduler.step(epoch_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {epoch_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

# ---------- график ----------
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("UnicodeAutoencoder с ParametricMemoryGate + BatchNorm")
plt.grid(True)
plt.show()

# ---------- проверка на примерах ----------
model.eval()
with torch.no_grad():
    indices = np.random.choice(len(X_full), 5, replace=False)
    for idx in indices:
        x = X_full[idx:idx+1]
        true_bits = y_full[idx]
        pred_bits = model(x).squeeze()
        mse = torch.mean((pred_bits - true_bits) ** 2).item()
        print(f"\nСимвол U+{codes_list[idx]:04X}")
        print(f"  MSE: {mse:.6f}")
        print(f"  Истина (первые 8 бит): {true_bits[:8].numpy()}")
        print(f"  Предск (первые 8 бит): {pred_bits[:8].numpy().round(2)}")

# ---------- итоговые параметры PMG ----------
print("\nПараметры ParametricMemoryGate после обучения:")
for i, pmg in enumerate(pmg_layers, 1):
    base, shift = pmg.get_parameters()
    print(f"  PMG{i}: base={base:.4f}, shift={shift:.4f}")