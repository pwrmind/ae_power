import torch
import torch.nn as nn
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DIST = 5
INPUT_FILE = "input.txt" # The text file you used for training

# 1. Unicode Autoencoder (GELU Architecture)
class UnicodeAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, 8), nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.GELU(),
            nn.Linear(16, 32), nn.Sigmoid()
        )
    def forward(self, x): return self.decoder(self.encoder(x))
    def get_embedding(self, x): return self.encoder(x)

# 2. Triplet Predictor (256-128-8 Architecture)
class TripletPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# Initialize and Load
ae = UnicodeAutoencoder().to(DEVICE)
ae.load_state_dict(torch.load("unicode_ae_8bit.pth", weights_only=True), strict=False)
ae.eval()

predictor = TripletPredictor().to(DEVICE)
predictor.load_state_dict(torch.load("predictor_weights.pth", weights_only=True))
predictor.eval()

@torch.no_grad()
def generate(seed_text, length=40):
    current_text = seed_text.upper()
    
    # --- BUILD VECTOR MAP (The Alphabet from your training text) ---
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Using basic alphabet.")
        chars = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ ")
    else:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            chars = sorted(list(set(f.read().upper())))
    
    # Pre-calculate 8-bit embeddings for every known character
    char_map = {}
    for c in chars:
        bits = torch.FloatTensor([int(b) for b in bin(ord(c))[2:].zfill(32)]).to(DEVICE).unsqueeze(0)
        char_map[c] = ae.get_embedding(bits).squeeze(0)

    print(f"Seed: {current_text} | Result: ", end="", flush=True)

    for _ in range(length):
        # Prepare current character embedding
        v_curr_bits = torch.FloatTensor([int(b) for b in bin(ord(current_text[-1]))[2:].zfill(32)]).to(DEVICE).unsqueeze(0)
        v_curr = ae.get_embedding(v_curr_bits)
        
        total_opinion = torch.zeros((1, 8)).to(DEVICE)
        active_w = 0
        
        # Consult experts at different distances
        for d in range(1, MAX_DIST + 1):
            if len(current_text) - d >= 0:
                past_char = current_text[-d]
                v_past_bits = torch.FloatTensor([int(b) for b in bin(ord(past_char))[2:].zfill(32)]).to(DEVICE).unsqueeze(0)
                v_past = ae.get_embedding(v_past_bits)
                
                dist_val = torch.tensor([[d / MAX_DIST]]).to(DEVICE)
                
                # Predict next 8-bit vector
                pred_v = predictor(torch.cat((v_curr, v_past, dist_val), dim=1))
                
                # Weight decay for experts further away
                w = 1 / (d ** 1.2)
                total_opinion += pred_v * w
                active_w += w
        
        avg_v = (total_opinion / active_w).squeeze(0)
        
        # --- NEAREST NEIGHBOR SEARCH (Replacing direct binary decoding) ---
        best_char = "?"
        min_dist = float('inf')
        
        for char, vec in char_map.items():
            # Euclidean distance in 8-bit space
            dist = torch.norm(avg_v - vec).item()
            if dist < min_dist:
                min_dist = dist
                best_char = char
        
        current_text += best_char
        print(best_char, end="", flush=True)

print("\n--- Start Generation ---")
generate("направ", length=40)
