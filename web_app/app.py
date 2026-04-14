import os
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
import timm
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import tempfile
import asyncio
from contextlib import asynccontextmanager

# Initialization of global variables for models with safe defaults
EFFNET = None
VIT_MODEL = None
IMG_SIZE = 224
NUM_FRAMES = 16
transform = None

async def initialize_models_background():
    global EFFNET, VIT_MODEL, IMG_SIZE, NUM_FRAMES, transform
    print("Initializing models in background...")
    try:
        EFFNET, VIT_MODEL, IMG_SIZE, NUM_FRAMES = load_models()
        # Initialize transformation after IMG_SIZE is loaded
        transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Models and transforms initialized successfully in background.")
    except Exception as e:
        print(f"CRITICAL: Failed to load models in background: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model loading in the background so the server can start immediately
    asyncio.create_task(initialize_models_background())
    yield
    print("Shutting down application...")

app = FastAPI(title="DeepFake Detection API", lifespan=lifespan)

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODELS_DIR = BASE_DIR / "models"
TEMP_DIR = Path(tempfile.gettempdir()) / "deepfake_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Architecture ---

class TemporalViT(nn.Module):
    def __init__(self, feature_dim=1280, d_model=256, n_heads=4, num_layers=4, num_classes=2, dropout=0.1, max_len=16):
        super().__init__()
        self.proj = nn.Linear(feature_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B, T, Fd = x.shape
        x = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :T + 1, :]
        h = self.encoder(x)
        cls_out = h[:, 0]
        cls_out = self.norm(cls_out)
        logits = self.fc(cls_out)
        return logits

# --- Helper Functions ---

def load_models():
    # Load metadata
    with open(MODELS_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    img_size = metadata.get("IMG_SIZE", 224)
    num_frames = metadata.get("NUM_FRAMES", 16)
    feature_dim = metadata.get("FEATURE_DIM", 1280)
    
    # Feature Extractor
    print("Loading feature extractor...")
    effnet = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, num_classes=0, global_pool="avg")
    effnet.eval().to(device)
    for p in effnet.parameters():
        p.requires_grad_(False)
    
    # Temporal ViT
    print("Loading Temporal ViT model...")
    vit_model = TemporalViT(feature_dim=feature_dim, max_len=num_frames).to(device)
    vit_model.load_state_dict(torch.load(MODELS_DIR / "best_temporal_vit.pth", map_location=device))
    vit_model.eval()
    
    return effnet, vit_model, img_size, num_frames

# The global variables are initialized in the lifespan context manager.

# Preprocessing transforms (initialized in lifespan)
# transform = T.Compose([...])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face_crop(img, size=224):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        # Fallback to center crop
        h, w = img.shape[:2]
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
        crop = img[y1:y1+side, x1:x1+side]
        return cv2.resize(crop, (size, size))

    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    face = img[y:y+h, x:x+w]
    return cv2.resize(face, (size, size))

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    
    indices = np.linspace(0, total-1, num_frames, dtype=int)
    faces = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok: continue
        face = detect_face_crop(frame, size=IMG_SIZE)
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        faces.append(transform(face_pil))
    
    cap.release()
    if len(faces) < num_frames:
        # Pad if not enough frames
        while len(faces) < num_frames:
            faces.append(faces[-1] if faces else torch.zeros(3, IMG_SIZE, IMG_SIZE))
    
    return torch.stack(faces[:num_frames])

# --- Endpoints ---

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/health")
async def health_check():
    status = "loading" if EFFNET is None else "ready"
    return {"status": status, "models_loaded": EFFNET is not None}

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    if EFFNET is None or VIT_MODEL is None or transform is None:
        raise HTTPException(status_code=503, detail="Models are still initializing. Please try again in a moment.")

    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Only video files (mp4, avi, mov) are supported.")
    
    # Save uploaded file temporarily
    temp_video_path = TEMP_DIR / file.filename
    with temp_video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Preprocess
        frames = extract_frames(temp_video_path, num_frames=NUM_FRAMES)
        if frames is None:
            raise HTTPException(status_code=500, detail="Could not process video frames.")
        
        # Inference
        with torch.no_grad():
            frames = frames.unsqueeze(0).to(device)  # (1, T, C, H, W)
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B*T, C, H, W)
            feats_flat = EFFNET(frames_flat)
            feats = feats_flat.view(B, T, -1)
            
            logits = VIT_MODEL(feats)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
            # Debugging
            print(f"DEBUG: Logits: {logits.cpu().numpy()}")
            print(f"DEBUG: Probs: {probs.cpu().numpy()}")
            print(f"DEBUG: Predicted Index: {pred_idx.item()}")
            
            classes = ["Fake", "Real"]
            result = classes[pred_idx.item()]
            score = confidence.item() * 100
            
        return {
            "prediction": result,
            "confidence": f"{score:.2f}%",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if temp_video_path.exists():
            temp_video_path.unlink()

if __name__ == "__main__":
    import uvicorn
    # Render provides the port via the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
