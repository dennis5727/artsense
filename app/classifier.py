import json
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

CONFIDENCE_THRESHOLD = 0.80
MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL_PATH = MODEL_DIR / "efficientnet_b3_artsense.pth"
CLASS_INDEX_PATH = MODEL_DIR / "class_indices.json"

_model: torch.nn.Module | None = None
_idx_to_class: dict[int, str] = {}
_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_model():
    global _model, _idx_to_class
    if _model is not None:
        return

    with open(CLASS_INDEX_PATH) as f:
        class_to_idx: dict[str, int] = json.load(f)
    _idx_to_class = {v: k for k, v in class_to_idx.items()}

    num_classes = len(class_to_idx)
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    model.to(_device)
    model.eval()
    _model = model


def classify_painting(image: Image.Image) -> dict:
    """
    Classify a PIL image and return prediction results.

    Returns a dict with keys:
      - below_threshold (bool)
      - artist (str)         -- top-1 predicted class name
      - confidence (float)   -- top-1 softmax probability
      - top3 (list of (str, float)) -- top-3 predictions
    """
    _load_model()

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = _transform(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top3_probs, top3_idx = torch.topk(probs, 3)
    top3 = [(
        _idx_to_class[idx.item()].replace("_", " "),
        round(prob.item(), 4),
    ) for idx, prob in zip(top3_idx, top3_probs)]

    top_artist, top_conf = top3[0]
    return {
        "below_threshold": top_conf < CONFIDENCE_THRESHOLD,
        "artist": top_artist,
        "confidence": top_conf,
        "top3": top3,
    }
