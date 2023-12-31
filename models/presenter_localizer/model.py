from pathlib import Path
from ultralytics import YOLO

model_path = Path(__file__).parent / "model.pt"
model = YOLO(model_path, task="detect")
