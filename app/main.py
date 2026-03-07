from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
import cv2
import time
from app.logger import setup_logger

app = FastAPI(title="YOLOv8 Inference API")
logger = setup_logger()

model = YOLO("yolov8n.pt")

@app.get("/health")
def health():
    logger.info("Health check called")
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    logger.info("Predict request received")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        logger.error("Invalid image uploaded")
        return {"error": "Invalid image"}

    results = model(img)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]

        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2]
        })

    latency = time.time() - start
    logger.info(f"Inference done in {latency:.3f}s")

    return {
        "latency_seconds": latency,
        "num_detections": len(detections),
        "detections": detections
    }