from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
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

# --- ENDPOINT 1: JSON Output (For Developers) ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    logger.info("JSON Predict request received")

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
    return {"latency_seconds": latency, "num_detections": len(detections), "detections": detections}


# --- ENDPOINT 2: Image Output (For Clients/Portfolio) ---
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    start = time.time()
    logger.info("Image Predict request received")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        logger.error("Invalid image uploaded")
        return {"error": "Invalid image"}

    results = model(img)[0]

    # Draw bounding boxes and labels directly on the image
    for box in results.boxes:
        # OpenCV requires coordinates to be integers
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]

        # Draw a bright green rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add the text label above the box
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert the modified image back into a format we can send over the internet
    _, encoded_img = cv2.imencode('.jpg', img)
    image_bytes = encoded_img.tobytes()

    logger.info(f"Visual inference done in {time.time() - start:.3f}s")

    # Return as an actual image file, not JSON
    return Response(content=image_bytes, media_type="image/jpeg")