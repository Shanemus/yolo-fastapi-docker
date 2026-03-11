# YOLOv8 Object Detection API

This project provides a **REST API for object detection using YOLOv8**, built with **FastAPI** and containerized using **Docker**.

The API supports two types of inference outputs:

* **Structured JSON results** (for developers and integrations)
* **Annotated images with bounding boxes** (for visualization and client applications)

The service is deployed in the cloud and accessible through an **interactive API documentation interface**.

---

# Features

* YOLOv8 object detection
* FastAPI REST API
* Image upload endpoint
* JSON detection output
* Annotated image output with bounding boxes
* Docker containerization
* Cloud deployment using Render

---

# Live Demo

Interactive API documentation:

```
https://yolo-fastapi-docker.onrender.com/docs
```

You can upload an image directly in the Swagger interface and test both endpoints.

---

# API Endpoints

## Health Check

`GET /health`

Returns:

```json
{
  "status": "ok"
}
```

---

# Object Detection (JSON Output)

`POST /predict`

Upload an image and receive structured detection results.

Example response:

```json
{
  "latency_seconds": 0.17,
  "num_detections": 1,
  "detections": [
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.91,
      "bbox_xyxy": [201, 473, 1009, 860]
    }
  ]
}
```

This endpoint is useful for:

* AI pipelines
* data processing systems
* machine learning integrations

---

# Object Detection (Annotated Image)

`POST /predict/image`

Upload an image and receive the **predicted image with bounding boxes and labels drawn on it**.

Response type:

```
image/jpeg
```

Example result:

```
Original image → YOLO detection → Annotated image returned
```

This endpoint is useful for:

* visualization
* demos
* user-facing AI applications
* computer vision dashboards

---

# Tech Stack

* Python
* FastAPI
* PyTorch (CPU)
* YOLOv8 (Ultralytics)
* OpenCV
* Docker
* Render (Cloud deployment)

---

# Run Locally

Clone the repository:

```bash
git clone https://github.com/shanemus/yolo-api-project.git
```

Navigate to the project folder:

```bash
cd yolo-api-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API server:

```bash
uvicorn app.main:app --reload
```

Open the documentation:

```
http://127.0.0.1:8000/docs
```

---

# Run with Docker

Build the Docker image:

```bash
docker build -t yolo-api .
```

Run the container:

```bash
docker run -p 8000:8000 yolo-api
```

Open:

```
http://127.0.0.1:8000/docs
```

---

# Example Use Cases

* AI inference APIs
* Object detection services
* Computer vision applications
* Machine learning deployment pipelines
* AI product prototypes

---

# Author

**Shan E Mustafa**

Computer Vision / Machine Learning Engineer


