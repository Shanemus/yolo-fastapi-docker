**YOLOv8 Object Detection API**

This project provides a REST API for object detection using YOLOv8, built with FastAPI and containerized using Docker.

The API accepts an image upload and returns structured JSON detection results, including:

1. detected object classes

2. confidence scores

3. bounding box coordinates

The service is deployed in the cloud and can be accessed through an interactive API documentation interface.

**Features**

YOLOv8 object detection

FastAPI REST API

Image upload endpoint

JSON detection output

Docker containerization

Cloud deployment using Render

**Live Demo**

Interactive API documentation:

https://yolo-fastapi-docker.onrender.com/docs

You can upload an image directly in the Swagger interface and receive detection results.

**API Endpoints**

1. Health Check
GET /health

Returns:

{
  "status": "ok"
}

2. Object Detection
POST /predict

Upload an image and receive detection results.

Example response:

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

**Tech Stack**

Python

FastAPI

PyTorch (CPU version)

YOLOv8 (Ultralytics)

OpenCV

Docker

Render (Cloud deployment)

**Run Locally**

Clone the repository:

git clone https://github.com/YOUR_USERNAME/yolo-api-project.git

Navigate to the project folder:

cd yolo-api-project

Install dependencies:

pip install -r requirements.txt

Run the API server:

uvicorn app.main:app --reload

Open the interactive documentation:

http://127.0.0.1:8000/docs
**Run with Docker**

Build the Docker image:

docker build -t yolo-api .

Run the container:

docker run -p 8000:8000 yolo-api

Open:

http://127.0.0.1:8000/docs
**Example Use Cases**

AI inference APIs

Object detection services

Computer vision applications

Machine learning deployment pipelines

Rapid prototyping for AI systems

Author

Shan E Mustafa
Computer Vision / Machine Learning Engineer



