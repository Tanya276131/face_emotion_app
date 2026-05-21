from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from fastapi import FastAPI, File, UploadFile

from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Add the CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)







# Load your custom trained model
# Ensure 'best.pt' is in the same directory as this script
model = YOLO("facial_emotion_best.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read image from request
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform inference
    results = model.predict(image)

    # Extract bounding box data
    # detections[0] refers to the first image in the batch
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    # Format output
    output = []
    for box, score, cls in zip(boxes, scores, classes):
        output.append({
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
            "confidence": float(score),
            "class": int(cls)
        })
        
    return {"detections": output}

    
