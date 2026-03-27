# main.py
import cv2
import time
import asyncio
import base64
import json
import io
from collections import deque
from typing import Dict, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from ultralytics import YOLO
import uvicorn
from pydantic import BaseModel, Field
from gtts import gTTS

# ============ CONFIGURATION ============

class Config:
    # Use a more accurate model (trade-off: slower)
    # Options: yolov8n.pt (fast), yolov8s.pt (balanced), yolov8m.pt (accurate)
    MODEL_PATH = "yolov8s.pt"
    
    # Detection confidence threshold (higher = fewer false positives)
    CONFIDENCE_THRESHOLD = 0.5
    
    # Camera calibration - MUST be calibrated for your specific camera
    # To calibrate: measure a known object at known distance, calculate:
    # FOCAL_LENGTH = (pixel_width * actual_distance) / real_width
    FOCAL_LENGTH = 650
    
    # Real-world object widths in meters (expanded list)
    REAL_WIDTHS = {
        # People
        "person": 0.45,
        
        # Electronics
        "laptop": 0.35,
        "tv": 1.1,
        "cell phone": 0.07,
        "keyboard": 0.45,
        "mouse": 0.06,
        "remote": 0.05,
        "monitor": 0.5,
        
        # Furniture
        "chair": 0.5,
        "couch": 2.0,
        "bed": 1.5,
        "dining table": 1.2,
        "desk": 1.2,
        
        # Kitchen items
        "bottle": 0.07,
        "cup": 0.08,
        "bowl": 0.15,
        "fork": 0.02,
        "knife": 0.025,
        "spoon": 0.04,
        "wine glass": 0.07,
        "microwave": 0.5,
        "oven": 0.6,
        "refrigerator": 0.9,
        "toaster": 0.3,
        
        # Vehicles
        "car": 1.8,
        "motorcycle": 0.8,
        "bicycle": 0.6,
        "bus": 2.5,
        "truck": 2.5,
        
        # Animals
        "dog": 0.5,
        "cat": 0.3,
        "bird": 0.15,
        
        # Other common objects
        "backpack": 0.35,
        "umbrella": 1.0,
        "handbag": 0.3,
        "suitcase": 0.5,
        "book": 0.2,
        "clock": 0.3,
        "vase": 0.15,
        "scissors": 0.08,
        "teddy bear": 0.3,
        "potted plant": 0.3,
    }
    
    # Speech settings
    SPEAK_DELAY = 3.0  # seconds between announcements for same object
    DISTANCE_SAFETY_FACTOR = 0.9  # Bias lower to avoid over-reporting distance
    
    # Processing settings
    FRAME_SKIP = 2  # Process every Nth frame for better performance
    MAX_DETECTIONS = 10  # Limit detections per frame


# ============ IMPROVED DISTANCE ESTIMATION ============
class DistanceEstimator:
    """
    Improved distance estimation with smoothing and validation.
    """
    
    def __init__(self, focal_length: float, real_widths: Dict[str, float], safety_factor: float = 0.9):
        self.focal_length = focal_length
        self.real_widths = real_widths
        self.safety_factor = safety_factor
        # Keep history for smoothing
        self.distance_history: Dict[str, deque] = {}
        self.history_size = 5
    
    def estimate(self, label: str, pixel_width: int, pixel_height: int, 
                 box_area: int, frame_area: int) -> Optional[float]:
        """
        Estimate distance using multiple heuristics for better accuracy.
        """
        if label not in self.real_widths:
            return None
        
        if pixel_width <= 0:
            return None
        
        real_width = self.real_widths[label]
        
        # Primary method: width-based estimation
        distance_by_width = (real_width * self.focal_length) / pixel_width
        
        # Secondary validation: check if distance is reasonable
        # Objects shouldn't be closer than 0.3m or farther than 20m typically
        if distance_by_width < 0.3:
            distance_by_width = 0.3
        elif distance_by_width > 20:
            distance_by_width = 20
        
        # Apply temporal smoothing
        if label not in self.distance_history:
            self.distance_history[label] = deque(maxlen=self.history_size)
        
        self.distance_history[label].append(distance_by_width)
        
        # Conservative estimate:
        # use a lower-biased value so we avoid showing more than actual distance.
        history = self.distance_history[label]
        smoothed = sum(history) / len(history)
        conservative = min(smoothed, min(history)) * self.safety_factor

        # Keep a sensible floor while still being conservative.
        conservative = max(0.2, conservative)

        return round(conservative, 2)
    
    def calibrate(self, label: str, pixel_width: int, actual_distance: float) -> float:
        """
        Helper method to calibrate focal length using a known object.
        Place an object of known size at a measured distance and call this.
        """
        if label not in self.real_widths:
            raise ValueError(f"Unknown object: {label}")
        
        real_width = self.real_widths[label]
        new_focal_length = (pixel_width * actual_distance) / real_width
        
        return new_focal_length

    def set_focal_length(self, focal_length: float) -> None:
        self.focal_length = focal_length
        self.distance_history.clear()

    def set_safety_factor(self, safety_factor: float) -> None:
        self.safety_factor = safety_factor
        self.distance_history.clear()


# ============ DETECTION PROCESSOR ============
class DetectionProcessor:
    """
    Handles YOLO detection with improved accuracy settings.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = YOLO(config.MODEL_PATH)
        self.distance_estimator = DistanceEstimator(
            config.FOCAL_LENGTH,
            config.REAL_WIDTHS,
            config.DISTANCE_SAFETY_FACTOR
        )
        self.last_spoken: Dict[str, float] = {}
        self.frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Process a frame and return annotated frame + detection data.
        """
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % self.config.FRAME_SKIP != 0:
            return frame, []
        
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width
        current_time = time.time()
        
        # Run detection with higher confidence threshold
        results = self.model(
            frame,
            conf=self.config.CONFIDENCE_THRESHOLD,
            verbose=False
        )
        
        detections = []
        
        for r in results:
            boxes = r.boxes
            
            # Sort by confidence and limit detections
            if len(boxes) > self.config.MAX_DETECTIONS:
                confidences = boxes.conf.cpu().numpy()
                top_indices = np.argsort(confidences)[-self.config.MAX_DETECTIONS:]
                boxes = [boxes[i] for i in top_indices]
            
            for box in boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.model.names[cls]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pixel_width = x2 - x1
                pixel_height = y2 - y1
                box_area = pixel_width * pixel_height
                
                # Calculate center for direction
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Determine direction with finer granularity
                if center_x < frame_width * 0.2:
                    direction = "far left"
                elif center_x < frame_width * 0.4:
                    direction = "left"
                elif center_x > frame_width * 0.8:
                    direction = "far right"
                elif center_x > frame_width * 0.6:
                    direction = "right"
                else:
                    direction = "center"
                
                # Estimate distance
                distance = self.distance_estimator.estimate(
                    label, pixel_width, pixel_height, box_area, frame_area
                )
                
                # Draw bounding box with color based on distance
                if distance:
                    if distance < 1.5:
                        color = (0, 0, 255)  # Red - close
                    elif distance < 3:
                        color = (0, 165, 255)  # Orange - medium
                    else:
                        color = (0, 255, 0)  # Green - far
                else:
                    color = (255, 255, 0)  # Cyan - unknown
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Create label text
                if distance:
                    text = f"{label} {distance}m ({direction})"
                else:
                    text = f"{label} ({direction})"
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame, 
                    (x1, y1 - text_height - 10), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                cv2.putText(
                    frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                
                # Check if we should announce this detection
                key = f"{label}_{direction}"
                should_speak = (
                    key not in self.last_spoken or 
                    current_time - self.last_spoken[key] > self.config.SPEAK_DELAY
                )
                
                detection_data = {
                    "label": label,
                    "confidence": round(confidence, 2),
                    "direction": direction,
                    "distance": distance,
                    "bbox": [x1, y1, x2, y2],
                    "should_speak": should_speak
                }
                
                if should_speak:
                    self.last_spoken[key] = current_time
                
                detections.append(detection_data)
        
        return frame, detections

    def calibrate_and_apply(self, label: str, pixel_width: int, actual_distance: float) -> float:
        new_focal_length = self.distance_estimator.calibrate(label, pixel_width, actual_distance)
        self.distance_estimator.set_focal_length(new_focal_length)
        self.config.FOCAL_LENGTH = round(new_focal_length, 2)
        return self.config.FOCAL_LENGTH

    def apply_safety_factor(self, safety_factor: float) -> float:
        self.distance_estimator.set_safety_factor(safety_factor)
        self.config.DISTANCE_SAFETY_FACTOR = safety_factor
        return self.config.DISTANCE_SAFETY_FACTOR


# ============ FASTAPI APPLICATION ============

# Global processor instance
processor: Optional[DetectionProcessor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor
    print("Loading YOLO model...")
    processor = DetectionProcessor(Config())
    print("Model loaded successfully!")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Object Detection Voice Assistant",
    description="Real-time object detection with voice feedback",
    lifespan=lifespan
)

class FocalCalibrationRequest(BaseModel):
    label: str
    pixel_width: int = Field(gt=0)
    actual_distance: float = Field(gt=0)

class SafetyFactorRequest(BaseModel):
    safety_factor: float = Field(ge=0.5, le=1.0)


LANG_TO_GTTS = {
    "en": "en",
    "hi": "hi",
    "te": "te",
}


class ConnectionManager:
    """Manages WebSocket connections."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video processing.
    Client sends base64 encoded frames, server responds with detections.
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                # Decode base64 image
                img_data = base64.b64decode(message["data"])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None and processor is not None:
                    # Process frame
                    annotated_frame, detections = processor.process_frame(frame)
                    
                    # Encode result frame
                    _, buffer = cv2.imencode('.jpg', annotated_frame, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send response
                    await websocket.send_json({
                        "type": "result",
                        "frame": frame_base64,
                        "detections": detections
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
@app.get("/api/calibrate")
async def calibrate_info():
    """
    Returns calibration instructions.
    """
    return {
        "instructions": [
            "1. Place a known object (e.g., laptop) at a measured distance (e.g., 1 meter)",
            "2. Run detection and note the pixel width of the object",
            "3. Calculate: FOCAL_LENGTH = (pixel_width * distance) / real_width",
            "4. Update FOCAL_LENGTH in Config class"
        ],
        "current_focal_length": Config.FOCAL_LENGTH,
        "known_objects": Config.REAL_WIDTHS
    }


@app.get("/api/config")
async def get_config():
    """Returns current configuration."""
    return {
        "model": Config.MODEL_PATH,
        "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
        "focal_length": Config.FOCAL_LENGTH,
        "distance_safety_factor": Config.DISTANCE_SAFETY_FACTOR,
        "speak_delay": Config.SPEAK_DELAY,
        "known_objects": list(Config.REAL_WIDTHS.keys())
    }

@app.post("/api/calibrate/focal")
async def calibrate_focal(payload: FocalCalibrationRequest):
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if payload.label not in Config.REAL_WIDTHS:
        raise HTTPException(status_code=400, detail="Unknown object label for calibration")

    try:
        updated_focal = processor.calibrate_and_apply(
            payload.label, payload.pixel_width, payload.actual_distance
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Focal length calibrated and applied",
        "label": payload.label,
        "updated_focal_length": updated_focal,
        "distance_safety_factor": Config.DISTANCE_SAFETY_FACTOR,
    }


@app.post("/api/calibrate/safety-factor")
async def calibrate_safety(payload: SafetyFactorRequest):
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    updated = processor.apply_safety_factor(payload.safety_factor)
    return {
        "message": "Distance safety factor updated",
        "distance_safety_factor": updated,
    }


@app.get("/api/tts")
async def tts(text: str, lang: str = "en"):
    normalized_lang = lang.lower()
    gtts_lang = LANG_TO_GTTS.get(normalized_lang)
    if gtts_lang is None:
        raise HTTPException(status_code=400, detail="Unsupported language. Use en, hi, te.")

    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        audio_buffer = io.BytesIO()
        tts_engine = gTTS(text=clean_text, lang=gtts_lang)
        tts_engine.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc

    return StreamingResponse(audio_buffer, media_type="audio/mpeg")


# ============ ENTRY POINT ============

if __name__ == "__main__":
    import os
    os.makedirs("static", exist_ok=True)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False
    )