# AU Final Year - Object Detection Voice Assistant

Real-time object detection web app using YOLO + FastAPI with:

- multilingual voice feedback (`English`, `Hindi`, `Telugu`)
- direction guidance (`far left`, `left`, `center`, `right`, `far right`)
- conservative distance estimation (designed to avoid over-reporting distance)
- calibration APIs for better practical accuracy
- mobile access over local Wi-Fi

## Features

- Live camera object detection in browser
- Voice announcements for detected objects
- Language selector: English/Hindi/Telugu
- Direction and distance shown in UI and speech
- Server-side TTS fallback (`/api/tts`) when device voices are unavailable
- Focal length calibration endpoint
- Distance safety-factor tuning endpoint

## Project Structure

- `new1.py` - FastAPI server, YOLO detection, distance logic, calibration + TTS APIs
- `static/index.html` - Web UI, camera capture, websocket stream, multilingual speech
- `requirements.txt` - Python dependencies
- `yolov8s.pt` - YOLO model weights

## Requirements

- Python 3.10+ (recommended)
- Windows/macOS/Linux
- Webcam (laptop/USB/phone camera via browser permission)
- Internet needed for server TTS (`gTTS`) fallback

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe new1.py
```

Server runs on:

- Local: `http://localhost:8001`
- LAN: `http://<your-pc-ip>:8001`

If port 8001 is busy:

```powershell
.\.venv\Scripts\python.exe -m uvicorn new1:app --host 0.0.0.0 --port 8002
```

## Mobile Access

1. Connect mobile and PC to the same Wi-Fi.
2. Find PC IPv4 address (`ipconfig` on Windows).
3. Open `http://<pc-ip>:8001` (or `:8002`) on mobile browser.
4. Grant camera/audio permissions.
5. Start camera and choose language.

## API Endpoints

- `GET /` - Web app UI
- `GET /api/config` - Current runtime config
- `GET /api/calibrate` - Calibration guidance
- `POST /api/calibrate/focal` - Update focal length from known object measurement
- `POST /api/calibrate/safety-factor` - Tune conservative distance multiplier
- `GET /api/tts?lang=en|hi|te&text=...` - Server-side TTS audio stream
- `WS /ws` - Real-time frame processing websocket

## Calibration Tips

- Best accuracy comes from calibrating with objects and distances you actually use.
- Use a known object width and measured distance to set focal length.
- Keep the safety factor at or below `1.0` to remain conservative.

## Notes

- Monocular camera distance is an estimate, not a laser measurement.
- This project intentionally biases distance lower to reduce overestimation risk.

