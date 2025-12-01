# Backend (Room Monitoring)

Run the FastAPI backend which exposes endpoints for room state, frames, and WebSocket notifications.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run locally:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Notes:
- The actual inference implementation should be provided as `real_inference.py` exposing `get_room_state(room_id)` and `get_room_frame(room_id)`.
- If `real_inference` is not present, the server will simulate responses.

SMTP / email notifications
- The backend can send emails when warnings or alerts are produced. Configure the following environment variables (create a `.env` file in `backend/`):

```
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=your-smtp-user
SMTP_PASSWORD=your-smtp-password
SMTP_FROM=alerts@example.com
SMTP_USE_TLS=true
ALERT_RECIPIENTS=ops@example.com,admin@example.com
```

If SMTP settings are not provided the backend will still run but will skip sending emails.

Docker / Raspberry Pi
---------------------
This repository includes a `Dockerfile` that uses the official multi-arch `python:3.11-slim` image so it can be built for Raspberry Pi (ARM) platforms.

Build the image on the Pi directly:

```powershell
cd c:/Users/moham/PycharmProjects/start-hack-one-ware/backend
docker build -t room-backend:latest .
```

Run the container (reads env from `.env`):

```powershell
docker run -d --name room-backend -p 8000:8000 --env-file .env room-backend:latest
```

Build for Raspberry Pi from an x86 machine (use Docker Buildx):

```powershell
# create a builder if you haven't already
docker buildx create --use
docker buildx build --platform linux/arm/v7,linux/arm64 -t room-backend:latest --load .
```

Notes:
- The image exposes port `8000` and runs `uvicorn app.main:app`.
- Provide your SMTP and alert recipients via `.env` as documented above for email notifications.

