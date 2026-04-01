# Vocal Emotion Service (Hugging Face)

This service provides real-time vocal emotion analysis using the `superb/wav2vec2-base-superb-er` transformer model.

## Confirmed Python Version
**Python 3.14.x** (Confirmed working for `torch` and `transformers` on this system).

1. **Python Virtual Environment (using verified 3.14):**
   - **Create:** `py -3.14 -m venv venv`
   - **Activate (Windows):** `.\venv\Scripts\activate`
   - **Activate (Mac/Linux):** `source venv/bin/activate`

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Requires an internet connection on first run to download the weights (~300MB).*

3. **FFmpeg Dependency:**
   To process `.m4a` files (common on mobile), this service requires **FFmpeg**.
   - Ensure `ffmpeg` is in your system PATH.
   - On Windows, the service will also try to auto-detect FFmpeg if installed in `C:\KMPlayer\ffmpeg.exe`.

## Running the Service

```bash
python -m uvicorn model:app --host 0.0.0.0 --port 5007 --reload
```
The service runs on Port **5007**.

## Usage Details

- **Endpoint:** `POST /predict_emotion`
- **Body:** Multipart/form-data with key `audio_file`.
- **Response Format:**
  ```json
  {
      "neutral": 65.2,
      "happy": 5.1,
      "sad": 20.4,
      "angry": 9.3,
      ...
  }
  ```
