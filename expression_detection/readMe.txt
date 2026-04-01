# Facial Expression Detection Service

This service uses `DeepFace` to analyze facial expressions from images.

## IMPORTANT: Python Version Requirement
This module **REQUIRES Python 3.12**. 
Download: [Python 3.12.x](https://www.python.org/downloads/windows/)

## Setup Instructions

1. **Create Python 3.12 Virtual Environment:**
   Using the Python Launcher:
   ```bash
   py -3.12 -m venv venv
   ```

2. **Activate Virtual Environment:**
   - **Windows:** `.\venv\Scripts\activate`
   - **Mac/Linux:** `source venv/bin/activate`

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Service

The server entry point is in the `app/` directory.
```bash
cd app
python server.py
```
The service will be available at `http://localhost:5005`.

## Usage Details

- **Endpoint:** `POST /api/emotion`
- **Body:** JSON
  ```json
  {
      "image_base64": "..." 
  }
  ```
- **Response:**
  ```json
  {
      "status": "success",
      "emotions": {
          "happy": 95.2,
          "neutral": 4.1,
          ...
      }
  }
  ```
