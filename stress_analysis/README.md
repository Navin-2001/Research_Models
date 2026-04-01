# Social Media Stress Analysis API

This is a **standalone** machine learning backend designed to analyze social media usage durations, frequencies, and late-night habits to compute stress and sleep impact scores.

As requested, this model is **completely decoupled** from the React Native frontend and runs entirely on its own.

## How to run this isolated model on a separate port

Since this module has its own `requirements.txt` and operates independently from the main Face/Voice/Chat server, you should run it in its own isolated Python environment.

Open a **new terminal window** on your Mac and run these commands:

### Step 1: Navigate to the folder

```bash
cd "Desktop/Wellness Tracker/backend/models/stress_analysis"
```

### Step 2: Create a separate virtual environment

```bash
python3 -m venv stress_env
```

### Step 3: Activate the new environment

```bash
source stress_env/bin/activate
```

### Step 4: Install the dedicated dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Start the standalone server (Runs on Port 5001)

```bash
python server.py
```

---

## Testing the API with Postman (No frontend required)

To test this API separately using Postman while the server is running on `port 5001`:

### 1. Test Pattern Behavioral Scoring

1. Open Postman and create a new **POST** request.
2. Enter the URL: `http://localhost:5001/analyze/patterns`
3. Go to the **Headers** tab and add:
   - **Key**: `Content-Type`
   - **Value**: `application/json`
4. Go to the **Body** tab, select **raw**, and ensure the dropdown says **JSON**.
5. Paste this exact testing data into the text box:

```json
{
  "patterns": [
    {
      "duration_minutes": 150,
      "frequency": 40,
      "late_night_logins": 2
    },
    {
      "duration_minutes": 200,
      "frequency": 55,
      "late_night_logins": 5
    }
  ]
}
```

6. Click **Send**! You will get back an AI risk assessment block.

### 2. Test Sleep Insight Scoring

1. Change the URL in Postman to: `http://localhost:5001/analyze/insights`
2. Keep the exact same JSON in the **Body**.
3. Click **Send**! You will get back a circadian rhythm disruption score.
