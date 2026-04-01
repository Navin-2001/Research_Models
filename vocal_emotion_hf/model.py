from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
import librosa
import numpy as np
import os
import shutil
import uuid
from pydub import AudioSegment
import io

app = FastAPI(title="Hugging Face Vocal Emotion Detection")

# Global model variable
classifier = None

# Expected output keys by Scoring Service
# Vocal: neutral, calm, happy, sad, angry, fearful, disgust, surprised
ALL_EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

@app.on_event("startup")
def load_model():
    global classifier
    print("Loading Hugging Face Wav2Vec2 Model... This may take a moment.")
    # superb/wav2vec2-base-superb-er detects: neutral, happy, angry, sad
    classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    print("Model Loaded Successfully!")
    
    # 1. Search for FFmpeg in common locations if not in PATH
    import shutil
    ffmpeg_exe = shutil.which("ffmpeg")
    
    if not ffmpeg_exe:
        common_ffmpeg_paths = [
            r"C:\Users\Yasitha.k\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
            r"/usr/local/bin/ffmpeg",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
        ]
        for path in common_ffmpeg_paths:
            if os.path.exists(path):
                ffmpeg_exe = path
                print(f"Auto-detected FFmpeg at: {ffmpeg_exe}")
                # Inject into environment path for librosa/audioread
                os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_exe)
                # Point pydub to it
                try:
                    AudioSegment.converter = ffmpeg_exe
                    AudioSegment.ffprobe = os.path.join(os.path.dirname(ffmpeg_exe), "ffprobe.exe")
                    if not os.path.exists(AudioSegment.ffprobe):
                         # KMPlayer might only have ffmpeg.exe
                         AudioSegment.ffprobe = ffmpeg_exe 
                except Exception as e:
                    print(f"Error configuring pydub paths: {e}")
                break
    
    if ffmpeg_exe:
        print(f"FFmpeg ready at: {ffmpeg_exe} (m4a support enabled)")
    else:
        print("WARNING: FFmpeg not found. m4a/mp4 decoding WILL fail.")
        print("Please install FFmpeg or ensure it's in your PATH.")

@app.get("/check_ffmpeg")
def check_ffmpeg():
    import shutil
    ffmpeg_exe = shutil.which("ffmpeg")
    return {
        "ffmpeg_found": ffmpeg_exe is not None,
        "path": ffmpeg_exe,
        "advice": "FFmpeg is ready." if ffmpeg_exe else "Install FFmpeg or point the server to its directory."
    }

@app.post("/predict_emotion")
async def predict_emotion(audio_file: UploadFile = File(...)):
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is still loading")

    # 1. Save temp file with original extension
    file_ext = os.path.splitext(audio_file.filename)[1] if audio_file.filename else '.m4a'
    temp_filename = f"temp_{uuid.uuid4()}{file_ext}"
    
    try:
        print(f"Receiving audio file: {audio_file.filename}")
        
        # Read file content into memory first to avoid holding file handle longer than needed
        content = await audio_file.read()
        with open(temp_filename, "wb") as buffer:
            buffer.write(content)
        
        print(f"Saved to: {temp_filename}")
        
        # 2. Resample to 16kHz (Model Requirement)
        speech, sr = None, None
        
        # Determine if it's a WAV file for direct loading
        is_wav = temp_filename.lower().endswith('.wav') or (audio_file.filename and audio_file.filename.lower().endswith('.wav'))

        try:
            print(f"Attempting to load {temp_filename}...")
            if is_wav:
                # soundfile is fast for WAV
                import soundfile as sf
                with open(temp_filename, 'rb') as f:
                    speech, sr_org = sf.read(f)
                    if speech.ndim > 1:
                        speech = speech.mean(axis=1)
                    if sr_org != 16000:
                        speech = librosa.resample(speech, orig_sr=sr_org, target_sr=16000)
                    sr = 16000
            else:
                # For m4a/mp3 etc, we need FFmpeg fallback
                print(f"Non-WAV file detected, attempting librosa/pydub load...")
                try:
                    speech, sr = librosa.load(temp_filename, sr=16000, duration=10.0)
                except Exception as lib_err:
                    print(f"Librosa direct load failed, trying pydub workaround: {lib_err}")
                    try:
                        audio = AudioSegment.from_file(temp_filename)
                        wav_io = io.BytesIO()
                        audio.export(wav_io, format="wav")
                        wav_io.seek(0)
                        import soundfile as sf
                        speech, sr_org = sf.read(wav_io)
                        if speech.ndim > 1:
                            speech = speech.mean(axis=1)
                        if sr_org != 16000:
                            speech = librosa.resample(speech, orig_sr=sr_org, target_sr=16000)
                        sr = 16000
                        wav_io.close()
                    except Exception as pydub_err:
                        err_msg = str(pydub_err)
                        if "find the file specified" in err_msg or "[WinError 2]" in err_msg:
                            raise HTTPException(status_code=400, detail="FFmpeg is not installed on the server. Please install FFmpeg to support mobile recordings.")
                        raise pydub_err
            
            print(f"Audio loaded successfully: {len(speech)} samples")
        except HTTPException:
            raise
        except Exception as err:
            print(f"All audio loading paths failed: {err}")
            raise HTTPException(status_code=400, detail=f"Could not load audio file: {str(err)}")
        
        # Check if audio is valid
        if speech is None or len(speech) == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty or invalid")
        
        # 3. Predict
        labels = classifier(speech, top_k=None) 
        
        # 4. Map to Standard Schema
        label_map = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}
        final_scores = {k: 0.0 for k in ALL_EMOTIONS}
        
        for item in labels:
            short_label = item['label']
            score_pct = item['score'] * 100
            mapped_label = label_map.get(short_label)
            if mapped_label:
                final_scores[mapped_label] = score_pct
        
        print(f"Prediction successful: {final_scores}")
        return final_scores

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Cleanup with retry for Windows locks
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            # Close the upload file handle if it exists
            await audio_file.close()
            
            # Explicitly clear variables that might hold handles
            if 'speech' in locals(): del speech
            if 'audio' in locals(): del audio
            
            import time
            for _ in range(3): # Try up to 3 times
                try:
                    os.remove(temp_filename)
                    break
                except Exception:
                    time.sleep(0.1)

@app.get("/")
def health_check():
    return {"status": "HF Vocal Emotion Service Running (Port 5007)"}
