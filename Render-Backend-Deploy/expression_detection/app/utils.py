from deepface import DeepFace
import numpy as np

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj

def detect_emotion(image_path):
    result = DeepFace.analyze(
        img_path=image_path,
        actions=['emotion'],
        enforce_detection=False
    )

    # If DeepFace returns a list, take first element
    if isinstance(result, list):
        result = result[0]

    # Convert numpy → Python float
    result = convert_numpy(result)

    return result["emotion"]

