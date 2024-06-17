from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import io

app = FastAPI()

# Load the pre-trained model
model = load_model('lrcn_model.h5')

categories = ['신호위반', '중앙선침범', '진로변경위반']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    video_bytes = await file.read()
    video_stream = io.BytesIO(video_bytes)
    
    frame_list = []
    TARGET_FRAME_COUNT = 25

    cap = cv2.VideoCapture(video_stream)
    if not cap.isOpened():
        print(f"Error opening video file {video_stream}")
        return frame_list
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (64, 64))
        normalized_frame = resized_frame / 255.0
        frame_list.append(normalized_frame)
    cap.release()
    
    # Adjust the frame list to have exactly TARGET_FRAME_COUNT frames
    if len(frame_list) < TARGET_FRAME_COUNT:
        # If less than TARGET_FRAME_COUNT, repeat frames
        max_len = len(frame_list)
        point = 0
        while len(frame_list) < TARGET_FRAME_COUNT:
            frame_list.append(frame_list[point])
            point += 1
            if point == max_len:
                point = 0
    elif len(frame_list) > TARGET_FRAME_COUNT:
        # If more than TARGET_FRAME_COUNT, sample frames evenly
        indices = np.linspace(0, len(frame_list) - 1, TARGET_FRAME_COUNT).astype(int)
        frame_list = [frame_list[i] for i in indices]
    
    predicted_labels_probabilities = model.predict(np.expand_dims(frame_list, axis=0))[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class = categories[predicted_label]
    
    return JSONResponse(content={"prediction": predicted_class})
