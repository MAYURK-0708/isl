# app.py - FastAPI web application for action recognition from uploaded videos and real-time recognition
# This script defines a FastAPI web application that allows users to upload videos
# and get action recognition predictions using a pre-trained LSTM model.
# It also supports real-time gesture recognition via WebSocket.

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from tempfile import NamedTemporaryFile
import os
import uvicorn
import numpy as np
import cv2
import base64
import json
import mediapipe as mp
import random
import glob

from keras.models import Sequential
from keras.layers import LSTM,Dense

from helper_functions import convert_video_to_pose_embedded_np_array, mediapipe_detection, extract_keypoints

app = FastAPI(title="Indian Sign Language Recognition", description="AI-powered sign language recognition system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/training-videos", StaticFiles(directory="training-data"), name="training-videos")

actions=np.array(["Hello","How are you","thank you","alright","good morning","good afternoon"])

def initialize_model():
    """ Initializes lstm model and loads the trained model weight  """
    model = Sequential()
    model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(45,258)))
    model.add(LSTM(128,return_sequences=True, activation = 'relu'))
    model.add(LSTM(256,return_sequences=True,activation="relu"))
    model.add(LSTM(64, return_sequences = False,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation = 'relu'))
    model.add(Dense(actions.shape[0],activation='softmax'))

    print(model.summary())
    model.compile(optimizer = 'Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])

    model.load_weights(r'lstm-model-6classes-best.hdf5')

    return model

model = initialize_model()

@app.get("/")
async def root():
    """ Serve the main HTML page """
    return FileResponse('static/index.html')

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """ Receives video from client and predicts the sign language action """
    
    # Get file extension
    video_format = os.path.splitext(file.filename)[1].lower()
    
    print(f"Received file: {file.filename}, format: {video_format}")

    # Check if the video format is valid (support more formats)
    valid_formats = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv', '.wmv', '.m4v']
    if video_format not in valid_formats:
        return JSONResponse(
            content={"error": f"Invalid video format '{video_format}'. Supported: MP4, AVI, MOV, WEBM, MKV, FLV, WMV, M4V"}, 
            status_code=400
        )

    # Save the uploaded video with the correct extension
    temp = NamedTemporaryFile(suffix=video_format, delete=False)
    temp_path = temp.name
    
    try:
        # Read and write file contents
        try:
            print(f"Reading uploaded file...")
            contents = await file.read()
            print(f"File size: {len(contents)} bytes")
            
            with open(temp_path, 'wb') as f:
                f.write(contents)
            
            print(f"Saved to temporary file: {temp_path}")
            
            # Verify file was written correctly
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise ValueError("Failed to save uploaded video file")
            
            print("Processing video for prediction...")
            out_np_array = convert_video_to_pose_embedded_np_array(temp_path, remove_input=False)
            
            # Verify output array shape
            if out_np_array.shape != (45, 258):
                raise ValueError(f"Invalid output array shape: {out_np_array.shape}. Expected (45, 258)")
    
            print(f"Input array shape: {out_np_array.shape}")
            prediction = model.predict(np.expand_dims(out_np_array, axis=0), verbose=0)
            print(f"Raw prediction probabilities: {prediction[0]}")
            arg_pred = np.argmax(prediction, axis=1)
            confidence = float(prediction[0][arg_pred[0]])
            
            # Get all predictions with confidence
            prediction_dict = {
                actions[i]: float(prediction[0][i]) for i in range(len(actions))
            }
            
            # Sort predictions by confidence
            sorted_predictions = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)
            print(f"Top 3 predictions:")
            for i, (gesture, prob) in enumerate(sorted_predictions[:3]):
                print(f"  {i+1}. {gesture}: {prob:.4f} ({prob*100:.1f}%)")
            
            predicted_gesture = actions[arg_pred[0]]
            print(f"Final prediction: {predicted_gesture} with confidence: {confidence:.4f}")
            
            # Check if prediction is too certain (100%) - indicates overfitting
            if confidence > 0.99:
                print(f"⚠️ WARNING: Extremely high confidence ({confidence:.2%}) - model may be overfitted")
                return JSONResponse(content={
                    "prediction": predicted_gesture,
                    "confidence": f"{confidence*100:.1f}%",
                    "all_predictions": prediction_dict,
                    "message": f"Predicted: {predicted_gesture}",
                    "warning": "⚠️ Model showing very high confidence - this may indicate overfitting."
                })
            
            # Lower confidence threshold
            MIN_CONFIDENCE = 0.20
            
            if confidence < MIN_CONFIDENCE:
                print(f"⚠️ Low confidence ({confidence:.2%}) - gesture not recognized clearly")
                return JSONResponse(content={
                    "prediction": f"Uncertain: {predicted_gesture}?",
                    "confidence": f"{confidence*100:.1f}%",
                    "all_predictions": prediction_dict,
                    "message": f"⚠️ Low confidence ({confidence*100:.1f}%). Top guess: {predicted_gesture}",
                    "warning": "The gesture may not be clear. Ensure good lighting and perform gesture clearly."
                })
            
            # Add warning for predictions between 20-50% confidence
            warning_message = None
            if confidence < 0.50:
                warning_message = f"⚠️ Medium confidence ({confidence*100:.1f}%). Consider performing gesture more clearly."
            
        except ValueError as ve:
            print(f"ValueError: {str(ve)}")
            return JSONResponse(
                content={
                    "error": f"Video processing error: {str(ve)}",
                    "message": "Could not process video. Ensure the video clearly shows sign language gestures."
                }, 
                status_code=400
            )
        except Exception as e:
            print(f"Processing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(
                content={
                    "error": f"Error processing video: {str(e)}",
                    "message": "An error occurred while analyzing the video. Please try again with a different video."
                }, 
                status_code=500
            )
        finally:
            await file.close()

    except Exception as e:
        print(f"File handling error: {str(e)}")
        return JSONResponse(
            content={
                "error": f"Error handling file: {str(e)}",
                "message": "Failed to read uploaded video file."
            }, 
            status_code=500
        )
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {str(e)}")
        
    return JSONResponse(content={
        "prediction": str(actions[arg_pred[0]]),
        "confidence": f"{confidence*100:.1f}%",
        "all_predictions": prediction_dict,
        "message": "Prediction successful!" if not warning_message else warning_message,
        "warning": warning_message
    })

@app.get("/test/")
async def test():
    return JSONResponse(content={"status": "working", "message": "ISL Recognition API is running"})

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """ WebSocket endpoint for real-time gesture recognition """
    await websocket.accept()
    
    frame_buffer = []
    sequence_length = 45  # Number of frames needed for prediction
    
    print("WebSocket connection established")
    
    try:
        with mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        ) as holistic:
            
            while True:
                # Receive frame data from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message['type'] == 'frame':
                    # Decode base64 image
                    image_data = message['data'].split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    
                    # Convert to numpy array
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Process frame with MediaPipe
                    image, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)
                    
                    # Check if hands are detected
                    hands_detected = bool(results.left_hand_landmarks or results.right_hand_landmarks)
                    
                    # Add keypoints to buffer
                    frame_buffer.append(keypoints)
                    
                    # Keep only last sequence_length frames
                    if len(frame_buffer) > sequence_length:
                        frame_buffer.pop(0)
                    
                    # Calculate progress
                    buffer_progress = int((len(frame_buffer) / sequence_length) * 100)
                    
                    # Send status update
                    await websocket.send_json({
                        "type": "status",
                        "buffer_progress": buffer_progress,
                        "frames_collected": len(frame_buffer),
                        "hands_detected": hands_detected,
                        "status": "collecting" if len(frame_buffer) < sequence_length else "ready"
                    })
                    
                    # Make prediction when buffer is full
                    if len(frame_buffer) == sequence_length:
                        try:
                            # Prepare input for model
                            input_array = np.array(frame_buffer)
                            input_array = np.expand_dims(input_array, axis=0)
                            
                            # Make prediction
                            prediction = model.predict(input_array, verbose=0)
                            arg_pred = np.argmax(prediction, axis=1)
                            confidence = float(prediction[0][arg_pred[0]])
                            predicted_gesture = actions[arg_pred[0]]
                            
                            # Get all predictions
                            prediction_dict = {
                                actions[i]: float(prediction[0][i]) 
                                for i in range(len(actions))
                            }
                            
                            # Sort predictions by confidence
                            sorted_predictions = sorted(
                                prediction_dict.items(), 
                                key=lambda x: x[1], 
                                reverse=True
                            )
                            
                            # Send prediction
                            await websocket.send_json({
                                "type": "prediction",
                                "prediction": predicted_gesture,
                                "confidence": f"{confidence*100:.1f}%",
                                "confidence_value": confidence,
                                "all_predictions": prediction_dict,
                                "top_3": [
                                    {"gesture": g, "confidence": f"{c*100:.1f}%"} 
                                    for g, c in sorted_predictions[:3]
                                ],
                                "buffer_progress": 100,
                                "status": "predicted"
                            })
                            
                            # Keep last 30 frames for continuity (sliding window)
                            frame_buffer = frame_buffer[15:]
                            
                        except Exception as e:
                            print(f"Prediction error: {str(e)}")
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Prediction error: {str(e)}"
                            })
                    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Error: {str(e)}"
            })
        except:
            pass

@app.post("/translate/")
async def translate_text(request: dict):
    """
    Translate recognized gesture text to another language
    Note: This is a placeholder implementation. For production, integrate with
    a translation API like Google Translate, DeepL, or Azure Translator.
    """
    try:
        text = request.get('text', '')
        language = request.get('language', 'en')
        
        # Placeholder translation dictionary (Hindi examples)
        translations = {
            'hi': {
                'Hello': 'नमस्ते',
                'How are you': 'आप कैसे हैं',
                'thank you': 'धन्यवाद',
                'alright': 'ठीक है',
                'good morning': 'सुप्रभात',
                'good afternoon': 'शुभ दोपहर'
            },
            'mr': {  # Marathi
                'Hello': 'नमस्कार',
                'How are you': 'तुम्ही कसे आहात',
                'thank you': 'धन्यवाद',
                'alright': 'ठीक आहे',
                'good morning': 'शुभ सकाळ',
                'good afternoon': 'शुभ दुपार'
            },
            'gu': {  # Gujarati
                'Hello': 'નમસ્તે',
                'How are you': 'તમે કેમ છો',
                'thank you': 'આભાર',
                'alright': 'બરાબર',
                'good morning': 'સુપ્રભાત',
                'good afternoon': 'શુભ બપોર'
            }
        }
        
        # Return original if English or translation if available
        if language == 'en':
            translated = text
        else:
            lang_dict = translations.get(language, {})
            translated = lang_dict.get(text, text)
        
        return JSONResponse(content={
            "original": text,
            "translated": translated,
            "language": language,
            "note": "Using basic translation dictionary. For better translations, integrate a translation API."
        })
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Translation error: {str(e)}", "original": text},
            status_code=500
        )

@app.post("/text-to-sign/")
async def text_to_sign(request: dict):
    """
    Find and return a video from training data that demonstrates the requested sign language gesture
    """
    try:
        text = request.get('text', '').strip()
        
        if not text:
            return JSONResponse(
                content={"error": "Please provide text"},
                status_code=400
            )
        
        # Normalize the text to match folder names
        # Map various inputs to the correct gesture names
        gesture_map = {
            'hello': 'Hello',
            'hi': 'Hello',
            'how are you': 'How are you',
            'how are you?': 'How are you',
            'thank you': 'thank you',
            'thanks': 'thank you',
            'alright': 'alright',
            'ok': 'alright',
            'okay': 'alright',
            'good morning': 'good morning',
            'morning': 'good morning',
            'good afternoon': 'good afternoon',
            'afternoon': 'good afternoon'
        }
        
        text_lower = text.lower()
        gesture_name = gesture_map.get(text_lower, text)
        
        # Check if folder exists in training-data
        folder_path = os.path.join('training-data', gesture_name)
        
        if not os.path.exists(folder_path):
            return JSONResponse(content={
                "error": f"No video found for '{text}'",
                "instruction": f"Available gestures: Hello, How are you, thank you, alright, good morning, good afternoon",
                "available_gestures": list(set(gesture_map.values()))
            })
        
        # Get all video files in the folder
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.MOV', '*.MP4', '*.AVI', '*.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if not video_files:
            return JSONResponse(content={
                "error": f"No video files found in folder for '{gesture_name}'",
                "instruction": "Please add video files to the training-data folder"
            })
        
        # Select a random video from the available videos
        selected_video = random.choice(video_files)
        
        # Convert path to URL format
        video_filename = os.path.basename(selected_video)
        video_url = f"/training-videos/{gesture_name}/{video_filename}"
        
        return JSONResponse(content={
            "video_url": video_url,
            "gesture": gesture_name,
            "total_videos": len(video_files),
            "message": f"Showing sign language for: {gesture_name}"
        })
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)