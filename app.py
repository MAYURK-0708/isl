# app.py - FastAPI web application for action recognition from uploaded videos
# This script defines a FastAPI web application that allows users to upload videos
# and get action recognition predictions using a pre-trained LSTM model.

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from tempfile import NamedTemporaryFile
import os
import uvicorn
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM,Dense

from helper_functions import convert_video_to_pose_embedded_np_array

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
    video_format = os.path.splitext(file.filename)[1]

    # Check if the video format is valid
    if video_format not in ['.mp4', '.avi', '.mov', '.webm']:
        return JSONResponse(content={"error": "Invalid video format. Supported: MP4, AVI, MOV, WEBM"}, status_code=400)

    # Save the uploaded video with the correct extension
    temp = NamedTemporaryFile(suffix=video_format, delete=False)
    try:
        
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);

            print(temp.name)
            print("Processing video for prediction...")
            out_np_array=convert_video_to_pose_embedded_np_array(temp.name,remove_input=False) #function to detect key points in each frame and return them as an numpy array.
    
            print(f"Input array shape: {out_np_array.shape}")
            prediction=model.predict(np.expand_dims(out_np_array,axis=0), verbose=0)
            print(f"Raw prediction probabilities: {prediction[0]}")
            arg_pred=np.argmax(prediction,axis=1)
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
            
            # Lower confidence threshold and add warning for low accuracy model
            MIN_CONFIDENCE = 0.25
            
            if confidence < MIN_CONFIDENCE:
                print(f"⚠️ Low confidence ({confidence:.2%}) - gesture not recognized clearly")
                return JSONResponse(content={
                    "prediction": "Gesture not recognized",
                    "confidence": f"{confidence*100:.1f}%",
                    "all_predictions": prediction_dict,
                    "message": f"⚠️ Model accuracy is low (69%). Please perform the gesture more clearly. Minimum confidence: {MIN_CONFIDENCE*100:.0f}%",
                    "warning": "This model needs more training data (100+ videos per gesture) for better accuracy."
                })
            
            # Add warning for predictions between 25-50% confidence
            warning_message = None
            if confidence < 0.50:
                warning_message = f"⚠️ Low confidence prediction. Model trained with limited data (21 videos per gesture). Recommended: 100+ videos for reliable predictions."
            
        except Exception as e:
            return JSONResponse(content={"error": f"Error processing video: {str(e)}"}, status_code=500)
        finally:
            file.file.close()

    except Exception as e:
        return JSONResponse(content={"error": f"Error handling file: {str(e)}"}, status_code=500)
    finally:
        #temp.close()  # the `with` statement above takes care of closing the file
        os.remove(temp.name)
        
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)