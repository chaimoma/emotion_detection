from fastapi import FastAPI , Depends,HTTPException,UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from app.models import Prediction,Base
from app.database import engine,SessionLocal
import cv2
import tensorflow as tf
import numpy as np 




# DATABASE HELPER
def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()

        
app=FastAPI(title='Emotion Detection')
Base.metadata.create_all(bind=engine) 

# SET UP CONSTANTS & LOAD MODELS
image_size=48
class_names=['angry','disgusted','fearful','happy','neutral','sad','surprised']

print('models loaded successfuly')


model = tf.keras.models.load_model("saved_model/emotion_model.h5", compile=False)
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



# JSON RESPONSE MODEL
class PredictionResponse(BaseModel):
    id:int
    emotion:str
    confidence:float
    created_at:datetime
    class Config:
        from_attributes = True

# API ROUTE 1: /predict_emotion
@app.post('/predict_emotion', response_model=PredictionResponse)
async def predict_emotion(file:UploadFile=File(...),db: Session=Depends(get_db)):
    
    file_byte= await file.read()
    np_array = np.frombuffer(file_byte, np.uint8)
    image=cv2.imdecode(np_array,cv2.IMREAD_COLOR) 
    gray_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces=cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=8)
    if not len(faces):
         raise HTTPException(status_code=400 ,detail='No faces got found in the photo !')
    (x,y,w,h)=faces[0]
    face_roi=gray_img[y:y+h,x:x+w] 
    resized_face=cv2.resize(face_roi, (image_size,image_size))
    processed_face=resized_face.reshape(1,image_size,image_size,1)
    prediction=model.predict(processed_face,verbose=0)
    confidence=float(np.max(prediction[0]))
    emotion_label=class_names[np.argmax(prediction)]
    db_prediction=Prediction(emotion=emotion_label,confidence=confidence)
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

# API ROUTE 2: 
@app.get('/history', response_model=List[PredictionResponse])
def get_prediction_history(db: Session=Depends(get_db)):
 predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
 return predictions