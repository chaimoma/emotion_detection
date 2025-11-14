import cv2
import tensorflow as tf
import numpy as np

# SET UP CONSTANTS
MODEL_IMAGE_SIZE = 48
CLASS_NAMES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# LOAD MODELS
model = tf.keras.models.load_model('saved_model/emotion_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Models loaded successfully!")

# VideoCapture to open your video.
cap = cv2.VideoCapture(0)
print("Video file opened. Processing...")


while True:
    # Read one frame from the video
    ret, frame = cap.read()
    
    # 'ret' will be False when the video is finished
    if not ret:
        print("Video processing finished.")
        break
        
    # Convert frame to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find faces
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Loop over each face found in this frame
    for (x, y, w, h) in faces:
        face_roi_gray = gray_img[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi_gray, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))
        processed_face = resized_face.reshape(1, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 1)
        
        # Predict
       
        prediction = model.predict(processed_face, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion_label = CLASS_NAMES[emotion_index]
        
        # Draw on the *color frame*
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the current frame in the pop-up window
    cv2.imshow('Emotion Detection (Press Q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows() 