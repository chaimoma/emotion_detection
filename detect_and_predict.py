import tensorflow as tf
import cv2 
import numpy as np

# SET UP  CONSTANTS
model_image_size=48
class_names=['angry','disgusted','fearful','happy','neutral','sad','surprised']
#LOAD MODELS
model=tf.keras.models.load_model('saved_model/emotion_model.h5') #load the cnn model 'brain'
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #load the face finder 'haar cascade'
print('models loaded successfully!')

#laod and prepare the test photo
img=cv2.imread('testphotos/test_image.jpg') 
#convert photo to grayscale ( for haar cascade)
gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(
    gray_img,
    scaleFactor=1.1,
    minNeighbors=8,
    minSize=(30,30)
)
print(f'founed {len(faces)} face(s)')
for(x,y,w,h) in faces:
    #1-cut out the face from the grayscaler image
    face_roi_gray=gray_img[y:y+h,x:x+w]
    #2-Resize the face to 84*48(ela wdit cnn)
    resized_face=cv2.resize(face_roi_gray, (model_image_size,model_image_size))
    #3-reshape the image for the brain (1,48,48,1)
    processed_face=resized_face.reshape(1,model_image_size,model_image_size,1)
    #4-predict the emotion
    prediction=model.predict(processed_face)
    #5-get the emotion name 
    emotion_index=np.argmax(prediction)
    emotion_label=class_names[emotion_index]

    #6- draws the results of the prediction onto the original, full-sized, color image that the user uploaded.
    cv2.rectangle(img,(x,y),(x+w,x+h),(0,255,0),2)
    cv2.putText(img,emotion_label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

#6. show the final image 
#open pop up window to display result
print('displaying result, press any key to close')
#cv2.namedWindow('emotion detection', cv2.WINDOW_NORMAL)
cv2.imshow('emotion detection',img)
cv2.waitKey(0) #waiting for press any key
cv2.destroyAllWindows()
