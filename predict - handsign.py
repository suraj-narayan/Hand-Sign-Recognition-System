

from tensorflow import keras
import cv2
import numpy as np


modelFileName = "model4F.01-0.50.hdf5"
loaded_model = keras.models.load_model(modelFileName) 

cap = cv2.VideoCapture(0)
_, first_frame = cap.read()
first_frame = cv2.flip(first_frame, 1)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    diff = cv2.subtract(first_frame,frame)
    diff_gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, 21, 255, cv2.THRESH_BINARY)
    res = cv2.bitwise_and(frame,frame,mask=mask)

    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])

    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    

    roi = res[y1:y2, x1:x2]
    

    roi = cv2.resize(roi, (128, 128))
    roi = cv2.flip(roi, 1)
    cv2.imshow("test", roi)

    image_arrays = [roi]
    image_arrays = np.array(image_arrays)

    img_features = image_arrays.astype('float32')
    img_features /= 255

    classnames = ['E','F','I','L','V']

    predictions = loaded_model.predict(img_features)
    confidence = sorted(predictions[0], reverse=1)
    confidence = confidence[0]

    class_idx = np.argmax(predictions[0])
    result = classnames[class_idx]
    

    cv2.putText(frame, result, (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)    
    cv2.imshow("Frame", frame)
    cv2.imshow("diff_gray", diff_gray)
    
    keyboard = cv2.waitKey(30) & 0xFF
    if keyboard == ord('q') or keyboard == 27:
        break
    elif keyboard == ord('b'):
        _, first_frame = cap.read()
        first_frame = cv2.flip(first_frame, 1)        
 
cap.release()
cv2.destroyAllWindows()
