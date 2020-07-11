from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import imutils
import time
import os
from imutils.video import VideoStream
from PIL import Image

def detect_and_predict_mask(frame,faceNet,maskNet):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
		
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            
    if len(faces) > 0:
        preds = maskNet.predict(faces)
        
    return (locs, preds)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("loading face detector model")
prototxtPath = os.path.sep.join([args['face'],"deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("loading face mask detector model...")
maskNet = load_model("model.h5")

#%%
vs = VideoStream(src=0).start()
time.sleep(2.0)

speedo = Image.open('examples/mask.png').convert('RGBA')


while True:
    frame = vs.read()
    frame = imutils.resize(frame,width=1080)
    
    (locs,preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    for (box,pred) in zip(locs,preds):
        (startX,startY,endX,endY) = box
        (mask,withoutMask) = pred
        
        label = "Maske Var" if mask > withoutMask else "Maske Yok"
        color = (0, 255, 0) if label == "Maske Var" else (0, 0, 255)
        cv2.rectangle(frame, (startX+20, startY), (endX, endY), color, 2)
        cv2.putText(frame, label, (startX+10,endY+25), cv2.FONT_HERSHEY_DUPLEX , 1, color)
        
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()













