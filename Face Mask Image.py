from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

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
net = cv2.dnn.readNet(prototxtPath, weightsPath)

print("loading face mask detector model...")
model = load_model("model.h5")

image = cv2.imread("examples/image5.jpg")
orig = image.copy()
(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

print("computing face detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0,detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > args['confidence']:
        box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        
        (mask, withoutMask) = model.predict(face)[0]
        
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0,255,0) if label == "Mask" else (0,0,255)
    
#         label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
#         cv2.putText(image, label, (w-110, h-10),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

cv2.imshow("Output", image)



