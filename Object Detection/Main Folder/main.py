from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2
import numpy
import pyttsx3
import itertools
model = YOLO("yolov8s.pt")
results = model.predict(source="0",stream="True",show=True,conf=.75)
text_speech = pyttsx3.init()
visited_list = []
result = results
for r in result:
    box = r.boxes
    ids_of_detect = list( box.cls.numpy())
    cords = box.xyxy.tolist()
    #print(cords)
    for cord in cords:
        x1 = cord[0]
        y1 = cord[1]
        x2 = cord[2]
        y2 = cord[3]
        x_center = (x1+x2)/2
        y_center = (y1+y2)/2
        if y_center < 320:
            posi = "upper "
        else:
            posi = "below "
        if x_center < 160:
            posi += "left"
        elif x_center < 320:
            posi += "center"
        else: 
            posi += "right"
    #print(ids_of_detect)
    for id in ids_of_detect:
        detected_object = r.names[int(id)]
        if detected_object not in visited_list:
            feedback = detected_object+" is detected at position "+posi
            print(feedback)
            text_speech.say(feedback)
            visited_list.append(detected_object)
            text_speech.runAndWait()