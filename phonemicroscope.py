import cv2 as cv
from roboflow import Roboflow
import json

rf = Roboflow(api_key="DP7cOzly6N68FZSQnWAB")
project = rf.workspace().project("things-qxtyl")
model = project.version(1).model

THRESHOLD = 0.5

vid = cv.VideoCapture(1)
def draw_prediction(model, frame):
    output = model.predict(frame, confidence=40, overlap=30).json()
    predictions = output["predictions"]
    if not predictions:
        print("NOTHING IS DETECTED")
        return
    
    for pred in predictions:
        conf = pred['confidence']
        print(pred)
        if conf < THRESHOLD:
            print('FAKE!')
            return
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']

        x1 = x - w/2
        x2 = x + w/2
        y1 = y - y/2
        y2 = y + y/2
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv.rectangle(frame, p1, p2, color=(0,0,255), thickness=2)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    draw_prediction(model, frame)
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break