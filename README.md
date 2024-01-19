# Dynamic-Urban-Traffic-Management-System

Install necessary libraries:

pip install opencv-python numpy

Download the YOLOv3 weights and configuration file from the official website: https://pjreddie.com/darknet/yolo/

Create a Python script (e.g., traffic_management.py) and use the following code:

import cv2
import numpy as np

# Load YOLOv3
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Load traffic video
cap = cv2.VideoCapture("traffic_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Detect objects using YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(layer_names)

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Filter for vehicles (class_id 2 in COCO dataset)
                if class_id == 2:
                    # Implement traffic management logic here
                    # For example, draw bounding boxes or perform actions based on detected vehicles
                    x, y, w, h = map(int, obj[0:4] * [width, height, width, height])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Traffic Management", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

Replace "yolov3.weights", "yolov3.cfg", and "traffic_video.mp4" with the paths to your YOLOv3 weights file, configuration file, and traffic video, respectively.
