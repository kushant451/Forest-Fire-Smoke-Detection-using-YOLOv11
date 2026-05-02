import cv2

# Load pre-trained YOLO model (using OpenCV DNN - simplified)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load image
image = cv2.imread("test.jpg")
height, width, _ = image.shape

# Create blob
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
layer_names = net.getUnconnectedOutLayersNames()
outputs = net.forward(layer_names)

# Process detections
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = scores.argmax()
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            label = str(classes[class_id])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(image, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Show output
cv2.imshow("Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
