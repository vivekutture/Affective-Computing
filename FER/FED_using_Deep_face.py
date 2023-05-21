import cv2
from deepface import DeepFace
import cv2
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
	frameOpencvDnn = frame.copy()
	frameHeight = frameOpencvDnn.shape[0]
	frameWidth = frameOpencvDnn.shape[1]
	blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

	net.setInput(blob)
	detections = net.forward()
	faceBoxes = []
	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > conf_threshold:
			x1 = int(detections[0, 0, i, 3] * frameWidth)
			y1 = int(detections[0, 0, i, 4] * frameHeight)
			x2 = int(detections[0, 0, i, 5] * frameWidth)
			y2 = int(detections[0, 0, i, 6] * frameHeight)
			faceBoxes.append([x1, y1, x2, y2])
			cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
	return frameOpencvDnn, faceBoxes


parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

faceProto = "assets/opencv_face_detector.pbtxt"
faceModel = "assets/opencv_face_detector_uint8.pb"
ageProto = "assets/age_deploy.prototxt"
ageModel = "assets/age_net.caffemodel"
genderProto = "assets/gender_deploy.prototxt"
genderModel = "assets/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-3)', '(4-7)', '(8-14)', '(15-24)', '(25-37)', '(38-47)', '(48-59)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
video_capture = cv2.VideoCapture(0)
padding = 20
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    resultImg, faceBoxes = highlightFace(faceNet, frame)

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
					 min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
																	:min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_img = frame[y:y + h, x:x + w]

        # Perform emotion detection
        predictions = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)

        # Get the dominant emotion label
        emotion_predictions = predictions[0]['emotion']
        emotion = max(emotion_predictions, key=emotion_predictions.get)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the emotion label on the frame
        cv2.putText(frame,"Emotion: "+ emotion +", Gender: "+gender+", Age: "+age, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
