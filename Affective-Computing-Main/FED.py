import cv2
from deepface import DeepFace
import os

def FacailExpressions():
    # Load pre-trained models
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the webcam
    video_capture = cv2.VideoCapture(0)
    padding = 20

    emotion_percentages={}

    while True:
        os.system('cls')
        # Capture frame-by-frame
        ret, frame = video_capture.read()

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
            emotion_value=emotion_predictions[emotion]
            #print(emotion_predictions)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display the emotion label on the frame
            cv2.putText(frame,"Emotion: "+ emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Store the probabilities in the dictionary
            for emotion, probability in emotion_predictions.items():
                if emotion in emotion_percentages:
                    emotion_percentages[emotion].append(probability)
                else:
                    emotion_percentages[emotion] = [probability]
                    
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ans=[]
    if emotion_percentages:
        most_probable_emotion = max(emotion_percentages, key=lambda x: sum(emotion_percentages[x]) / len(emotion_percentages[x]))
        probabilities = emotion_percentages[most_probable_emotion]
        average_percentage = sum(probabilities) / len(probabilities)
        ans.append(most_probable_emotion)
        ans.append(average_percentage)

    # Release the webcam and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

    return ans
