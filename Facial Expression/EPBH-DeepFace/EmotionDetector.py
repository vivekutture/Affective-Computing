import cv2
from statistics import mode
from deepface import DeepFace


class EmotionDetector:

    # def __int__(self, face_cascade_name, face_cascade):
    face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # getting a haarcascade xml file
    face_cascade = cv2.CascadeClassifier()  # processing it for our project
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  # adding a fallback event
        print("Error loading xml file")

    def showLiveEmotions(self,source):
        video = cv2.VideoCapture(source)  # requisting the input from the webcam or camera
        while video.isOpened():  # checking if are getting video feed and using it
            ret, frame = video.read()
            # print(frame)
            gray = cv2.cvtColor(frame,
                                cv2.COLOR_BGR2GRAY)  # changing the video to grayscale to make the face analisis work properly
            face = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in face:
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
                                    1)  # making a recentangle to show up and detect the face and setting it position and colour
                # making a try and except condition in case of any errors
                try:
                    analyze = DeepFace.analyze(frame, actions=['emotion'])  # same thing is happing here as the previous example, we are using the analyze class from deepface and using ‘frame’ as input
                    emotion = analyze['dominant_emotion']
                    print(
                        emotion)  # here we will only go print out the dominant emotion also explained in the previous example
                    cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, 20)
                except Exception as e:
                    print(e)
                # this is the part where we display the output to the user
                cv2.imshow('video', frame)
                key = cv2.waitKey(1)
                if key == ord("q"):  # here we are specifying the key which will stop the loop and stop all the processes going
                    break
        video.release()
        cv2.destroyAllWindows()

    def getInitialEmotions(self, source,time):
        emotionsList = []
        video = cv2.VideoCapture(source)
        for i in range(1, time):
            ret, frame = video.read()
            # print(frame)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # changing the video to grayscale to make the face analisis work properly
            face = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for x, y, w, h in face:
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),1)  # making a recentangle to show up and detect the face and setting it position and colour
                # making a try and except condition in case of any errors
                try:
                    analyze = DeepFace.analyze(frame, actions=['emotion'])  # same thing is happing here as the previous example, we are using the analyze class from deepface and using ‘frame’ as input
                    emotion = analyze['dominant_emotion']
                    print(emotion)  # here we will only go print out the dominant emotion also explained in the previous example
                    cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, 20)
                    if emotion != "neutral":
                        emotionsList.append(emotion)
                except Exception as e:
                    print(e)
                # this is the part where we display the output to the user
                cv2.imshow('video', frame)
                key = cv2.waitKey(1)
                if key == ord("q"):  # here we are specifying the key which will stop the loop and stop all the processes going
                    break
        video.release()
        cv2.destroyAllWindows()
        res = mode(emotionsList)
        return res
