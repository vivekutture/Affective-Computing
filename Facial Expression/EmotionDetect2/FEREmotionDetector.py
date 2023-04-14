import cv2
import fer
from statistics import mode

class FEREmotionDetector:

    face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # getting a haarcascade xml file
    face_cascade = cv2.CascadeClassifier()  # processing it for our project
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  # adding a fallback event
        print("Error loading xml file")

    def showLiveEmotions(self,source):
        video = cv2.VideoCapture(source)  # requisting the input from the webcam or camera

        while video.isOpened():  # checking if are getting video feed and using it
            ret, frame = video.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # changing the video to grayscale to make the face analisis work properly
            face = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in face:
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
                                    1)  # making a recentangle to show up and detect the face and setting it position and colour

            emo_detector = fer.FER(mtcnn='True')
            emotions = emo_detector.top_emotion(frame);
            cv2.putText(frame, str(emotions), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, 20)
            print(emotions)

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
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # changing the video to grayscale to make the face analisis work properly
            face = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in face:
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),1)  # making a recentangle to show up and detect the face and setting it position and colour

            emo_detector = fer.FER(mtcnn='True')
            emotions = emo_detector.top_emotion(frame);
            cv2.putText(frame, str(emotions), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, 20)
            print(emotions[0])
            if(emotions[0]!="neutral"):
                emotionsList.append(emotions[0])
            cv2.imshow('video', frame)
            key = cv2.waitKey(1)
            if key == ord("q"):  # here we are specifying the key which will stop the loop and stop all the processes going
                break
        video.release()
        cv2.destroyAllWindows()
        res = mode(emotionsList)
        return res
