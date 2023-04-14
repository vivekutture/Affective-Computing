from deepface import DeepFace
import os
import cv2

def recognize():
    face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # getting a haarcascade xml file
    face_cascade = cv2.CascadeClassifier()  # processing it for our project
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  # adding a fallback event
        print("Error loading xml file")
    video = cv2.VideoCapture(0)  # requisting the input from the webcam or camera
    ret, frame = video.read()
    cv2.imshow("face",frame)
    cv2.imwrite("captured/newImg.jpg",frame)
    key = cv2.waitKey(1)
    video.release()
    # cv2.destroyAllWindows()
    photos = os.listdir("images")
    #for i in photos:
    img = DeepFace.verify(img1_path="images/surekha.jpg",img2_path="captured/newImg.jpg",model_name='VGG-Face')
    print(img)