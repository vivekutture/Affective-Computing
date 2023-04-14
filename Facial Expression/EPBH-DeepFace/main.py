import EmotionDetector
import FaceRecognizer
if __name__ == '__main__':
    e = EmotionDetector.EmotionDetector()
    emotion = e.getInitialEmotions(0,100)
    print(emotion)
