from transformers import pipeline
import speech_recognition as sr
def speechToText():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Please say something")
        audio = r.listen(source)
        print("Recognizing Now .... ")
        try:
            speech = open('read.txt', 'w')
            speech.write(r.recognize_google(audio))
        except Exception as e:
            print("Error :  " + str(e))

speechToText()
sentence = open('read.txt', encoding='utf-8').read()
lower_case = sentence.lower()
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
emotion_labels = emotion(lower_case);
res = emotion_labels[0]['label']
print('Emotion by NLP : ' + res)

