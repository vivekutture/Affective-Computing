from transformers import pipeline
def NLP():
    sentence = open('read.txt', encoding='utf-8').read()
    lower_case = sentence.lower()
    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    emotion_labels = emotion(lower_case);
    res = emotion_labels[0]['label']
    print('Emotion by NLP : ' + res)

