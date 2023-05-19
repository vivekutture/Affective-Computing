from transformers import pipeline
def NLP():
    sentence = open('read.txt', encoding='utf-8').read()
    sentence.strip()
    lower_case = sentence.lower()
    print('\n\nSpeech to Text : {}\n\n'.format(lower_case))
    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    emotion_labels = emotion(lower_case);
    res = emotion_labels[0]['label']
    print('\n\nSentiment by NLP : ' + res.capitalize())

