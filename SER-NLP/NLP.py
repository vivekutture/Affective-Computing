from transformers import pipeline
import logging
def NLP():
    logging.getLogger("transformers.modeling_tf_utils").setLevel(logging.ERROR)
    sentence = open('read.txt', encoding='utf-8').read()
    sentence.strip()
    lower_case = sentence.lower()
    print('Speech to Text : {}'.format(lower_case))
    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    emotion_labels = emotion(lower_case);
    res = emotion_labels[0]['label']
    print('\nSentiment by NLP : ' + res.capitalize())

