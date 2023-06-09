import os
import pickle
import speech_recognition as sr
from NLP import NLP
from SER import trainModel
from utils import extract_feature
from FED import FacailExpressions


FEDans=FacailExpressions()

# Clearing the Screen
os.system('cls')

# if model is not trained train the model
if not os.path.exists("SER-Trained-Model.pkl"):
  trainModel()

# load the saved model (after training)
model = pickle.load(open("SER-Trained-Model.pkl", "rb"))

filename = "test.wav"
txtFile= "read.txt"

if os.path.exists(filename):
  os.remove(filename)

if os.path.exists(txtFile):
  os.remove(txtFile)


# record the file (start talking)

r = sr.Recognizer()
with sr.Microphone() as source:
  r.adjust_for_ambient_noise(source)
  print("Voice Recording...")
  print("Please talk")
  audio=r.listen(source)
  try:
    with open(txtFile, "w") as text_file:
      text_file.write(r.recognize_google(audio)) 
    with open(filename, "wb") as wav_file:
      wav_file.write(audio.get_wav_data()) 
    print("Voice Recorded Successfully!")
  except Exception as e:
    print("Error : Voice not Recorded!!! ")

# Clearing the Screen
os.system('cls')

if os.path.exists(filename) and os.path.exists(txtFile):
  try:
    print("\nPrediciting Emotion...")
    
    # By NLP
    emotion_labels=NLP()
    res = emotion_labels[0]['label']
    score= emotion_labels[0]['score']
    percentage=score*100


    # By SER
    # extract features and reshape it
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    result = model.predict(features)[0]
    emotions=["neutral", "happy", "sad", "angry", "fearful"]
    indx=emotions.index(result)
    probabilities=model.predict_proba(features)[0]
    proba=probabilities[indx]
    per=proba*100
    # show the result
    print('\nSentiment by Facial Expression : ' + FEDans[0].capitalize())
    print('\nSentiment Percentage by Facial Expression : {}%'.format(FEDans[1]))
    print('\nSentiment by NLP : ' + res.capitalize())
    print('\nSentiment Percentage by NLP : {}%'.format(percentage))
    print("\nSentiment by SER : "+result.capitalize())
    print("\nSentiment Percentage by SER : {}%".format(per))
    print("\nEmotion Predicted Successfully!")
  except Exception as e:
    print(e)

print("\n")
