import os
import pickle
import speech_recognition as sr
from NLP import NLP
from SER import trainModel
from utils import extract_feature

# if model is not trained train the model
if not os.path.exists("SER-Trained-Model.model"):
  trainModel()

# load the saved model (after training)
model = pickle.load(open("SER-Trained-Model.model", "rb"))

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


if os.path.exists(filename) and os.path.exists("read.txt"):
  try:
    print("\nPrediciting Emotion...")
    
    NLP()

    # extract features and reshape it
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    result = model.predict(features)[0]
    # show the result
    print("Sentiment by SER : ",result.capitalize())
    print("Emotion Predicted Successfully!")
  except Exception as e:
    print("Error")

print("\n")
