import string
from collections import Counter
import speech_recognition as sr
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize



def NLP():
    data = pd.read_csv(r"NRC-Emotion.txt", sep="\t", header=None, encoding="utf-8", on_bad_lines='skip')

    # Rename the columns
    data.columns = ["word", "emotion", "score"]
    
    sentence = open('read.txt', encoding='utf-8').read()
    lower_case = sentence.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    # Using word_tokenize because it's faster than split()
    tokenized_words = word_tokenize(cleaned_text, "english")

    # Removing Stop Words
    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_words.append(word)


    # Lemmatization - From plural to single + Base form of a word (example better-> good)
    lemma_words = []
    for word in final_words:
        word = WordNetLemmatizer().lemmatize(word)
        lemma_words.append(word)

    # Create a list to store the scores for each emotion
    scores = [0] * 10

    # Define the emotions in the NRC Emotion Lexicon
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]

    # Loop through each word in the sentence
    for word in lemma_words:
        # Filter the data to only include rows with the current word and a score of 1
        filtered_data = data[(data["word"] == word) & (data["score"] == 1)]
        # Loop through each row in the filtered data
        for index, row in filtered_data.iterrows():
            # Increment the score for the appropriate emotion
            if row["emotion"] in ["anger", "disgust", "fear", "negative", "sadness"]:
                scores[emotions.index(row["emotion"])] += 2
            else:
                scores[emotions.index(row["emotion"])] += 1

    # Determine the emotion with the highest score
    max_score = max(scores)
    min_score = min(scores)

    # Check if all emotions have the same score
    if max_score == min_score:
        print("Sentiment by NLP : Neutral")
    else:
        # Create a list to store the emotions with the highest score
        max_emotions = []

        # Loop through each score
        for i, score in enumerate(scores):
            # If the score is equal to the maximum score, add the corresponding emotion to the list
            if score == max_score:
                max_emotions.append(emotions[i])

        # Determine if the sentiment is positive, negative, or neutral
        if "positive" in max_emotions and "negative" not in max_emotions:
            print("Sentiment by NLP : Positive")
        elif "negative" in max_emotions and "positive" not in max_emotions:
            print("Sentiment by NLP : Negative")
        else:
            print("Sentiment by NLP : Mixed")