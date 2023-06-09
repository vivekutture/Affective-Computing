import pickle # to save model after training
import os
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import accuracy_score # to measure how good we are

from utils import load_data

def trainModel():

    print("[*] Training the model...")
    
    # load RAVDESS dataset, 75% training 25% testing
    X_train, X_test, y_train, y_test = load_data(test_size=0.25)

    # print some details
    # number of samples in training data
    print("[-] Number of training samples:", X_train.shape[0])
    # number of samples in testing data
    print("[-] Number of testing samples:", X_test.shape[0])
    # number of features used
    # this is a vector of features extracted 
    # using extract_features() function
    print("[-] Number of features:", X_train.shape[1])


    # best model, determined by a grid search
    model_params = {
        'alpha': 0.01,
        'batch_size': 256,
        'epsilon': 1e-08, 
        'hidden_layer_sizes': (300,), 
        'learning_rate': 'adaptive', 
        'max_iter': 500, 
    }

    # initialize Multi Layer Perceptron classifier
    # with best parameters ( so far )
    model = MLPClassifier(**model_params)

    # train the model
    
    model.fit(X_train, y_train)

    # predict 25% of data to measure how good we are
    y_pred = model.predict(X_test)

    print("[*] Training completed!")

    # calculate the accuracy
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    print("Accuracy: {:.2f}%".format(accuracy*100))

    pickle.dump(model, open("SER-Trained-Model.pkl", "wb"))