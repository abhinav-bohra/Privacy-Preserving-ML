#-------------------------------------------------------------------------------------------------------------------
# DSAI ASSIGNMENT 2
# ABHINAV BOHRA 18CS30049
#-------------------------------------------------------------------------------------------------------------------
import os
import math
import time
import numpy as np
import pandas as pd
import phe.encoding
from phe import paillier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def encrypt_vector(vec, public_key):
  encrypted_vector = [public_key.encrypt(ev) for ev in vec]
  return encrypted_vector

def decrypt_vector(vec, private_key):
  decrypted_vector = [private_key.decrypt(x) for x in vec]
  return decrypted_vector

def load_data(input_file, target_feature):
  data = pd.read_csv(input_file).dropna()
  X = data.drop(target_feature, axis=1).values.tolist()
  y = data[target_feature].values.tolist()
  return X, y
  
class Client:

    def __init__(self, key_length):
        public_key, private_key = paillier.generate_paillier_keypair(n_length=key_length)
        self.public_key, self.private_key = public_key, private_key

    def encrypt_data(self, input_file, target_feature):        
        self.X_test, self.y_test = load_data(input_file, target_feature)
        self.X_test_encrypted = [encrypt_vector(x, self.public_key) for x in self.X_test]
        self.y_test_encrypted = encrypt_vector(self.y_test, self.public_key)
        return self.X_test_encrypted, self.y_test_encrypted
    
    def eval(self, encrypted_predictions):
        logits = decrypt_vector(encrypted_predictions, self.private_key)
        y_pred = [1 if l>0 else 0 for l in logits]
        test_accuracy = accuracy_score(self.y_test, y_pred)
        return test_accuracy

class Server:

    def __init__(self, input_file, target_feature):
        self.model = None
        self.X_train, self.y_train = load_data(input_file, target_feature)
        
    def train_model(self,hyperparams):
        svm_model = SVC(kernel=hyperparams['kernel'], C=hyperparams['C'], gamma=hyperparams['gamma'])
        svm_model.fit(self.X_train, self.y_train)
        self.model = svm_model

    def predict(self, X_test_encrypted):
        encrypted_logits = list()
        w = self.model.coef_[0]
        b = self.model.intercept_[0]
        for x in X_test_encrypted:
          score = b
          for i in range(len(x)):
            score += x[i]*w[i]
          encrypted_logits.append(score)
        return encrypted_logits

key_length = 1024
target_feature = "Outcome"
hyperparameters = {'kernel':'linear', 'C':1, 'gamma':'auto'}
server = Server("server/train.csv", target_feature)
svm_model = server.train_model(hyperparameters)
client = Client(key_length)
X_test_encrypted, y_test_encrypted = client.encrypt_data("client/test.csv", target_feature)
encrypted_preds = server.predict(X_test_encrypted)
test_accuracy = client.eval(encrypted_preds)
print(test_accuracy)