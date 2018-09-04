from sklearn.neural_network import MLPClassifier
import time

class myMlp():
    def __init__(self, train_data, train_label, test_data, test_label):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.predict_label = None
        self.train_time = 0
        self.test_time = 0
        self.clf = None

    def setActivationFunction(self, fun = 0):
        if fun == 1: # relu
            self.clf = MLPClassifier(hidden_layer_sizes=(100,50,), activation='relu',solver='adam',alpha=0.0001,max_iter=300 )
        elif fun == 2: # tanh
            self.clf = MLPClassifier(hidden_layer_sizes=(100,50,), activation='tanh',solver='adam',alpha=0.0001,max_iter=300 )
        elif fun == 3: # identity 
            self.clf = MLPClassifier(hidden_layer_sizes=(100,50,), activation='identity',solver='adam',alpha=0.0001,max_iter=300 )
         
    def train(self):
        print("Start train")
        time_start = time.time()
        self.clf.fit(self.train_data, self.train_label)
        time_end = time.time() - time_start
        print("End train", time_end)
        self.train_time = time_end
        return self.train_time

    def test(self):
        print("Start test")
        time_start = time.time()
        self.predict_label = self.clf.predict(self.test_data)
        time_end = time.time() - time_start
        print("End test", time_end)
        self.test_time = time_end
        return self.test_label, self.test_time

    def getTestLabel(self):
        return self.test_label

    def getPredictLabel(self):
        return self.predict_label
    
    def getTrainTime(self):
        return self.train_time

    def getTestTime(self):
        return self.test_time
    
    def getParams(self):
        return self.clf.get_params()