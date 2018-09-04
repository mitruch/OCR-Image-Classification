from sklearn.svm import LinearSVC
import sklearn.svm as svm
import time

class mySvm():
    def __init__(self, train_data, train_label, test_data, test_label):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.predict_label = None
        self.train_time = 0
        self.test_time = 0
        self.clf = None

    def setKernel(self, kernel = 1):
        if kernel == 1:
            self.clf = svm.SVC(kernel='linear')
        elif kernel == 2:
            self.clf = svm.SVC(kernel='poly')      
        elif kernel == 3:
            self.clf = svm.SVC(kernel='rbf') 
    
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
    
