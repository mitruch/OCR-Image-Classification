import sys

import pickle
import operator
import struct
import itertools
import matplotlib.pyplot as plt
import numpy as np

from myKnn import myKnn
from mySvm import mySvm
from myMlp import myMlp

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix

class Main():

    def save_model(self,model, filepath):
        pickle.dump(model, open(filepath, 'wb'))

    def load_model(self, filepath):
        model = pickle.load(open(filepath, 'rb'))
        return model        

    def write_stat(self, file, title, clf_name, y_true, y_pred, trainTime=0, testTime=0, 
                    trainTime_ar=None, testTime_ar=None, acc_ar=None, params=" ", norm=True, lang='PL'):

        if acc_ar is not None:
            acc_ar.append(acc)
        if trainTime_ar is not None:
            trainTime_ar.append(trainTime)
        if testTime_ar is not None:
            testTime_ar.append(testTime)

        if lang == 'PL':
            file.write(title + "\n")
            file.write("\nKlasyfikator: " + clf_name + "\n")
            file.write("\nParametry: \n")
            file.write(str(params))
            acc = accuracy_score(y_true, y_pred)
            file.write("\n\nDokladnosc: {}".format(acc) + "\n")
            file.write("\nCzas treningu: {}".format(trainTime))
            file.write("\nCzas testowania: {}".format(testTime) + "\n")

            file.write("\nMiary jakosci klasyfikacji: \n")
            file.write("\n              precyzja     czulosc     f1      probki\n")
            file.write(metrics.classification_report(y_true, y_pred))

            file.write("\nMacierz pomylek: \n\n")
            cm = metrics.confusion_matrix(y_true, y_pred)
            file.write(np.array2string(cm, separator=', '))

            if norm:
                file.write("\n\nZnormalizowana macierz pomylek: \n")
                cm = metrics.confusion_matrix(y_true, y_pred)
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                file.write(np.array2string(cm, separator=', '))
                
        else: # ENG
            file.write(title + "\n")
            file.write("\nClassifier: " + clf_name + "\n")
            file.write("\nParams: \n")
            file.write(params)
            acc = accuracy_score(y_true, y_pred)
            file.write("\n\nAccuracy: {}".format(acc) + "\n")
            file.write("\nTrain time: {}".format(trainTime))
            file.write("\nTest time: {}".format(testTime) + "\n")

            file.write("\nClassification metrix: \n")
            file.write(metrics.classification_report(y_true, y_pred))

            file.write("\nConfusion matrix: \n")
            cm = metrics.confusion_matrix(y_true, y_pred)
            file.write(np.array2string(cm, separator=', '))

            if norm:
                file.write("\n\nNormalized confusion matrix: \n")
                cm = metrics.confusion_matrix(y_true, y_pred)
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                file.write(np.array2string(cm, separator=', '))
            
        file.write("\n##########################\n")
        

    def print_stat(self, title, clf_name, y_true, y_pred, trainTime=0, testTime=0, 
                    trainTime_ar=None, testTime_ar=None, acc_ar=None, params=" ", norm=True, lang='PL'):

        if lang == 'PL':
            print(title + "\n")
            print("\nKlasyfikator: " + clf_name + "\n")
            print("\nParametry: ")
            print(params)
            acc = accuracy_score(y_true, y_pred)
            print("\nDokladnosc: {}".format(acc) + "\n")
            print("\nCzas treningu: {}".format(trainTime) + "\n")
            print("\nCzas testowania: {}".format(testTime) + "\n")

            print("\nMiary jakosci klasyfikacji: ")
            print("\n              precyzja     czulosc     f1      probki\n")
            print(metrics.classification_report(y_true, y_pred))

            print("\nRozpoczecie rysowania macierzy pomylek\n")
            cm = metrics.confusion_matrix(y_true, y_pred)
            self.plot_confusion_matrix(cm, [x for x in range (0, 10)], False, True, True, fileName=title + "CM.png")
            print(np.array2string(cm, separator=', '))
            print("\nKoniec rysowania macierzy pomylek\n")

            print("\nRozpoczecie rysowania znormalizowanej macierzy pomylek\n")
            cm = metrics.confusion_matrix(y_true, y_pred)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            self.plot_confusion_matrix(cm, [x for x in range (0, 10)], True, True, True, fileName=title + "CM.png")
            print(np.array2string(cm, separator=', '))
            print("\nKoniec rysowania znormalizowanej macierzy pomylek\n")

        else:
            print(title + "\n")
            print("\nClassifier: " + clf_name + "\n")
            print("\nParams: ")
            print(params)
            acc = accuracy_score(y_true, y_pred)
            print("\nAccuracy: {}".format(acc) + "\n")
            print("\nTrain time: {}".format(trainTime) + "\n")
            print("\nTest time: {}".format(testTime) + "\n")

            print("\nClassification metrix: \n")
            print(metrics.classification_report(y_true, y_pred))

            print("\nStart ploting confusion matrix\n")
            cm = metrics.confusion_matrix(y_true, y_pred)
            self.plot_confusion_matrix(cm, [x for x in range (0, 10)], False, True, True, fileName=title + "CM.png")
            print(np.array2string(cm, separator=', '))
            print("\nEnd ploting confusion matrix\n")

            print("\nStart ploting normalized confusion matrix\n")
            cm = metrics.confusion_matrix(y_true, y_pred)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            self.plot_confusion_matrix(cm, [x for x in range (0, 10)], True, True, True, fileName=title + "CM.png")
            print(np.array2string(cm, separator=', '))
            print("\nEnd ploting normalized confusion matrix\n")

        print("\n##########################\n")
    
    def plot_confusion_matrix(self, cm, classes,
                            normalize=False,
                            show=True, save=True, fileName='cm.png',
                            title='Confusion matrix',
                            cmap=plt.cm.Greys):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fileName = "norm" + fileName 

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save:
            plt.savefig("images/" + fileName)
        if show:
            plt.show()


    def configMNIST(self):
        
        path_tr_data = r"dataset\MNIST\train-images-idx3-ubyte"
        path_tr_label = r"dataset\MNIST\train-labels-idx1-ubyte"
        path_test_data = r"dataset\MNIST\t10k-images-idx3-ubyte"
        path_test_label = r"dataset\MNIST\t10k-labels-idx1-ubyte"


        raw_train = self.read_idx(path_tr_data)
        train_data = np.reshape(raw_train, (60000, 28*28))
        train_label = self.read_idx(path_tr_label)

        raw_test = self.read_idx(path_test_data)
        test_data = np.reshape(raw_test, (10000, 28*28))
        test_label = self.read_idx(path_test_label)
        return train_data, train_label, test_data, test_label

    def configEMNIST(self):
        
        path_tr_data = r"dataset\EMNIST\emnist-digits-train-images-idx3-ubyte"
        path_tr_label = r"dataset\EMNIST\emnist-digits-train-labels-idx1-ubyte"
        path_test_data = r"dataset\EMNIST\emnist-digits-test-images-idx3-ubyte"
        path_test_label = r"dataset\EMNIST\emnist-digits-test-labels-idx1-ubyte"

        raw_train = self.read_idx(path_tr_data)
        train_data = np.reshape(raw_train, (240000, 28*28))
        train_label = self.read_idx(path_tr_label)

        raw_test = self.read_idx(path_test_data)
        test_data = np.reshape(raw_test, (40000, 28*28))
        test_label = self.read_idx(path_test_label)

    def read_idx(self, filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

    def findBestK(self, dataset, start_k, end_k, fileName="best_k_EMNIST.txt"):
        acc_ar, trainTime_ar, testTime_ar = [], [], []
        clf_knn = None
        test_name = " "
        if dataset == 1:
            test_name = "MNIST_test_1_1_"
        if dataset == 2:
            test_name = "EMNIST_test_1_1_"
        for k in range(start_k, end_k):
            clf_knn = start([dataset, 0, k, "stat_files/knn_stat/" + fileName, test_name + str(k)], show=False, save=True)
            acc = accuracy_score(clf_knn.getTestLabel(), clf_knn.getPredictLabel())
            acc_ar.append(acc)
            trainTime_ar.append(clf_knn.getTrainTime())
            testTime_ar.append(clf_knn.getTestTime())

        with open("stat_files/knn_stat/winners_" + fileName, 'a') as stat_file:
            stat_file.write("\nK range: [{}, {}) \n".format(start_k, end_k))
            stat_file.write("\nAccuracy table: \n")
            for ac in acc_ar:
                stat_file.write("{}, ".format(ac))
                
            stat_file.write("\nTraining times table: \n")
            for tr in trainTime_ar:
                stat_file.write("{}, ".format(tr))
                
            stat_file.write("\nTesting times table: \n")
            for ts in testTime_ar:
                stat_file.write("{}, ".format(ts))

            k_index = [idx for idx in range (start_k, end_k)]
            index_acc_w, win_acc = max(enumerate(acc_ar), key=operator.itemgetter(1))
            index_train_w, win_train_time = min(enumerate(trainTime_ar), key=operator.itemgetter(1))
            index_test_w, win_test_time = min(enumerate(testTime_ar), key=operator.itemgetter(1))
            
            stat_file.write("\n\nBEST ACCURACY WINNER: \n")
            stat_file.write("\nK = {} \n".format(k_index[index_acc_w]))
            stat_file.write("\nAccuracy = {} \n".format(acc_ar[index_acc_w]))
            stat_file.write("\nTrain time = {} \n".format(trainTime_ar[index_acc_w]))
            stat_file.write("\nTest time = {} \n".format(testTime_ar[index_test_w]))

            stat_file.write("\n\nBEST TRAIN TIME WINNER: \n")
            stat_file.write("\nK = {} \n".format(k_index[index_train_w]))
            stat_file.write("\nAccuracy = {} \n".format(acc_ar[index_train_w]))
            stat_file.write("\nTrain time = {} \n".format(trainTime_ar[index_train_w]))
            stat_file.write("\nTest time = {} \n".format(testTime_ar[index_train_w]))
            
            stat_file.write("\n\nBEST TESTING TIME WINNER: \n")
            stat_file.write("\nK = {} \n".format(k_index[index_test_w]))
            stat_file.write("\nAccuracy = {} \n".format(acc_ar[index_test_w]))
            stat_file.write("\nTrain time = {} \n".format(trainTime_ar[index_test_w]))
            stat_file.write("\nTest time = {} \n".format(testTime_ar[index_test_w]))


    def start(self, argv, show=True, save=True):

        train_data, train_label, test_data, test_label = None, None, None, None
        clf = None
        clf_name = " "

        dataset = int(argv[0])
        algorithm = int(argv[1])
        config = int(argv[2])
        file_path = argv[3]
        test_name = argv[4]

        # Ustawienie zbioru danych
        if dataset == 1: # MNIST
            train_data, train_label, test_data, test_label = self.configMNIST()
        if dataset == 2: # EMNIST
            train_data, train_label, test_data, test_label = self.configEMNIST()
        
        # Ustawienie algorytmu
        if algorithm == 1: # KNN
            clf = myKnn(train_data, train_label, test_data, test_label)
            clf_name = "KNN"
            clf.setK(config)
        if algorithm == 2: # SVM
            clf_name = "SVM"
            clf = mySvm(train_data, train_label, test_data, test_label)
            clf.setKernel(config)
        if algorithm == 3: # MLP
            clf_name = "MLP"
            clf = myMlp(train_data, train_label, test_data, test_label)
            clf.setActivationFunction(config)

        # Trenowanie 
        clf.train()
        # Testowanie
        clf.test()

        clf.getTypes()
        # Zapis statystyk
        if save:
            self.save_model(clf, "clf_model_" + test_name + ".pkl")
            with open(file_path, 'a') as output_file:
                self.write_stat(output_file, test_name, clf_name, test_label, clf.getPredictLabel(), clf.getTrainTime(), clf.getTestTime(), params=clf.getParams(), lang='PL')
        if show:
            self.print_stat(test_name, clf_name, test_label, clf.getPredictLabel(), clf.getTrainTime(), clf.getTestTime(), params=clf.getParams(), lang='PL')

        return clf

    def runAllMNIST(self, show=True, save=True):
        # MNIST DATASET
        # KNN
        self.start([1, 1, 3, "MNIST_stat.txt", "MNIST_test_1_1_3"], show, save) # with the best K
        # SVM
        self.start([1, 2, 1, "MNIST_stat.txt", "MNIST_test_1_2_1"], show, save)
        self.start([1, 2, 2, "MNIST_stat.txt", "MNIST_test_1_2_2"], show, save)
        self.start([1, 2, 3, "MNIST_stat.txt", "MNIST_test_1_2_3"], show, save)
        # MLP
        self.start([1, 3, 1, "MNIST_stat.txt", "MNIST_test_1_3_1"], show, save)
        self.start([1, 3, 2, "MNIST_stat.txt", "MNIST_test_1_3_2"], show, save)
        self.start([1, 3, 3, "MNIST_stat.txt", "MNIST_test_1_3_3"], show, save)

    def runAllEMNIST(self, show=True, save=True):
        # MNIST DATASET
        # KNN
        self.start([2, 1, 3, "EMNIST_stat.txt", "EMNIST_test_2_1_3"], show, save) # with the best K
        # SVM
        self.start([2, 2, 1, "EMNIST_stat.txt", "EMNIST_test_2_2_1"], show, save)
        self.start([2, 2, 2, "EMNIST_stat.txt", "EMNIST_test_2_2_2"], show, save)
        self.start([2, 2, 3, "EMNIST_stat.txt", "EMNIST_test_2_2_3"], show, save)
        # MLP
        self.start([2, 3, 1, "EMNIST_stat.txt", "EMNIST_test_2_3_1"], show, save)
        self.start([2, 3, 2, "EMNIST_stat.txt", "EMNIST_test_2_3_2"], show, save)
        self.start([2, 3, 3, "EMNIST_stat.txt", "EMNIST_test_2_3_3"], show, save)

    def runAll(self, show=False, save=True):
        self.runAllMNIST(show, save)
        self.runAllEMNIST(show, save)

    def main(self):
        
        show = False
        save = True

        if 'show' in sys.argv:
            show = True

        if 'load' in sys.argv:
            clf = self.load_model(sys.argv[2])
            train_data, train_label, test_data, test_label = None, None, None, None
            if show:
                self.print_stat("Loaded model", " ", clf.getTestLabel(), clf.getPredictLabel(), clf.getTrainTime(), clf.getTestTime(), params=clf.getParams(), lang='PL')

            if 'write' in sys.argv:
                with open(sys.argv[4], 'a') as output_file:
                    self.write_stat(output_file, "Loaded model", " ", clf.getTestLabel(), clf.getPredictLabel(), clf.getTrainTime(), clf.getTestTime(), params=clf.getParams(), lang='PL')
                    
            return

        if 'runall' in sys.argv:
            if '1' in sys.argv:
                self.runAllMNIST(show, save)
            if '2' in sys.argv:
                self.runAllEMNIST(show, save)
            else:
                self.runAll(show, save)
        else:
            self.start(sys.argv[1::], show, save)


if __name__ == "__main__":
    app = Main()
    app.main()
    