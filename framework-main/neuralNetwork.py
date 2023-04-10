import numpy as np
import pickle
import matplotlib.pyplot as plt
import utils
import activation_functions as af


class NeuralNetwork:

    def __init__(self, X, Y, batch, alpha, iterations, first_layer_len, second_layer_len):

        self.X = X 
        self.Y = Y
        self.batch = batch
        self.alpha = alpha
        self.iterations = iterations
        self.first_layer_len = first_layer_len
        self.second_layer_len = second_layer_len

        self.x = [] # batch input 
        self.y = [] # batch ground_truth
        self.loss = []
        self.accuracy = []
        
        self.initialize_weights()
      

    def initialize_weights(self):

        self.W1 = np.random.randn(self.X.shape[1],self.first_layer_len)
        self.b1 = np.random.randn(self.W1.shape[1],)

        self.W2 = np.random.randn(self.W1.shape[1],self.second_layer_len)
        self.b2 = np.random.randn(self.W2.shape[1],)

        self.W3 = np.random.randn(self.W2.shape[1],self.Y.shape[1])
        self.b3 = np.random.randn(self.W3.shape[1],)
    

    def forward_propagation(self):

        self.z1 = self.x.dot(self.W1) + self.b1
        self.a1 = af.ReLU(self.z1)

        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = af.ReLU(self.z2)

        self.z3 = self.a2.dot(self.W3) + self.b3
        self.a3 = af.softmax(self.z3)

        self.error = self.a3 - self.y


    def back_propagation(self):

        dcost = (1/self.batch) * self.error
        
        DW3 = np.dot(dcost.T, self.a2).T
        db3 = np.sum(dcost, axis = 0)

        DW2 = np.dot((np.dot((dcost), self.W3.T) * af.dReLU(self.z2)).T, self.a1).T
        db2 = np.sum(np.dot((dcost), self.W3.T) * af.dReLU(self.z2), axis = 0)

        DW1 = np.dot((np.dot(np.dot((dcost), self.W3.T) * af.dReLU(self.z2), self.W2.T) * af.dReLU(self.z1)).T, self.x).T
        db1 = np.sum((np.dot(np.dot((dcost), self.W3.T) * af.dReLU(self.z2), self.W2.T) * af.dReLU(self.z1)), axis = 0)

        self.W3 = self.W3 - self.alpha * DW3
        self.b3 = self.b3 - self.alpha * db3

        self.W2 = self.W2 - self.alpha * DW2
        self.b2 = self.b2 - self.alpha * db2

        self.W1 = self.W1 - self.alpha * DW1
        self.b1 = self.b1 - self.alpha * db1
        

    def train(self):
        percentage = self.iterations // 100
        for i in range(self.iterations):
            l = 0   # loss
            acc = 0

            utils.shuffle(self.X, self.Y)

            for batch in range(self.X.shape[0]//self.batch-1):

                start = batch * self.batch
                end = (batch + 1) * self.batch

                self.x = self.X[start:end]
                self.y = self.Y[start:end]

                self.forward_propagation()
                self.back_propagation()
                
                l += np.mean(self.error**2)
                acc += np.count_nonzero(np.argmax(self.a3, axis = 1) == np.argmax(self.y, axis =1 )) / self.batch

            self.loss.append(l/(self.X.shape[0]//self.batch))
            self.accuracy.append(acc * 100 / (self.X.shape[0]//self.batch))

            if i % percentage == 0:
                accuracy = np.count_nonzero(np.argmax(self.a3,axis=1) == np.argmax(self.y,axis=1)) / self.x.shape[0]
                print("Training: ", i//percentage, "% .    Accuracy: ", 100 * accuracy, "%")

        
    def validation(self,x_validation,y_validation):

        self.x = x_validation
        self.y = y_validation

        self.forward_propagation()
        acc = np.count_nonzero(np.argmax(self.a3,axis=1) == np.argmax(self.y,axis=1)) / self.x.shape[0]

        print("Accuracy:", 100 * acc, "%")


    def predict(self, X):

        X = X.reshape(1, -1)

        z1 = X.dot(self.W1) + self.b1
        a1 = af.ReLU(z1)

        z2 = a1.dot(self.W2) + self.b2
        a2 = af.ReLU(z2)

        z3 = a2.dot(self.W3) + self.b3
        a3 = af.softmax(z3)

        return np.argmax(a3, axis=1)[0]


    def getConfusionMatrix(self, X, Y):
        confusion_m = np.zeros((Y.shape[1],Y.shape[1])).astype(int)
        Y = np.argmax(Y,axis=1)
        for i in range(len(X)):
            confusion_m[X[i]][Y[i]] += 1
        return confusion_m


    def plot_learning(self):
        plt.figure(dpi = 125)
        plt.plot(self.loss)
        plt.title("Evolution of mse during training")
        plt.xlabel("iterations")
        plt.ylabel("loss")


    def plot_accuracy(self):
        plt.figure(dpi = 125)
        plt.plot(self.accuracy)
        plt.title("Evolution of accuracy during training")
        plt.xlabel("iterations")
        plt.ylabel("accuracy")

    def save(self, path):

        self.X = []
        self.Y = []

        self.x = []
        self.y = []

        self.z1 = []
        self.a1 = []

        self.z2 = []
        self.a2 = []

        self.z3 = []
        self.a3 = []

        self.error = []

        with open(path, 'wb') as f:
            pickle.dump(self, f)