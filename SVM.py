# -*- coding: utf-8 -*-

import csv
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

x1_sepal_length = []
x2_sepal_width = []
x3_petal_length = []
x4_petal_width = []
Y_flower = []

################################################################################
def Read_Data(file_name):
        #getting the system path

    CSV_Data = csv.reader(open(file_name, newline=''))
    
    isHeading = True
    
    for row in CSV_Data:
        #skip the first row of csv file
        if isHeading == True:
            isHeading = False
            continue
        
        x1_sepal_length.append(float(row[1]))
        x2_sepal_width.append(float(row[2]))
        x3_petal_length.append(float(row[3]))
        x4_petal_width.append(float(row[4]))
        
        flower_type = 0
        if(row[5] == 'Iris-setosa'):
            flower_type = 1
        if(row[5] == 'Iris-versicolor'):
            flower_type = -1
        Y_flower.append(flower_type)
        

################################################################################     
def Create_Matrix():
    
    Array_Sepal_length = np.array(x1_sepal_length)
    Array_Sepal_width = np.array(x2_sepal_width)
    Array_Petal_length = np.array(x3_petal_length)
    Array_Petal_width = np.array(x4_petal_width)

    global Flower_Feature_Matrix
    Flower_Feature_Matrix =  np.column_stack((Array_Sepal_length,Array_Sepal_width,Array_Petal_length,Array_Petal_width))
    

################################################################################
################################################################################
          
class SVM():
    
    def __init__(self, max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001):
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
    def fit(self, X, y):
        # n : number of samples (100)
        n = X.shape[0]
        
        #we have alpha per sample of training set. Initially set to zeros
        alpha = np.zeros((n))
        
        #pick the kernel user selected
        kernel = self.kernels[self.kernel_type]
        
        
        iteration = 0
        while True:
            iteration += 1
            
            #saving the copy of alpha from previous iteration
            alpha_prev = np.copy(alpha)
            
            #going through all the samples in one iteration
            for j in range(0, n):
                
                #selcting random sample index where i is not equal to j
                i = self.get_rnd_int(0, n-1, j) # Get random int i~=j
                
                x_i = X[i,:]
                x_j = X[j,:]
                y_i = y[i]
                y_j = y[j]
                
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
                
                if k_ij == 0:
                    continue
                
                #select alpha of i and j from the alpha array to calculate L and H
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            # Terminating condition: reacing convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            #Terminating condition: Reaching max iterations
            if iteration >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                return
            
        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
            
            
    def predict(self, X):
        return self.h(X, self.w, self.b)
    
    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)
    
    def calc_w(self, alpha, y, X):
        return np.dot(alpha * y, X)
    
    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)
    
    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k
    
    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))
        
    def get_rnd_int(self, a,b,z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = rnd.randint(a,b)
            cnt=cnt+1
        return i
    
    # Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)
    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)
    
    #Printing out the parameters of SVM
    def print_info(self):
        print("############ PRINT SVM INFO ################")
        print("C:",self.C)
        print("max_iter:",self.max_iter)
        print("epsilon:",self.epsilon)
        print("kernel_type:",self.kernel_type)


################################################################################
def calc_acc(y, y_hat):
    
    correct_counter = 0
    for i in range(0,len(y)):
        if(y[i] == -1 and y_hat[i] == -1):
            correct_counter = correct_counter + 1
        if(y[i] == 1 and y_hat[i] == 1):
            correct_counter = correct_counter + 1
    
    return (correct_counter/len(y))
        

################################################################################
def main():
    
    
    Read_Data('Iris_Flowers_100.csv')
    Create_Matrix()
    
    C = 1
    max_iter = 1000
    epsilon = 0.001
    kernel_type = 'linear'
    
    model = SVM(max_iter,kernel_type,C,epsilon)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(Flower_Feature_Matrix, Y_flower, test_size = 0.2, random_state = 0)
    
    global support_vectors

    model.fit(X_train,y_train)

    # Make prediction
    y_hat = model.predict(X_test)

    # Calculate accuracy
    accuracy = calc_acc(y_test, y_hat)
    
    model.print_info()
    print("\nAccuracy:",accuracy)

    for i in range(0,len(y_train)):
        if(y_train[i] == 1):
            plt.plot(X_train[i,0],X_train[i,1],'r*')
        else:
            plt.plot(X_train[i,0],X_train[i,1],'b*')
    plt.title("Training Set")

    plt.show()
    
    
    for i in range(0,len(y_hat)):
        if(y_hat[i] == 1):
            plt.plot(X_test[i,0],X_test[i,1],'r*')
        else:
            plt.plot(X_test[i,0],X_test[i,1],'b*')
    plt.title("Test Set Result")

    plt.show()



################################################################################
if __name__ == "__main__":
    main()
