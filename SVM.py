# -*- coding: utf-8 -*-

import csv
import numpy as np
import random


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
    

    
    #setting user defined parameters for training SVM
    def __init__(self,C,max_iter,epsilon,kernel_type):
        self.C = C     # C is the regularization parameter
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.kernel_type = kernel_type
        self.kernels_available = {
            'linear': self.Linear_Kernel,
            'quadratic': self.Quadratic_Kernel
        }
        
    
    def Train_SVM(self, Input_Flower_Features, Flower_Type):
        
        N_total_samples = Input_Flower_Features.shape[0]
        print('N_total_samples:',N_total_samples)
        
        #we have alpha per sample of training set
        alpha = np.zeros(N_total_samples)
        #print(alpha)
        
        #picking the kernel specified by the user
        self.kernel = self.kernels_available[self.kernel_type]
        
        
        
        iteration_counter = 0
        while (True):
            iteration_counter = iteration_counter + 1
            
            #saving the copy of previous alphas
            alpha_previous = np.copy(alpha)
            
            #Going through all the data samples in one pass
            for sample_index in range(0,N_total_samples):
                #picking a random number from the sample
                i = random.randrange(0,N_total_samples-1)
                j = sample_index
                x_i = Input_Flower_Features[i,:]
                x_j = Input_Flower_Features[j,:]
                y_i = Y_flower[i:]
                y_j = Y_flower[j:]
                
                k_ij = self.kernel(x_i,x_i) + self.kernel(x_j,x_j) - 2*self.kernel(x_i,x_j)
                if k_ij == 0:
                    continue
                
                print("k_ij:",k_ij)
            
            
            #Terminating Condition
            if (iteration_counter == self.max_iter):
                print("Max iteration reached....Stopping the training")
                break
            
            
    
    def Predict_New_Flower():
        pass
        
    
    #Printing out the parameters of SVM
    def print_info(self):
        print("############ PRINT SVM INFO ################")
        print("C:",self.C)
        print("max_iter:",self.max_iter)
        print("epsilon:",self.epsilon)
        print("kernel_type:",self.kernel_type)
        
        
    def Linear_Kernel(self,x1,x2):
        return np.dot(x1,x2.T)

    def Quadratic_Kernel(self,x1,x2):
        return (np.dot(x1,x2.T) ** 2)





################################################################################
def main():
    
    
    Read_Data('Iris_Flowers_100.csv')
    Create_Matrix()
    
    C = 1
    max_iter = 1000
    epsilon = 0.001
    kernel_type = 'linear'
    model = SVM(C,max_iter,epsilon,kernel_type)
    model.Train_SVM(Flower_Feature_Matrix,Y_flower)
 
    #model.print_info()



################################################################################
if __name__ == "__main__":
    main()
