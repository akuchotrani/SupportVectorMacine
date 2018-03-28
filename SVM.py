# -*- coding: utf-8 -*-

import csv
import numpy as np


x1_sepal_length = []
x2_sepal_width = []
x3_petal_length = []
x4_petal_width = []

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
    

################################################################################
################################################################################
class SVM():
    

    
    #setting user defined parameters for training SVM
    def __init__(self,C,max_iter,epsilon,kernel_type):
        self.C = C     # C is the regularization parameter
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.kernel_type = kernel_type
        
    
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
    
    C = 1
    max_iter = 1000
    epsilon = 0.001
    kernel_type = 'linear'
    model = SVM(C,max_iter,epsilon,kernel_type)
 
    model.print_info()








################################################################################
if __name__ == "__main__":
    main()
