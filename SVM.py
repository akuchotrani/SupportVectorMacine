# -*- coding: utf-8 -*-



import csv


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
def main():
    
    
    Read_Data('Iris_Flowers_100.csv')
    pass








################################################################################
if __name__ == "__main__":
    main()
