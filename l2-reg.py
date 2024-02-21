# Jan C. Bierowiec
# CISC 5790: Data Mining
# Assignment 1
# 02/20/2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

# Define the current working directory and plots directory
current_directory = '/Users/janbierowiec/Data-Mining-Project-1'
plots_directory = os.path.join(current_directory, 'Plots')

# Ensure the plots directory exists
if not os.path.exists(plots_directory):
    os.makedirs(plots_directory)

# Move all .png files to the plots directory
for file_name in os.listdir(current_directory):
    if file_name.endswith('.png'):
        source_path = os.path.join(current_directory, file_name)
        destination_path = os.path.join(plots_directory, file_name)
        try:
            shutil.move(source_path, destination_path)
            print(f"Moved: {file_name}")
        except Exception as e:
            print(f"Error moving {file_name}: {e}")
        
# test data
Test1 = pd.read_csv("Data/TestData/test-100-10.csv")
Test2 = pd.read_csv("Data/TestData/test-100-100.csv")
Test3 = pd.read_csv("Data/TestData/test-1000-100.csv")

# train data
Train1 = pd.read_csv("Data/TrainData/train-100-10.csv")
Train1 = Train1.drop(columns=["Unnamed: 11", "Unnamed: 12"])
Train2 = pd.read_csv("Data/TrainData/train-100-100.csv")
Train3 = pd.read_csv("Data/TrainData/train-1000-100.csv")
Train4 = pd.read_csv('Data/TrainData/train-50(1000)-100.csv')
Train5 = pd.read_csv('Data/TrainData/train-100(1000)-100.csv')
Train6 = pd.read_csv('Data/TrainData/train-150(1000)-100.csv')

# function to change the data into matrix form
def Data_to_Matrix(data):
    data.insert(0, '1', 1)
    XTrain = data.iloc[:,:-1]
    XArray = XTrain.values
    YTrain = data.iloc[:,-1:]
    YArray = YTrain.values
    return(XArray, YArray)

# creating x & y componets of the data converted into matricies to then perform L2 Regularization
Xtrain1, Ytrain1 = Data_to_Matrix(Train1)
Xtrain2, Ytrain2 = Data_to_Matrix(Train2)
Xtrain3, Ytrain3 = Data_to_Matrix(Train3)

Xtest1, Ytest1 = Data_to_Matrix(Test1)
Xtest2, Ytest2 = Data_to_Matrix(Test2)
Xtest3, Ytest3 = Data_to_Matrix(Test3)

Xtrain4, Ytrain4 = Data_to_Matrix(Train4)
Xtrain5, Ytrain5 = Data_to_Matrix(Train5)
Xtrain6, Ytrain6 = Data_to_Matrix(Train6)

# L2 Regularization function
def L2_Regularization(XTrain, YTrain, start, end):
    values = np.arange(start, end + 1, 1)
    Weights = []

    for num in values:
        X_transpose = np.transpose(XTrain)
        XTX = np.dot(X_transpose, XTrain)
        I = np.identity(len(XTX))
        XTX_lambda_I = XTX + np.dot(num, I)
        XTX_lambda_I_inverse = np.linalg.inv(XTX_lambda_I)
        XTY = np.dot(X_transpose, YTrain)
        W = np.dot(XTX_lambda_I_inverse, XTY)
        Weights.append(W.flatten())
        FinalDatasetWeight = np.transpose(np.array(Weights))

    return(FinalDatasetWeight)

# training weight values
WTrain1 = L2_Regularization(Xtrain1, Ytrain1, 0, 150)
WTrain2 = L2_Regularization(Xtrain2, Ytrain2, 0, 150)
WTrain3 = L2_Regularization(Xtrain3, Ytrain3, 0, 150)
WTrain4 = L2_Regularization(Xtrain4, Ytrain4, 0, 150)
WTrain5 = L2_Regularization(Xtrain5, Ytrain5, 0, 150)
WTrain6 = L2_Regularization(Xtrain6, Ytrain6, 0, 150)

# Mean Square Error calculation
def MSE(XTrain, YTrain, W):
    Y_predict = np.dot(XTrain, W)
    sum = 0.0

    for i in range(len(YTrain)):
        YpredictedError = Y_predict[i] - YTrain[i]
        sum += (YpredictedError ** 2)

    mean_squared_error = sum / float(len(YTrain))
    
    return mean_squared_error

MSE(Xtrain1, Ytrain1, WTrain1)
MSE(Xtest1, Ytest1, WTrain1)
MSE(Xtrain2, Ytrain2, WTrain2)
MSE(Xtest2, Ytest2, WTrain2)
MSE(Xtrain3, Ytrain3, WTrain3)
MSE(Xtest3, Ytest3, WTrain3)

# definition to plot the first six plots comparing the test to the train data using L2 Regularization
def plot_MSE_for_datasets(train_datasets, test_datasets, Weights, titles, lambdas):
    for i, (train_data, test_data, w) in enumerate(zip(train_datasets, test_datasets, Weights)):
        MSETrain = MSE(train_data[0], train_data[1], w)
        MSETest = MSE(test_data[0], test_data[1], w)
        
        plt.figure(i+1)
        plt.plot(range(lambdas[0], lambdas[1]+1), MSETrain, label="Training MSE")
        plt.plot(range(lambdas[0], lambdas[1]+1), MSETest, label="Testing MSE")
        plt.title(titles[i])
        plt.xlabel("Lambda")
        plt.ylabel("MSE")
        plt.legend()
        filename = titles[i].replace(" ", "_") + "_lambda_=[0-150].png"
        plt.savefig(filename)
        plt.close()

# Lists of training and testing datasets and their Weights
train_datasets = [(Xtrain1, Ytrain1), (Xtrain2, Ytrain2), (Xtrain3, Ytrain3), (Xtrain4, Ytrain4), (Xtrain5, Ytrain5), (Xtrain6, Ytrain6)]
test_datasets = [(Xtest1, Ytest1), (Xtest2, Ytest2), (Xtest3, Ytest3), (Xtest3, Ytest3), (Xtest3, Ytest3), (Xtest3, Ytest3)]
Weights = [WTrain1, WTrain2, WTrain3, WTrain4, WTrain5, WTrain6]
titles = ["Train-100-10 vs Test-100-10", "Train-100-100 vs Test-100-100", "Train-1000-100 vs Test-1000-100", "Train-50(1000)-100 vs Test-1000-100", "Train-100(1000)-100 vs Test-1000-100", "Train-150(1000)-100 vs Test-1000-100"]

# Lambda range
lambdas = (0, 150)

# Call the function
plot_MSE_for_datasets(train_datasets, test_datasets, Weights, titles, lambdas)

MSE_1 = MSE(Xtest1, Ytest1, WTrain1)
MSE_2 = MSE(Xtest2, Ytest2, WTrain2)
MSE_3 = MSE(Xtest3, Ytest3, WTrain3)
MSE_4 = MSE(Xtest3, Ytest3, WTrain4)
MSE_5 = MSE(Xtest3, Ytest3, WTrain5)
MSE_6 = MSE(Xtest3, Ytest3, WTrain6)

MSE_values = [MSE_1, MSE_2, MSE_3, MSE_4, MSE_5, MSE_6]

for value in MSE_values:
    lowest_lambda = value.argmin()
    least_MSE = value[lowest_lambda]

    print("The lowest lambda =", lowest_lambda, "gives the least MSE =", round(least_MSE, 3))

# weights ranging from 1-150
W1_1_150 = L2_Regularization(Xtrain2, Ytrain2, 1, 150)
W2_1_150 = L2_Regularization(Xtrain4, Ytrain4, 1, 150)
W3_1_150 = L2_Regularization(Xtrain5, Ytrain5, 1, 150)

# plots of the training and test data ranging from 1 to 150
plt.figure(7)
plt.plot(MSE(Xtrain2, Ytrain2, W1_1_150), label="Training MSE")
plt.plot(MSE(Xtest2, Ytest2, W1_1_150), label="Testing MSE")
plt.title("Train-100-100 vs Test-100-100 Plot")
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.legend()
plt.savefig("Train-100-100 vs Test-100-100 Plot lambda = [1-150].png")

plt.figure(8)
plt.plot(MSE(Xtrain4, Ytrain4, W2_1_150), label="Training MSE")
plt.plot(MSE(Xtest3, Ytest3, W2_1_150), label="Testing MSE")
plt.title("Train-50(1000)-100 vs Test-1000-100 Plot")
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.legend()
plt.savefig("Train-50(1000)-100 vs Test-1000-100 Plot lambda = [1-150].png")

plt.figure(9)
plt.plot(MSE(Xtrain5, Ytrain5, W3_1_150), label="Training MSE")
plt.plot(MSE(Xtest3, Ytest3, W3_1_150), label="Testing MSE")
plt.title("Train-100(1000)-100 vs Test-1000-100 Plot")
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.legend()
plt.savefig("Train-100(1000)-100 vs Test-1000-100 Plot lambda = [1-150].png")

# Cross Validation code below

Fold1 = int(len(Ytrain1) / 10)

MSE_sum_1 = 0

for i in range(10):
    X1TestFold = Xtrain1[ i*Fold1 : (i+1)*Fold1 ]
    Y1TestFold = Ytrain1[ i*Fold1 : (i+1)*Fold1 ]

    X1TrainFold = np.concatenate((Xtrain1[ : i*Fold1], Xtrain1[ (i+1)*Fold1 : ]), axis=0)
    Y1TrainFold = np.concatenate((Ytrain1[ : i*Fold1], Ytrain1[ (i+1)*Fold1 : ]), axis=0)

    Weight1 = L2_Regularization(X1TrainFold, Y1TrainFold, 0, 150)

    MSE_sum_1 += MSE(X1TestFold, Y1TestFold, Weight1)

MSETest1 = MSE_sum_1/10

Lambda1 = MSETest1.argmin()
MSE1 = MSETest1[Lambda1]

print("For dataset 100-10, the lambda =", Lambda1, ", which gives the least test MSE of", round(MSE1, 3))

Fold2 = int(len(Ytrain2) / 10)

MSE_sum_2 = 0

for i in range(10):
    X2TestFold = Xtrain2[ i*Fold2 : (i+1)*Fold2 ]
    Y2TestFold = Ytrain2[ i*Fold2 : (i+1)*Fold2 ]

    X2TrainFold = np.concatenate((Xtrain2[ : i*Fold2], Xtrain2[ (i+1)*Fold2 : ]), axis=0)
    Y2TrainFold = np.concatenate((Ytrain2[ : i*Fold2], Ytrain2[ (i+1)*Fold2 : ]), axis=0)

    Weight2 = L2_Regularization(X2TrainFold, Y2TrainFold, 0, 150)

    MSE_sum_2 += MSE(X2TestFold, Y2TestFold, Weight2)

MSETest2 = MSE_sum_2/10

Lambda2 = MSETest2.argmin()
MSE2 = MSETest2[Lambda2]

print("For dataset 100-100, the lambda =", Lambda2, ", which gives the least test MSE of", round(MSE2, 3))

Fold3 = int(len(Ytrain3) / 10)

MSE_sum_3 = 0

for i in range(10):
    X3TestFold = Xtrain3[ i*Fold3 : (i+1)*Fold3 ]
    Y3TestFold = Ytrain3[ i*Fold3 : (i+1)*Fold3 ]

    X3TrainFold = np.concatenate((Xtrain3[ : i*Fold3], Xtrain3[ (i+1)*Fold3 : ]), axis=0)
    Y3TrainFold = np.concatenate((Ytrain3[ : i*Fold3], Ytrain3[ (i+1)*Fold3 : ]), axis=0)

    Weight3 = L2_Regularization(X3TrainFold, Y3TrainFold, 0, 150)

    MSE_sum_3 += MSE(X3TestFold, Y3TestFold, Weight3)

MSETest3 = MSE_sum_3/10

Lambda3 = MSETest3.argmin()
MSE3 = MSETest3[Lambda3]

print("For dataset 1000-100, the lambda =", Lambda3, ", which gives the least test MSE of", round(MSE3, 3))

Fold4 = int(len(Ytrain4) / 10)

MSE_sum_4 = 0

for i in range(10):
    X4TestFold = Xtrain4[ i*Fold4 : (i+1)*Fold4 ]
    Y4TestFold = Ytrain4[ i*Fold4 : (i+1)*Fold4 ]

    X4TrainFold = np.concatenate((Xtrain4[ : i*Fold4], Xtrain4[ (i+1)*Fold4 : ]), axis=0)
    Y4TrainFold = np.concatenate((Ytrain4[ : i*Fold4], Ytrain4[ (i+1)*Fold4 : ]), axis=0)

    Weight4 = L2_Regularization(X4TrainFold, Y4TrainFold, 0, 150)

    MSE_sum_4 += MSE(X4TestFold, Y4TestFold, Weight4)

MSETest_4 = MSE_sum_4/10

Lambda4 = MSETest_4.argmin()
MSE4 = MSETest_4[Lambda4]

print("For dataset 50(1000)-100, the lambda =", Lambda4, ", which gives the least test MSE of", round(MSE4, 3))

Fold5 = int(len(Ytrain5) / 10)

MSE_sum_5 = 0

for i in range(10):
    X5TestFold = Xtrain5[ i*Fold5 : (i+1)*Fold5 ]
    Y5TestFold = Ytrain5[ i*Fold5 : (i+1)*Fold5 ]

    X5TrainFold = np.concatenate((Xtrain5[ : i*Fold5], Xtrain5[ (i+1)*Fold5 : ]), axis=0)
    Y5TrainFold = np.concatenate((Ytrain5[ : i*Fold5], Ytrain5[ (i+1)*Fold5 : ]), axis=0)

    Weight5 = L2_Regularization(X5TrainFold, Y5TrainFold, 0, 150)

    MSE_sum_5 += MSE(X5TestFold, Y5TestFold, Weight5)

MSETest_5 = MSE_sum_5/10

Lambda5 = MSETest_5.argmin()
MSE5 = MSETest_5[Lambda5]

print("For dataset 100(1000)-10, the lambda =", Lambda5, ", which gives the least test MSE of", round(MSE5, 3))

Fold6 = int(len(Ytrain6) / 10)

MSE_sum_6 = 0

for i in range(10):
    X6TestFold = Xtrain6[ i*Fold6 : (i+1)*Fold6 ]
    Y6TestFold = Ytrain6[ i*Fold6 : (i+1)*Fold6 ]

    X6TrainFold = np.concatenate((Xtrain6[ : i*Fold6], Xtrain6[ (i+1)*Fold6 : ]), axis=0)
    Y6TrainFold = np.concatenate((Ytrain6[ : i*Fold6], Ytrain6[ (i+1)*Fold6 : ]), axis=0)

    Weight6 = L2_Regularization(X6TrainFold, Y6TrainFold, 0, 150)

    MSE_sum_6 += MSE(X6TestFold, Y6TestFold, Weight6)

MSETest_6 = MSE_sum_6/10

Lambda6 = MSETest_6.argmin()
MSE6 = MSETest_6[Lambda6]

print("For dataset 150(1000)-10, the lambda =", Lambda6, ", which gives the least test MSE of", round(MSE6, 3))

# learning curve function that plots the learning curve of the lambda values of 1, 25, and 150
def learning_curve(XTrain, YTrain, x_test, y_test):
    lamda_vals = [1, 25, 150]
    
    for num in lamda_vals:
        list_size = range(10, 1000, 10)
        MSETestArray = np.zeros(len(list_size))
        MSETrainArray = np.zeros(len(list_size))
        
        for i, size in enumerate(list_size):
            RTest = []
            RTrain = []

            for _ in range(10):
                idx = np.random.choice(len(XTrain), size, replace=False)
                Weights = L2_Regularization(XTrain[idx], YTrain[idx], num, num)
                MSETest = MSE(x_test, y_test, Weights)
                MSETrain = MSE(XTrain[idx], YTrain[idx], Weights)
                RTest.append(MSETest)
                RTrain.append(MSETrain)

            MSETestArray[i] = np.average(RTest)
            MSETrainArray[i] = np.average(RTrain)

        # A new figure is created for each lamda value
        plt.figure(figsize=(10, 6))
        plt.plot(list_size, MSETestArray, label='MSE Test')
        plt.plot(list_size, MSETrainArray, label='MSE Train')
        plt.xlabel('Training Set Size')
        plt.ylabel(f'MSE with lambda = {num}')
        plt.title(f'MSE vs. Training Set Size (Lambda = {num})')
        plt.legend()
        filename = f'MSE_vs_Training_Set_Size_Lambda_{num}.png'
        plt.savefig(filename)
        plt.close()

# calling the learning curve function to plot the three plots
learning_curve(Xtrain3, Ytrain3, Xtest3, Ytest3)
