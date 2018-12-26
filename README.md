# Project Purpose
We predict stock price moving direction with deep neural network implemented in Python

# Dependencies
TensorFlow

Numpy

sckit-learn

# Detailed Description of the Machine Learning Algorithm
We predict the direction of stock price movement in the future months with 3-layer deep neural network. So essentially, it is a binary classification problem.

The features we use include inflation rate, unemployment rate, rate of change of Dow Jones Industrial Average and rate of change of S & P 500, three-month moving average of stock price, two month moving average of stock price, stock price in current month and current fundamental value of the company.

The machine learning technique we chose here will take the macroeconomic environment into consideration when predicting the stock price movement. It is helpful in improving prediction precision since stock prices may behave differently during different periods of business cycles.

The data we used are all monthly data from Jan. 01 1990 to Sep. 01 2018. We avoided the noisy daily or weekly data.  

The example we offered here include Apple,ATT.

# Components of the Repository
1. The folder named *data* which contains the data for the deep neural network. The raw data are scraped from the web with tools such as Beautiful Soup etc.
2. The python file named *getData.py*. This file contains functions that transforms the raw data into format suitable for training. Since the raw data are taken from different sources, we wrote the ad hoc functions to transform the data. And then save the transformed data into *data* folder as well.
3. The python file *dnn.py*. This is the main file. It contains the deep neural network model. In this file, we created
- *forwardPropagation* function that perform the obvious forward propagation calculation for the neural network.
- *train* function that performs the backward-propagation training algorithm and set up TensorFlow training sessions to run the experiments and training.

# Future Work
1. Use dimension reduction techniques such as PCA to reduce features and thus the model complexity to prevent overfitting.
2. Use hypothesis testing techniques to identify best layer numbers, node numbers, regularization coefficients and trainning step sizes etc. 
3. Since we are dealing with time series data, it may be more appropriate to use reccurent neural network.

