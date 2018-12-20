import tensorflow as tf
import sklearn as sk

inputNode = 9 # data dimension is 9
layer1Node, layer2Node = 2 # Two hidden layers, each has two nodes in it
outputNode = 1 # binary classification problem 
learningRate = 0.8
learningRateDecay = 0.99
regularizerRate = 0.0001
trainingSteps = 3000
movingAverageDecay = 0.99
