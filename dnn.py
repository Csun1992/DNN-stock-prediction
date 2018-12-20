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

def forwardPropagation(inputTensor, weights1, bias1, weights2, bias2, avgClass):
    if avgClass == None:
        layer1 = tf.relu(tf.matmul(inputTensor, weights1)+bias1)
        return tf.matmul(layer1, weights2)+bias2
    else:
        layer1 = tf.nn.relu(tf.matmul(inputTensor, avgClass.average(weights1)) +
                avgClass.average(bias1))
        return tf.matmul(layer1, avgClass.average(weights2) + avgClass.average(bias2))

def train(stockData):
    x = tf.placeholder(tf.float32, [None, inputNode], name='xInput')
    y = tf.placeholder(tf.float32, [None, outputNode], name='yInput')
    weights1 = tf.Variable(tf.truncated_normal([inputNode, layer1Node], stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape=[layer1Node]))
    weight2 = tf.Variable(tf.truncated_normal([layer1Node, outputNode], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[outputNode]))
    yHat = forwardPropagation(x, weights1, bias1, weights2, bias2, None)
    globalStep = tf.Variable(0, trainable=False)
