import tensorflow as tf
import sklearn as sk
from sklearn import model_selection
import numpy as np

def trainTestSplit(fileName, trainSize=0.8, randomState=41):
    clusterData = np.loadtxt('./data/clusterData.txt')
    trainFileName = './data/' + fileName + 'TrainData.txt'
    trainData = np.loadtxt(trainFileName)
    label = trainData[:, -1]
    data = np.concatenate((trainData[:,:-1], clusterData), axis=1)
    XTrain, XTest, yTrain, yTest = model_selection.train_test_split(data, label,
            test_size=1-trainSize, random_state=randomState)
    return (XTrain, XTest, yTrain, yTest) 


     
    
totalDataSize = 345
trainingPercent = 0.8
trainingDataSize = int(totalDataSize * trainingPercent)
testDataSize = totalDataSize - trainingDataSize
batchSize = 10
inputNode = 8 # data dimension is 9
layer1Node, layer2Node = 2, 2 # Two hidden layers, each has two nodes in it
outputNode = 1 # binary classification problem 
learningRateBase = 0.8
learningRateDecay = 0.99
regularizationRate = 0.0001
trainingSteps = tf.constant(3000)
movingAverageDecay = 0.99

def forwardPropagation(inputTensor, weights1, bias1, weights2, bias2, avgClass):
    if avgClass == None:
        layer1 = tf.nn.relu(tf.matmul(inputTensor, weights1)+bias1)
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
    weights2 = tf.Variable(tf.truncated_normal([layer1Node, outputNode], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[outputNode]))
    yHat = forwardPropagation(x, weights1, bias1, weights2, bias2, None)
    globalStep = tf.Variable(0, trainable=False)
    variableAverages = tf.train.ExponentialMovingAverage(movingAverageDecay, globalStep)
    variablesAveragesOp = variableAverages.apply(tf.trainable_variables()) # run moving average on
                                                                          # on all the trainables
    yAverage = forwardPropagation(x, weights1, bias1, weights2, bias2, variableAverages)

    crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(yHat, 1))
    crossEntropyMean = tf.reduce_mean(crossEntropy)

    regularizer = tf.contrib.layers.l2_regularizer(regularizationRate)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = crossEntropyMean + regularization 

    learningRate = tf.train.exponential_decay(learningRateBase, globalStep, trainingDataSize /
            batchSize, learningRateDecay)

    trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(loss, global_step=globalStep)
    
    trainOp = tf.group(trainingSteps,variablesAveragesOp)
    
    correctPrediction = tf.equal(tf.argmax(yAverage, 1), tf.argmax(yHat, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    
    trainFeature, testFeature, trainLabel, testLabel = trainTestSplit(stockData,
            trainSize = trainingPercent, randomState = 41)
    
    trainSize = np.size(trainFeature, 0)  
    maxBatchNO = trainSize - batchSize
    
    init = tf.global_variables_initializer()
        
    with tf.Session() as sess:
        init.run()
        for i in range(3000):
            randomBatch = np.random.randint(0, maxBatchNO+1)
            testFeed = {x:testFeature[randomBatch:(randomBatch+batchSize), :],
                y:testLabel[randomBatch:(randomBatch+batchSize)].reshape(-1,1)}
            sess.run(trainOp, feed_dict=testFeed)
    
        testAccuracy  = sess.run(accuracy, feed_dict=testFeed)
        print testAccuracy


def main(argv=None):
    stock = 'apple'
    train(stock)


if __name__ == '__main__':
    tf.app.run()
