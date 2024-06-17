import collections
from util import *

def extractWordFeatures(x):
    """
    Extract word features from a string x. Words should be delimited by whitespace characters only.
    @param string: x 
    @return dict: feature vector representation of x
    """
    
    wordDict=collections.defaultdict(float)
    for word in x.split():
        wordDict[word]+=1
    return wordDict

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string, x, and returns a sparse feature vector consisting of all n-grams of x without spaces.
    '''
    if n < 1:
        raise Exception("Numbers below 1 are not allowed")
    
    def extract(x):
    
        nDict=collections.defaultdict(float)
        x=x.replace(' ','')
        for i in range(0,len(x)-(n-1)):
            nDict[x[i:i+n]]+=1
        return nDict
    return extract

def learnPredictor(trainingData, testData, featureExtractor, epochs, step):
    '''
    Given trainingData and testData (as a list of (x,y) pairs), a featureExtractor to apply to x, and the epochs to train for, the step size, return the weight vector (sparse
    feature vector) learned.

    Implemented using stochastic gradient descent.
    '''
    weights = {}
    
    def predict(x):
        phi=featureExtractor(x)
        if dotProduct(weights,phi) < 0.0:
            return -1
        else:
            return 1
    for i in range(epochs):
        for item in trainingData:
            x,y = item
            phi = featureExtractor(x)
            temp = dotProduct(weights, phi) * y
            if temp < 1:
                increment(weights, -step * -y, phi)
        print("Epoch: %s, Training error: %s, Test error: %s"%(i, evaluatePredictor(trainingData,predict), evaluatePredictor(testData,predict)))

    return weights

learnPredictor(readExamples("polarity.train"), readExamples("polarity.dev"), extractWordFeatures, 500, 0.01)
learnPredictor(readExamples("polarity.train"), readExamples("polarity.dev"), extractCharacterFeatures(5), 500, 0.01)