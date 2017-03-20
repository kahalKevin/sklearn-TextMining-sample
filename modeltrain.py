import pandas as pd
import cPickle
import math 
import numpy
import sklearn.pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from itertools import izip

# HERE YOU SEPARATE THE FEATURE AND LABEL
# THIS IS NEEDED FOR THE ALGORITHM DATA INPUT
def separateFeatureAndLabel(nama_file):
    print "\nReading data..."    
    readtrain = pd.read_csv(nama_file+'.csv')
    cols = readtrain.columns.tolist()
    features = [c for c in cols if c not in ["label"]]
    labels = ['label']
    X = readtrain.as_matrix(features)
    y = readtrain.as_matrix(labels)
    y=y.ravel()    
    print "\nSplitting feature and label..."
    # GENERALLY, IN SCIKITLEARN X IS FEATURE SET AND Y IS LABEL SET
    return (X, y)

# THIS PART DO SAME THING AS ABOVE
# EXCEPT IT ALSO SEPARATE THE DATA TO TRAIN DATA AND TEST DATA
# TEST DATA IS CREATED ACCORDING TO NUMBER OF 'percentage'
def separateFeatureAndLabel_Train_and_Test(nama_file,percentage):
    print "\nReading data..."    
    readtrain = pd.read_csv(nama_file+'.csv')
    cols = readtrain.columns.tolist()
    features = [c for c in cols if c not in ["label"]]
    labels = ['label']
    X = readtrain.as_matrix(features)
    y = readtrain.as_matrix(labels)
    y=y.ravel()    
    print "\nSplitting feature and label, some to train and some to test data..."
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=percentage)    
    return (X_train, X_test, y_train, y_test)

# TRAIN THE MODEL! AND SAVE IT AS FILE TO MAKE IT REUSEABLE
# AND TO AVOID TO RETRAIN MODEL EVERYTIME YOU WANT TO PREDICT
# BECAUSE IT IS VERY EXHAUTIVE, EVEN FOR SMALL DATA
def trainingModel(ml_algorithm, x, y):
    import os    
    print "\nTraining Model..."    
    ml_algorithm.fit(x, y.ravel())
    with open('TRAINEDMODEL.pkl','wb') as f:
        cPickle.dump(MACHINE_LEARNING, f)
    # FINALLY THE MODEL IS CREATED AND CAN BE SAVED AS A FILE(ACTUALLY YOU SERIALIZED IT TO PKL FILE)
    # SO THEN YOU CAN REUSE IT LATER
    if os.path.exists('TRAINEDMODEL.pkl'):
        return 'Model has been saved'


# PREDIT THE LABEL OF A TEST SET, HERE WE TEST THE INTELLIGENCE OF OUR MACHINE
def detect_testset(ml_algorithm, x):
    print "\nTesting data..."    
    with open('TRAINEDMODEL.pkl', 'rb') as f:
        ml_algorithm = cPickle.load(f)
    y_prediction = ml_algorithm.predict(x)
    return y_prediction

# HERE YOU PROCESS THE INPUT DATA WITH NO LABEL
# JUST THE RESULT OF NGRAM WITH CHOSEN FEATURE AFTER CHI2
def prepareDataToBePredicted(nama_file):
    print "\nReading data..."
    readtrain = pd.read_csv(nama_file+'.csv')
    cols = readtrain.columns.tolist()
    X = readtrain.as_matrix(cols)
    print "\nCreating feature for prediction input..."
    # GENERALLY, IN SCIKITLEARN X IS FEATURE SET AND Y IS LABEL SET
    return X


# WHERE YOU SAVE THE CSV FILE WHICH IS READY TO USE AS TRAIN DATA    
nama_file = "trainingready"

# THE RESULT OF, THIS NUMBER TIMES NUMBER OF DATA MUST AT LEAST EQUAL TO NUMBER OF UNIQUE LABEL
# IF N IS THE NUMBER, MUST BE 0 < N < 1  | IN OTHER WORDS, PERCENTAGE
percentageOfTestDataCreated = 0.03 

train_x, test_x, train_y, test_y = separateFeatureAndLabel_Train_and_Test(nama_file,percentageOfTestDataCreated)

# DO THIS INSTEAD IF YOU DONT WANT TO SPLIT TO TRAIN AND TEST, AND JUST USE ALL DATA TO CREATE MODEL
# IN SCIKIT LEARN X IS GENERALLY THE FEATURE SET AND Y IS THE LABEL SET
x, y = separateFeatureAndLabel(nama_file)




########-----------------------------------------------------------------------########

# HERE IS THE PART TO CHOOSE BETWEEN RANDOM FOREST, NAIVE BAYES, SVM

# THERE ARE SO MANY AVAILABLE PARAMETER ACTUALLY
# LEARN TO ADJUST IT, EXAMPLE....
# RandomForestClassifier(n_jobs=1, max_features= "auto" ,n_estimators=100,  min_samples_leaf = 55, oob_score = True)


# MACHINE_LEARNING = RandomForestClassifier(n_estimators=100) 

# MACHINE_LEARNING = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)

MACHINE_LEARNING = MultinomialNB()

########-----------------------------------------------------------------------########




# IF YOU WANT TO TRAIN THE MODEL WITH SEPARATED TRAIN SET FROM ALL DATA
trainingModel(MACHINE_LEARNING, train_x, train_y)

# IF YOU WANT TO TRAIN THE MODEL WITH ALL DATA
# trainingModel(MACHINE_LEARNING, x, y)

# PREDICT THE UNLABELED DATA USING THE MACHINE WHICH HAVE CREATED
predictions = detect_testset(MACHINE_LEARNING, test_x)

# COMPARE THE ANSWER!!
print list(test_y.ravel()) # ACTUAL TRUE ANSWER
print list(predictions) # MACHINE'S ANSWER, TRAIN THEM BETTER IF THEY ARE STILL DUMB




#TEST FILE FROM OUTSIDE!!!

# 'testInput.csv' IS EXAMPLE ACCEPTABLE FORMAT STRUCTURE THAT CAN BE ACCEPTED BY ALGORITHM TO PREDICT
# DO THIS KIND OF PREPROCESS FOR DATA INPUT FOR THE MODEL TO PREDICT
fileFromOutside = "testInput"
dataToPredictFromOutside = prepareDataToBePredicted(fileFromOutside)

predictions = detect_testset(MACHINE_LEARNING, dataToPredictFromOutside)
print "JAWABAN YG BENAR : banking, ham, operator, penipuan, promosi, unknown"
print list(predictions) # MACHINE'S ANSWER, TRAIN THEM BETTER IF THEY ARE STILL DUMB
