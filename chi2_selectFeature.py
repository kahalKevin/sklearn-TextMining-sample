import pandas as pd
import cPickle
import math 
import numpy
import sklearn.pipeline
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from itertools import izip

#pip install -U scikit-learn
#pip install pandas
#pip install scipy

def aboveExpected(nama_file):
    readtrain = pd.read_csv(nama_file+'.csv')
    cols = readtrain.columns.tolist()
    features = [c for c in cols if c not in ["label"]]
    labels = ['label']
    X = readtrain.drop(labels,axis=1)
    v = DictVectorizer(sparse=False)
    X=v.fit_transform(X.T.to_dict().values())
    y = readtrain.as_matrix(labels)
    y=y.ravel()
    X_new = SelectKBest(chi2, k='all')
    X_final=X_new.fit_transform(X, y)
    skor_Hi=0    
    temp=X_new.scores_
    for s in temp:
        if(s>10.83):            
            skor_Hi=skor_Hi+1
    return skor_Hi
    
def remakeHeaderList(nama_file, topfeature):
    headernew = []
    readtrain = pd.read_csv(nama_file+'.csv')
    cols = readtrain.columns.tolist()
    features = [c for c in cols if c not in ["label"]]
    labels = ['label']
    X = readtrain.drop(labels,axis=1)
    v = DictVectorizer(sparse=False)
    X=v.fit_transform(X.T.to_dict().values())
    y = readtrain.as_matrix(labels)
    y=y.ravel()
    X_new = SelectKBest(chi2, k=topfeature)
    X_final=X_new.fit_transform(X, y)
    top_ranked_features = sorted(enumerate(X_new.scores_),key=lambda x:x[1], reverse=True)[:topfeature]
    top_ranked_features_indices = map(list,zip(*top_ranked_features))[0]
    for feature_pvalue in zip(numpy.asarray(v.get_feature_names())[top_ranked_features_indices],X_new.pvalues_[top_ranked_features_indices]):
        headernew.append(feature_pvalue[0])
    return headernew

def recreateNgram(newHeader):
    counts = Counter()
    template = {}
    alltemplate = []
    header = newHeader #INSTEAD OF MAKING NEW HEADER, WE JUST COPY THE LIST OF PRE FILTERED HEADER

    csv_ready = ""

    with open('200sms.txt', 'r') as inFile:
            isi = inFile.readlines()
            hasil = []
            num = 1
            for line in isi:
                line = line.split()
                counts.update(hasil for hasil in line)
                words = [item for item in line]
                hasil += words
            for isi in hasil :
                template[isi] = 0
            counts.clear()


    with open('200sms.txt', 'r') as inFile:
            isi = inFile.readlines()
            hasil = []
            num = 1
            for line in isi:
                copyTemplate = template.copy()
                line = line.split()
                for _word in line:
                    copyTemplate[_word] = copyTemplate[_word] + 1
                alltemplate.append(copyTemplate)

    # INSERTION PROCESS

    i = 0
    for fitur in header:
        if(i==len(header)-1):
            csv_ready += fitur + '\n'
        else:
            csv_ready += fitur + ','
            i=i+1


    for filled in alltemplate:
        i = 0
        for fitur in header:
            if(i==len(header)-1):
                csv_ready += str(filled[fitur]) + '\n'
            else:
                csv_ready += str(filled[fitur]) + ','
                i=i+1

    # print csv_ready

    file = open("trainingready.csv","w") 
    file.write(csv_ready)
    file.close()

    labeled = ""

    label = []

    with open('label.txt', 'r') as inFile:
            isi = inFile.readlines()
            for line in isi:
                line = line.split('\n')
                label.append(line[0])

    with open('trainingready.csv', 'r') as inFile:
            isi = inFile.readlines()
            i=-1
            for line in isi:
                line = line.split('\n')
                if i==-1:
                    labeled += line[0] + ',label\n' 
                else:
                    labeled += line[0] + ',' + label[i] + '\n'
                i=i+1

    file = open("trainingready.csv","w") 
    file.write(labeled)
    file.close()


#################################RUN################################################

# nama_file = raw_input("Nama File :")
# train_x, test_x, train_y, test_y = split_data("labeledall")

#SEPARATE THE PROCESS TO COUNT NUMBER OF FEATURE ABOVE 10.83 AND DISPLAY THEM
#CAN NOT FIND ANY REFERENCE TO BE ABLE TO DO SO, QUITE SAD ACTUALLY
fiturpersist = aboveExpected("labeledall") #COUNT FEATURE ABOVE 10.83
selectedHeader = remakeHeaderList("labeledall",fiturpersist) #LIST ALL FEATURE ABOVE 10.83

recreateNgram(selectedHeader)