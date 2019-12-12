import csv
import math
import os
import random
import re
import string
import time
from io import StringIO
import shutil, sys, os
import warnings
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import export_graphviz
from subprocess import call

from ExtractFeatures import ExtractFeatures

############# Finger Writing Personalized#################

def start_func(fileListMotion,scriptFlag):

    # fileListMotion=[]
    # for f in subfolders:
    #     fileListMotion.extend(glob.glob(os.path.join(f, "*.csv")))

    # print(fileListMotion[0])
    global suffixL, suffixU

    resultsListL=[]
    resultsListU = []
    for file in fileListMotion:
        #print(file)
        #print(file[-14:-4])
        #print(file.endswith('Chars_L', 0, -18))
        if (file.endswith('Chars_L', 0, -18)):
            # print(file)
            suffixL = '_L'
            lowercaseListDf = pd.read_csv(file,header=0)
            #classification_strat_k_fold(lowercaseListDf, suffixL, 'Lower')
            result=classification_train_split_cv(lowercaseListDf, suffixL, 'Lower')
            resultsListL.append(result)
        elif (file.endswith('Chars_U', 0, -18)):
            suffixU = '_U'
            uppercaseListDf = pd.read_csv(file,header=0)
            #classification_strat_k_fold(uppercaseListDf, suffixU, 'Upper')
            result=classification_train_split_cv(uppercaseListDf, suffixU, 'Upper')
            resultsListU.append(result)

    avgL,sdL=compute_average_results_from_list(resultsListL,'Lowercase',scriptFlag)
    avgU,sdU=compute_average_results_from_list(resultsListU,'Uppercase',scriptFlag)

    return avgL,sdL,avgU,sdU



# train_split_test_cv
def classification_train_split_cv(featureSet, suffix,case):
    #print('inputFileNameSensor',inputFileNameSensor)
    #suffix = inputFileNameSensor[-9:-4]
    #print("Suffix: ",suffix)

    alphabet = []
    if case is 'Lower':
        path = './Data/Results/Finger/Personalized/Lowercase/'
        for letter in range(97, 123):
            alphabet.append(chr(letter))

    if case is 'Upper':
        path = './Data/Results/Finger/Personalized/Uppercase/'
        for letter in range(65, 91):
            alphabet.append(chr(letter))

    #https://stackoverflow.com/questions/47390313/sklearn-loocv-split-returning-a-smaller-test-and-train-array-than-expected
    featureSet=featureSet[featureSet.iloc[:,0].isin(alphabet)]

    # X = charDf[:, 1:].reshape(-1,1)
    X = featureSet.iloc[:, 1:]
    y = featureSet.iloc[:, 0]

    resultsDF=call_classifier(X, y, 'RandomF')

    avgAccuracy = get_avg_from_report(resultsDF)
    rndmSt = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

    save_report(resultsDF, 'RandomF', rndmSt, path, suffix)

    return avgAccuracy

def get_avg_from_report(resultsListDf):
    report = classification_report(resultsListDf['GTChar'], resultsListDf['PredictedChar'], output_dict=True)
    report = pd.DataFrame(report).transpose()
    avgAccuracy=(report['recall'].iloc[-2])
    return  avgAccuracy

def save_report(resultsListDf, classfType,rndmSt,path,suffix):
    report = classification_report(resultsListDf['GTChar'], resultsListDf['PredictedChar'], output_dict=True)
    report = pd.DataFrame(report).transpose()
    if classfType is "DTree":
        savefileReport = os.path.join(path, rndmSt + 'Report' + suffix + "DTree" +'.csv')
    elif classfType is "LogisticR":
        savefileReport = os.path.join(path, rndmSt + 'Report' + suffix + "LogR" + '.csv')
    elif classfType is "RandomF":
        savefileReport = os.path.join(path, rndmSt + 'Report' + suffix + "RandF" + '.csv')
    else:
        savefileReport = os.path.join(path, rndmSt + 'Report' + suffix + "NB" + '.csv')

    report.to_csv(savefileReport)

def call_classifier(X, y, classfType):
    X=X.values
    y=y.values
    #print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, stratify=y)
    #print(y_test)
    resultsList = []
    if classfType is "DTree":
        model=build_DTree(X_train,y_train)
    elif classfType is "LogisticR":
        model = build_LogisticR(X_train,y_train)
    elif classfType is "RandomF":
        model = build_RandomF(X_train,y_train)
    else:
        model = build_NaiveBayes(X_train,y_train)

    resultsDF=model_test(X_test,y_test, model)

    return resultsDF
    #return resultsDF,report

def build_RandomF(X_train,y_train):
    warnings.filterwarnings("ignore")
    
    model = RandomForestClassifier(n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=40, bootstrap=False).fit(X_train, y_train)
    enablePrint()

    return model

def build_DTree(X_train,y_train):
    #blockPrint()
    warnings.filterwarnings("ignore")

    #print(X_train)
    model = DecisionTreeClassifier(max_depth=5, max_features=0.5,
                                   min_samples_leaf=4, min_samples_split=8, class_weight='balanced',random_state=2).fit(X_train,
                                                                                                         y_train)
                                                                                                         
    export_graphviz(model, out_file='./imgs/tree.dot',rounded = True, proportion = True, precision = 2, filled = True, leaves_parallel=False)
    enablePrint()
    #warnings.filterwarnings("default")

    return model

def build_LogisticR(X_train,y_train):
    #blockPrint()
    warnings.filterwarnings("ignore")

    #print(X_train)
    model = LogisticRegression(C=1000, multi_class='auto', solver='liblinear', n_jobs=2) \
        .fit(X_train, y_train)

    enablePrint()
    #warnings.filterwarnings("default")

    return model

def build_NaiveBayes(X_train,y_train):
    #blockPrint()
    warnings.filterwarnings("ignore")

    #print(X_train)
    model = GaussianNB().fit(X_train, y_train)

    enablePrint()
    #warnings.filterwarnings("default")

    return model

def model_test(X_test,y_test, model):
    blockPrint()

    # dset.replace(np.inf, 0, inplace=True)
    # dset.replace(np.nan, 0, inplace=True)
    #print(X_test)
    y_pred = model.predict(X_test)
    #print('ttttttttttttttt',y_test)
    resultsDf = compare_results_with_gt(y_test, y_pred)
    # print(resultsList)

    # report=report_to_df(classification_report(dset[:, 0], y_pred))
    # rndmSt=''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    # report.to_csv(rndmSt+'.csv')
    # Print the predicted output
    # print('\nOriginal: ', original_label)
    # print('Predicted:', predicted_label)

    enablePrint()
    return resultsDf


def open_csv(filename):
    #COLUMNS = ['Label', 'timestamp', 'xAxis', 'yAxis', 'zAxis','Extra', 'Char','magnitude']
    COLUMNS = ['Label', 'timestamp', 'xAxis', 'yAxis', 'zAxis', 'Extra', 'Char']
    dataf = pd.read_csv(filename, header=None, names=COLUMNS, low_memory=False)
    #dataf = D
    return dataf

def save_csv(filename):
    pass

####### Compare DTW results frames with ground truth ########
# Returns TPCount,FPCount,FNCount (a tuple)
def compare_results_with_gt(gtCharList, predictedCharList):
    #print("GT", gtMotionFrameList)
    #print('Predicted', DTWResultsFrameList)

    #print("GT",gtCharList)
    #print("Predicted",predictedCharList)

    TPCount=0
    FPCount=0
    FNCount=0

    cols = ['GTChar', 'PredictedChar']
    resultsList=[]
    #print(type(gtCharList),type(predictedCharList))
    for i in range(len(gtCharList)):
        #print(predictedCharList[i])
        resultsList.append([gtCharList[i],predictedCharList[i]])
        if(gtCharList[i]==predictedCharList[i]):
            TPCount=TPCount+1
        else:
            FPCount=FPCount+1

    resultsDf = pd.DataFrame(resultsList, columns=cols)
    #print('resultsDf', resultsDf)
    #report = classification_report(gtCharList, predictedCharList)

    # confmDf = pd.crosstab(gtCharList, predictedCharList, rownames=['True'],
    #                       colnames=['Predicted']).apply(lambda r: 100.0 * r / r.sum())

    #return resultsDf,report,confmDf

    return resultsDf


def save_results(inputFileName,resultsList,GT):
    ###########PD#################
    #print('results list',resultsList)
    npResultsList = np.asarray(resultsList)
    avg=npResultsList.mean(axis=0)
    if(math.isnan(avg[4])):
        AvgPrecision='null'
        AvgRecall='null'
    else:
        AvgPrecision=avg[4]
        AvgRecall = avg[5]
    #print(AvgPrecision)


    # print('audioFrameList',npResultsList)
    # # TPsum = npResultsList.sum(axis=1)
    # # FPsum = npResultsList.sum(axis=2)
    # # FNsum = npResultsList.sum(axis=3)
    # npResultsList.sum(axis=0)
    #
    #
    # precision=TPsum/(TPsum+FPsum)
    # recall=TPsum/(TPsum+FNsum)

    filename=os.path.basename(inputFileName)
    savefile=os.path.join('Data/TSI-DTW-Results/', filename[:-4] + 'Results' + GT + '.csv')
    #savefile = filename[:-4] + 'Results' + '.csv'
    with open(savefile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['GTCount','TPCount', 'FPCount', 'FNCount','Precision','Recall'])
        writer.writerows(resultsList)
        writer.writerow(['AvgPrecision','AvgRecall'])
        writer.writerow([AvgPrecision,AvgRecall])
        # writer.writerow(['TPCountTotal', 'FPCountTotal', 'FNCountTotal'])
        # writer.writerow(total)
        # resultsList.to_csv('Results.csv', index=False)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


from statistics import mean,stdev
def compute_average_results(case):
    if case is 'Lowercase':
        path = '../Data/Results/Finger/Personalized/Lowercase/'
    else:
        path = '../Data/Results/Finger/Personalized/Uppercase/'

    all_files = glob.glob(os.path.join(path, "*.csv"))

    df_from_each_file = (pd.read_csv(f,header=0) for f in all_files)

    #print(len(list(df_from_each_file)))

    avgList=[]
    for df in df_from_each_file:
        avgList.append(df['recall'].iloc[-2])

    avg=mean(avgList)
    standardDev=stdev(avgList)
    #print(case + ": ", "Average Accuracy: ", avg, "Standard Deviation: ",standardDev)
    print(case + ": ", "Average Accuracy: ", "{0:.0%}".format(avg), "Standard Deviation: ", "{0:.0%}".format(standardDev))

    #Remove Temp Files
    #Remove Temp Files
    for file in os.scandir(path):
        if file.name.endswith(".csv"):
            os.remove(file)

def compute_average_results_from_list(avgList,case,scriptFlag):
    avg=mean(avgList)
    standardDev=stdev(avgList)
    if scriptFlag is 'script':
        print(case + ": ", "Average Accuracy: ", "{0:.0%}".format(avg), "Standard Deviation: ", "{0:.0%}".format(standardDev))

    return avg,standardDev

from sklearn.metrics import confusion_matrix
def generate_confusion_matrix(case):
    if case is 'Lowercase':
        path = './Data/Results/Finger/Personalized/Lowercase/'
    else:
        path = './Data/Results/Finger/Personalized/Uppercase/'

    all_files = glob.glob(os.path.join(path, "*.csv"))


    df_from_each_file = (pd.read_csv(f, header=0) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(concatenated_df['GTChar'], concatenated_df['PredictedChar'])
    np.set_printoptions(precision=2)

    plot_confusion_matrix(cnf_matrix, classes=concatenated_df['GTChar'].unique(), normalize=True,
                          title=case+' Confusion Matrix')

    rndmSt = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    plt.savefig('Personalized '+'ConfMatrix' + case + '.png', dpi=330)
    plt.show()

    #Remove Temp Files
    # Gather directory contents
    contents = [os.path.join(path, i) for i in os.listdir(path)]

    # Iterate and remove each item in the appropriate manner
    [os.remove(i) if os.path.isfile(i) or os.path.islink(i) else shutil.rmtree(i) for i in contents]

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, format(cm[i, j], fmt),
        #          horizontalalignment="center",
        #          color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



import glob
import FeatureExtractionHandler

def main():
    subfolders = [f.path for f in os.scandir(r'./Data/Finger/') if f.is_dir()]
    fileList = []
    for f in subfolders:
        fL = glob.glob(os.path.join(f, "*.csv"))
        for file in fL:
            fileList.append(file)

    #print(fileList)
    for i in range(len(fileList)):
        FeatureExtractionHandler.start_func(fileList[i])

    pathFS = r'./Data/Results/Finger/FeatureSets'
    fileListFeatureSets = glob.glob(os.path.join(pathFS, "*.csv"))
    return(start_func(fileListFeatureSets,'notS'))


#https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments