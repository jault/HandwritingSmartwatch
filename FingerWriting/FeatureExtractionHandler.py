import csv
import itertools
import math

import pandas as pd
import numpy as np
import time

from DataCleaner import DataCleaner
from ExtractFeatures import ExtractFeatures

import os,sys
import glob

#Finger Writing



def start_func(filename):
    extract_save_features(filename)

############# Extract Features for Classification#  #################

def extract_save_features(inputFileNameSensor):
    #blockPrint()
    print('Filename: ',inputFileNameSensor)

    #
    dataCleaner=DataCleaner()

    #
    extFeatures=ExtractFeatures()

    # Query data
    sensorStream = open_csv(inputFileNameSensor)


    #singleCharListLA = dataCleaner.get_single_char_list(sensorStream, 'LA')
    singleCharListLA = dataCleaner. \
         get_single_char_list_rm_rj(sensorStream, 'LA', None)

    singleCharListGY = dataCleaner. \
        get_single_char_list_rm_rj(sensorStream, 'GY', None)


    singleCharListLA[:] = [item for item in singleCharListLA
                           if not (item['Char'].iloc[0]=='Do some PenDowns & PenUps'
                                   or item['Char'].iloc[0] == 'Press Accept!'
                                   or item['Char'].iloc[0] == 'Sync Validation. Press Accept.'
                                   or (item['Char'].iloc[0])!=(item['Char'].iloc[0]))]

    singleCharListGY[:] = [item for item in singleCharListGY
                           if not (item['Char'].iloc[0]=='Do some PenDowns & PenUps'
                                   or item['Char'].iloc[0] == 'Press Accept!'
                                   or item['Char'].iloc[0] == 'Sync Validation. Press Accept.'
                                   or (item['Char'].iloc[0])!=(item['Char'].iloc[0]))]

    totalFeatureSet=[]
    #singleCharListLA,singleCharListGY=dataCleaner.sync_LA_GY(singleCharListLA, singleCharListGY)
    #print("No of CharsLA", len(singleCharListLA))
    #print("No of CharsGY", len(singleCharListGY))

    totalFeatureSet.extend(get_features(extFeatures, singleCharListLA,singleCharListGY))
    #totalFeatureSet.extend(get_features(extFeatures, singleCharListGY, 'GY'))

    #suffix = inputFileNameSensor[-10:]
    #print("Suffix: ", suffix)

    #filename = os.path.join('Data/FW/', inputFileNameSensor[-27:-4])
    #filename = inputFileNameSensor[:-4]

    #filename = os.path.join('Data/TSI-Char-FeatureSets/', inputFileNameSensor[-10:-4])
    #save_features_to_csv(filename + 'CharFeatureSet' + '.csv', totalFeatureSet)

    filename = inputFileNameSensor[-17:-4]
    path = './Data/Results/Finger/FeatureSets/'
    filename = os.path.join(path, filename)
    save_features_to_csv(filename + 'CharFeatureSet' + '.csv', totalFeatureSet)

    #print('FeatureSets saved')

    enablePrint()
    #return gtMotionFrames,matchedFrameList

def get_features(extFeatures, singleCharListLA,singleCharListGY):
    TotFeatureSet=[]
    for i in range(len(singleCharListLA)):
        singleCharLA = singleCharListLA[i].loc[(singleCharListLA[i]['Label'] == 'LA')].copy()
        singleCharGY = singleCharListGY[i].loc[(singleCharListGY[i]['Label'] == 'GY')].copy()

        # Convert data type to numeric
        singleCharLA[['timestamp', 'xAxis', 'yAxis', 'zAxis']] = singleCharLA[
            ['timestamp', 'xAxis', 'yAxis', 'zAxis']].apply(pd.to_numeric)

        singleCharGY[['timestamp', 'xAxis', 'yAxis', 'zAxis']] = singleCharGY[
            ['timestamp', 'xAxis', 'yAxis', 'zAxis']].apply(pd.to_numeric)

        # Remove any Nan values
        # print(singleCharList[i].isnull().any())
        singleCharLA[['timestamp', 'xAxis', 'yAxis', 'zAxis']] = singleCharLA[
            ['timestamp', 'xAxis', 'yAxis', 'zAxis']].dropna(axis=0, how='any')

        singleCharGY[['timestamp', 'xAxis', 'yAxis', 'zAxis']] = singleCharGY[
            ['timestamp', 'xAxis', 'yAxis', 'zAxis']].dropna(axis=0, how='any')


        start = time.time()
        ######## Extracting Features #########
        #print('Feature Extraction Started')
        featureSetLA = list(extFeatures.get_feature_set_char(singleCharLA, 'LA'))
        featureSetGY = list(extFeatures.get_feature_set_char(singleCharGY, 'GY'))

        #print('featureSetLA',featureSetLA)
        #print('featureSetGY', featureSetGY[0][1:])

        fset=featureSetLA[0] + featureSetGY[0][1:]
        featureSet=[]
        featureSet.append(fset)
        #print('featureSet',fset)
        end = time.time()
        #print('Feature Extraction Finished')

        if (featureSet):
            TotFeatureSet.append(featureSet)

    return TotFeatureSet

def save_features_to_csv(filename,featureSetList):
    with open(filename, 'w', newline='') as out:
        rows = csv.writer(out)
        colNames = list(range(len(featureSetList[0][0])))
        rows.writerow(colNames)
        for f in featureSetList:
            if (len(f) > 0):
                for g in f:
                    rows.writerow(g)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def open_csv(filename):
    COLUMNS = ['Label', 'timestamp', 'xAxis', 'yAxis', 'zAxis','Extra', 'Char','magnitude']
    # dataf = pd.read_csv(filename, header=0, names=COLUMNS, low_memory=False, skiprows=20)
    dataf = pd.read_csv(filename, header=None, names=COLUMNS, low_memory=False)
    return dataf



import glob

if __name__ == '__main__':
    subfolders1 = [f.path for f in os.scandir(r'../Data/Finger/') if f.is_dir()]

    fileList1 = []
    for f in subfolders1:
        fileList = glob.glob(os.path.join(f, "*.csv"))
        for file in fileList:
            fileList1.append(file)
    for i in range(len(fileList1)):
        start_func(fileList1[i])