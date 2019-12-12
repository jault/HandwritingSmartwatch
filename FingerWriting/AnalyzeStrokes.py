import pandas as pd

from OOP3.DataCleaner import DataCleaner
from OOP2.Frame import Frame
import os

############# Extract the Segmented Ground Truth Characters-Tablet/Stylus #################
# Saves segmented characters of each file into relevant file name
def start_func(filePath):
    fileList = glob.glob(os.path.join(filePath, "*.csv"))
    #print('fileListMotion',fileListMotion)

    for file in fileList:
        print(file)
        #print(file.endswith('Chars_L',0,-4))
        if(file.endswith('Chars_L',0,-4)):
            suffixL = file[-17:-4]
            pre_process(file,suffixL)
        elif(file.endswith('Chars_U'),0,-4):
            suffixU = file[-17:-4]
            pre_process(file,suffixU)

# Tablet/Stylus
def pre_process(inputFileNameSensor,suffixU):

    #suffix = inputFileNameSensor[-9:]
    suffix=suffixU
    print("Suffix: ",suffix)


    #
    #thresholdHandler=ThresholdHandler()

    #
    dataCleaner=DataCleaner()

    # Query data
    sensorStream = open_csv(inputFileNameSensor)


    # singleCharListLA = dataCleaner.get_single_char_list(sensorStream[:30000], 'LA')
    # singleCharListGY = dataCleaner.get_single_char_list(sensorStream[:30000], 'GY')
    singleCharListLA = dataCleaner.get_single_char_list_rm_rj(sensorStream, 'LA',None)
    singleCharListGY = dataCleaner.get_single_char_list_rm_rj(sensorStream, 'GY',None)
    #singleCharListLA,singleCharListGY=dataCleaner.sync_LA_GY(singleCharListLA,singleCharListGY)

    singleCharListLA[:] = [item for item in singleCharListLA
                           if not (item['Char'].iloc[0]=='Do some PenDowns & PenUps'
                                   or item['Char'].iloc[0] == 'Press Accept!'
                                   or item['Char'].iloc[0] == 'Sync Validation. Press Accept.'
                                   or (item['Char'].iloc[0])!=(item['Char'].iloc[0]))]

    segmentedDf=pd.DataFrame()
    print("No of Chars", len(singleCharListLA))
    pre_segmentedDf=gt_segmentation(dataCleaner, singleCharListLA)

    segmentedDf=segmentedDf.append(pre_segmentedDf, ignore_index=True)

    segmentedDf.reset_index(drop=True)

    #if (len(inputFileNameSensor) == 33):
    savefile = os.path.join('../Data/Results/NoOfStrokes', suffix + '.csv')
    print('savefile-L1', savefile)
    segmentedDf.to_csv(savefile, index=False)
    # else:
    #     savefile = os.path.join('Data/TSI-GT-Segmented/', 'Segmd' + inputFileNameSensor[12:-4] + '.csv')
    #     print('savefile-L2', savefile)
    #     segmentedDf.to_csv(savefile, index=False)

def gt_segmentation(dataCleaner,singleCharListLA):
    segdDf = pd.DataFrame()
    strokedf = pd.DataFrame()
    for i in range(len(singleCharListLA)):
        # Get the timestamps of Ground Truth rows
        gtPDMotionTimestamps = dataCleaner.get_PD_PM_OR_PU(singleCharListLA[i], 'PD')
        gtPUMotionTimestamps = dataCleaner.get_PD_PM_OR_PU(singleCharListLA[i], 'PU')

        # Remove PD,PM,PU rows
        singleCharListLA[i] = singleCharListLA[i][singleCharListLA[i].Label != 'PM']
        singleCharListLA[i] = singleCharListLA[i][singleCharListLA[i].Label != 'PD']
        singleCharListLA[i] = singleCharListLA[i][singleCharListLA[i].Label != 'PU']

        # Convert data type to numeric
        singleCharListLA[i][['timestamp', 'xAxis', 'yAxis', 'zAxis']] = singleCharListLA[i][
            ['timestamp', 'xAxis', 'yAxis', 'zAxis']].apply(pd.to_numeric)

        # Remove any Nan values
        # print(singleCharList[i].isnull().any())
        singleCharListLA[i][['timestamp', 'xAxis', 'yAxis', 'zAxis']] = singleCharListLA[i][
            ['timestamp', 'xAxis', 'yAxis', 'zAxis']].dropna(axis=0, how='any')

        # Create a Frame object
        singleCharLA = Frame(singleCharListLA[i])
        if len((singleCharLA))>1:
            #singleCharLA.frame_filter()
            # singleCharLA.normalize()
            singleCharLA.magnitude()


            motionStrokes=extract_gt_char_strokes_motion2(gtPDMotionTimestamps,
                                                         gtPUMotionTimestamps,
                                                         singleCharListLA[i])
            numOfStrokes=len(motionStrokes)
            if(numOfStrokes==0):
                numOfStrokes=1

            GTChar=singleCharLA['Char'].iloc[0]

            # ['Label', 'timestamp', 'xAxis', 'yAxis', 'zAxis','Extra', 'Char' ]
            strokedf = strokedf.append({'Char': GTChar, 'NumOfStrokes':numOfStrokes, },
                                           ignore_index=True)

    return strokedf

def extract_gt_char_strokes_motion(gtPDMotionTimestamps, gtPUMotionTimestamps,motionStream):

    strokeTimestamps=[]
    if(len(gtPDMotionTimestamps)!=0 and len(gtPUMotionTimestamps)!=0):
            #and len(gtPDMotionTimestamps)==len(gtPUMotionTimestamps)):
        for i, j in zip(gtPDMotionTimestamps, gtPUMotionTimestamps):
            if (i < j):
                strokeTimestamps.append([i,j])

    motionCharStrokes = []
    # print('motion times',motionStream['timestamp'])
    # print('audio strokes',strokeTimestamps)
    if (len(strokeTimestamps) > 0):
        for stroke in strokeTimestamps:
            startTime = round(stroke[0], -1)
            endTime = round(stroke[1], -1)
            # print('start time: ',stroke[0], startTime)
            # print('end time: ', stroke[1], endTime)

            motionStroke = motionStream.loc[(motionStream['timestamp'] >= startTime)
                                            & (motionStream['timestamp'] <= endTime)]
            # print('motionstroke', motionStroke)
            motionCharStrokes.append(motionStroke)

        # If no strokes return the whole character as one stroke
        # Omit the first 5 rows and last  rows
    elif (len(strokeTimestamps) == 0):
        motionCharStrokes.append(motionStream[5:-5])

    return motionCharStrokes

#Only count PDs
def extract_gt_char_strokes_motion2(gtPDMotionTimestamps, gtPUMotionTimestamps,motionStream):

    strokeTimestamps=[]
    return gtPDMotionTimestamps
    #return motionCharStrokes

def open_csv(filename):
    COLUMNS = ['Label', 'timestamp', 'xAxis', 'yAxis', 'zAxis','Extra', 'Char' ]
    # dataf = pd.read_csv(filename, header=0, names=COLUMNS, low_memory=False, skiprows=20)
    dataf = pd.read_csv(filename, header=None, names=COLUMNS, low_memory=False)
    return dataf

#main()

import glob

subfolders = [f.path for f in os.scandir(r'../Data/FW/') if f.is_dir() ]


from multiprocessing import Pool

if __name__ == '__main__':
    #for i in range(len(subfolders)):
    # for i in range(1):
    #     start_func(subfolders[i])

    pool = Pool(3)
    pool.map(start_func,subfolders)
    pool.close()
    pool.join()


#https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments