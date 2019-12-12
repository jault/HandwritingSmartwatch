import math
import random

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

class DataCleaner:

    #############Motion Sensor Methods######################
    # sensor = 'LA' or 'GY'
    # returns a list
    def get_single_char_list(self,user_data, sensor):
        # Extract all LA,PD,PU rows

        # Forward fill char column to help grouping
        user_data.loc[:,'Char'] = user_data['Char'].fillna(method='ffill')
        #print('after',user_data['Char'])
        #user_data['Char'].fillna(method='ffill',inplace=True)
        #print('ttttttttttttttttt',user_data['Char'])

        outputLA = user_data.loc[(user_data['Label'] == sensor) |
                                 (user_data['Label'] == 'NS') |
                                 (user_data['Label'] == 'PD') |
                                 (user_data['Label'] == 'PU')]

        ########## Group by NS ##########

        mNS = outputLA.iloc[:, 0].eq('NS')

        # ~ for not (| for or, & for and )
        cumgrpNS = mNS.cumsum()[~mNS]

        grpsNS = outputLA[~mNS].groupby(cumgrpNS)

        list_of_groupsNS = [g for n, g in grpsNS]
        print('list_of_groupsNS', len(list_of_groupsNS))

        # Trim 2 rows from begining and end of a char
        final_list=[]
        for i in range(len(list_of_groupsNS)):
            #list_of_groupsNS[i] = list_of_groupsNS[i].iloc[2:-2]
            li=list_of_groupsNS[i].iloc[2:-2]
            if(len(li)>0):
                final_list.append(li)
            #list_of_groupsNS[i] = list_of_groupsNS[i].iloc[10:-10]

        return final_list

    def get_single_char_list_Large(self,user_data, sensor):
        # Extract all LA,PD,PU rows

        # Forward fill char column to help grouping
        user_data.loc[:,'Char'] = user_data['Char'].fillna(method='ffill')
        #print('after',user_data['Char'])
        #user_data['Char'].fillna(method='ffill',inplace=True)
        #print('ttttttttttttttttt',user_data['Char'])

        outputLA = user_data.loc[(user_data['Label'] == sensor) |
                                 (user_data['Label'] == 'NS') |
                                 (user_data['Label'] == 'PD') |
                                 (user_data['Label'] == 'PU')]

        ########## Group by NS ##########

        mNS = outputLA.iloc[:, 0].eq('NS')

        # ~ for not (| for or, & for and )
        cumgrpNS = mNS.cumsum()[~mNS]

        grpsNS = outputLA[~mNS].groupby(cumgrpNS)

        list_of_groupsNS = [g for n, g in grpsNS]
        print('list_of_groupsNS', len(list_of_groupsNS))

        # Trim 2 rows from begining and end of a char
        #print('before',list_of_groupsNS[0])
        final_list=[]
        for i in range(len(list_of_groupsNS)):
            c = list_of_groupsNS[i].iloc[25:-25]
            if(not c.empty):
                final_list.append(c)


        #print('after', list_of_groupsNS[0])
        return final_list
        #return list_of_groupsNS

    # With Skew Adjustment
    def get_single_char_list_rm_rj(self,user_data, sensor, skew):

        ######### sort by timestamp #########
        user_data = user_data.sort_values(by='timestamp')

        user_data.loc[:,'Char'] = user_data['Char'].fillna(method='ffill')

        # Extract all LA,PD,PU rows
        outputLA = user_data.loc[(user_data['Label'] == 'WS') |
                                 (user_data['Label'] == 'PD') |
                                 (user_data['Label'] == 'PU') |
                                 (user_data['Label'] == 'PM') |
                                 (user_data['Label'] == sensor) |
                                 (user_data['Label'] == 'AC') |
                                 (user_data['Label'] == 'RJ') |
                                 (user_data['Label'] == 'NS')]

        # user_data['Char'] = user_data['Char'].fillna(method='ffill')
        # skew = list_of_groups[group].iloc[0, 2]
        # list_of_groups[group].loc[list_of_groups[group].Label != sensor, 'timestamp'] -= int(skew)

        #Skew Adjustment
        # outputLA2 = outputLA.copy()
        # outputLA2.loc[outputLA2['Label'] != sensor, 'timestamp'] -= skew


        ########## Group by NS ##########

        mNS = outputLA.iloc[:, 0].eq('NS')

        # ~ for not (| for or, & for and )
        cumgrpNS = mNS.cumsum()[~mNS]

        grpsNS = outputLA[~mNS].groupby(cumgrpNS)

        list_of_groupsNS = [g for n, g in grpsNS]

        ########## Remove rejected rows from each group ##########

        for group in range(len(list_of_groupsNS)):
            list_of_groupsNS[group] = list_of_groupsNS[group].reset_index(drop=True)
            if ('RJ' in list_of_groupsNS[group].Label.values):
                # remove all rows above a 'RJ'
                idx = np.r_[:list_of_groupsNS[group].index[list_of_groupsNS[group]['Label'] == 'RJ'][-1] + 1]
                # print(idx)
                # group = group.drop(idx)
                list_of_groupsNS[group].drop(list_of_groupsNS[group].index[idx], inplace=True)

            # Drop the row with 'AC','WS'
            list_of_groupsNS[group] = list_of_groupsNS[group][list_of_groupsNS[group].Label != 'AC']
            list_of_groupsNS[group] = list_of_groupsNS[group][list_of_groupsNS[group].Label != 'WS']

            # Remove PD,PM,PU rows
            # list_of_groupsNS[group] = list_of_groupsNS[group][list_of_groupsNS[group].Label != 'PD']
            # list_of_groupsNS[group] = list_of_groupsNS[group][list_of_groupsNS[group].Label != 'PU']
            # list_of_groupsNS[group] = list_of_groupsNS[group][list_of_groupsNS[group].Label != 'PM']

        list_of_groupsNS_Final = []
        for group in range(len(list_of_groupsNS)):
            if (len(list_of_groupsNS[group]) > 60):
                #print('tttttttttt', len(list_of_groupsNS[group]))
                list_of_groupsNS_Final.append(list_of_groupsNS[group])

        # Trim 5 rows from begining and end of a char
        #print('before',list_of_groupsNS[0])

        # final_list=[]
        # for i in range(len(list_of_groupsNS)):
        #     c = list_of_groupsNS[i].iloc[5:-5]
        #     if(not c.empty):
        #         final_list.append(c)
        #

        #print('len', len(list_of_groupsNS))
        #print('len2', len(list_of_groupsNS_Final))

        # for c in list_of_groupsNS_Final:
        #     print(c['Char'].iloc[0])

        # return list_of_groupsNS
        return list_of_groupsNS_Final

        #return final_list

    # With Skew Adjustment
    def get_single_char_list_rm_rj2(self,user_data, sensor, skew,audioSkew):

        ######### sort by timestamp #########
        #user_data = user_data.sort_values(by='timestamp')

        #user_data.loc[:,'Char'] = user_data['Char'].fillna(method='ffill')

        # Extract all LA,PD,PU rows
        outputLA = user_data.loc[(user_data['Label'] == 'WS') |
                                 (user_data['Label'] == 'PD') |
                                 (user_data['Label'] == 'PU') |
                                 (user_data['Label'] == 'PM') |
                                 (user_data['Label'] == sensor) |
                                 (user_data['Label'] == 'AC') |
                                 (user_data['Label'] == 'RJ') |
                                 (user_data['Label'] == 'NS')]

        # user_data['Char'] = user_data['Char'].fillna(method='ffill')
        # skew = list_of_groups[group].iloc[0, 2]
        # list_of_groups[group].loc[list_of_groups[group].Label != sensor, 'timestamp'] -= int(skew)

        #Skew Adjustment
        outputLA2 = outputLA.copy()
        outputLA2.loc[outputLA2['Label'] != sensor, 'timestamp'] -= (skew)
        outputLA2['timestamp'] += audioSkew

        ######### sort by timestamp #########
        outputLA2 = outputLA2.sort_values(by='timestamp')
        outputLA2.loc[:, 'Char'] = outputLA2['Char'].fillna(method='ffill')

        ########## Group by NS ##########

        mNS = outputLA2.iloc[:, 0].eq('NS')

        # ~ for not (| for or, & for and )
        cumgrpNS = mNS.cumsum()[~mNS]

        grpsNS = outputLA2[~mNS].groupby(cumgrpNS)

        list_of_groupsNS = [g for n, g in grpsNS]

        ########## Remove rejected rows from each group ##########

        for group in range(len(list_of_groupsNS)):
            list_of_groupsNS[group] = list_of_groupsNS[group].reset_index(drop=True)
            if ('RJ' in list_of_groupsNS[group].Label.values):
                # remove all rows above a 'RJ'
                idx = np.r_[:list_of_groupsNS[group].index[list_of_groupsNS[group]['Label'] == 'RJ'][-1] + 1]
                # print(idx)
                # group = group.drop(idx)
                list_of_groupsNS[group].drop(list_of_groupsNS[group].index[idx], inplace=True)

            # Drop the row with 'AC','WS'
            list_of_groupsNS[group] = list_of_groupsNS[group][list_of_groupsNS[group].Label != 'AC']
            list_of_groupsNS[group] = list_of_groupsNS[group][list_of_groupsNS[group].Label != 'WS']

            # Remove PD,PM,PU rows
            # list_of_groupsNS[group] = list_of_groupsNS[group][list_of_groupsNS[group].Label != 'PD']
            # list_of_groupsNS[group] = list_of_groupsNS[group][list_of_groupsNS[group].Label != 'PU']
            # list_of_groupsNS[group] = list_of_groupsNS[group][list_of_groupsNS[group].Label != 'PM']

        list_of_groupsNS_Final = []
        for group in range(len(list_of_groupsNS)):
            if (len(list_of_groupsNS[group]) > 60):
                #print('tttttttttt', len(list_of_groupsNS[group]))
                list_of_groupsNS_Final.append(list_of_groupsNS[group])

        # Trim 5 rows from begining and end of a char
        #print('before',list_of_groupsNS[0])

        # final_list=[]
        # for i in range(len(list_of_groupsNS)):
        #     c = list_of_groupsNS[i].iloc[5:-5]
        #     if(not c.empty):
        #         final_list.append(c)
        #
        print('len', len(list_of_groupsNS))
        print('len2', len(list_of_groupsNS_Final))

        # for c in list_of_groupsNS_Final:
        #     print(c['Char'].iloc[0])

        # return list_of_groupsNS
        return list_of_groupsNS_Final

        #return final_list

    def magnitude(self,signal):
        x2 = signal['xAxis'].astype('float') * signal['xAxis'].astype('float')
        y2 = signal['yAxis'].astype('float') * signal['yAxis'].astype('float')
        z2 = signal['zAxis'].astype('float') * signal['zAxis'].astype('float')
        m2 = x2 + y2 + z2
        m = m2.apply(lambda x: math.sqrt(x))
        signal['magnitude'] = m
        return signal

    def get_single_char_list_rm_rj_word_segment(self,singleWordListLA):

        wordLabelList = []
        outputLA=pd.DataFrame()
        for word in singleWordListLA:
            wordLabelList.append(word['Char'].iloc[0])
            outputLA=outputLA.append(word,ignore_index=True)

        print('word list length: ',len(wordLabelList))
        outputLA = self.magnitude(outputLA)

        ######### sort by timestamp #########
        outputLA = outputLA.sort_values(by='timestamp')
        #outputLA2.loc[:, 'Char'] = outputLA2['Char'].fillna(method='ffill')
        self.plot_la_gy_magnitude(outputLA)
        # plt.plot(outputLA['magnitude'])
        # plt.show()
        breakpoint()

        minAvg=0
        maxAvg = 0
        minAvg = outputLA['zAxis'].min()
        maxAvg = outputLA['zAxis'].max()
        mean = outputLA['zAxis'].mean()

        PUFoIs = []
        PDFoIs = []
        for i in range(len(outputLA)):
            boolPU = outputLA[i]['magnitude'] > thresholdPU
            # print(type(bool))
            if (True in boolPU.unique()):
                PUFoIs.append(i)

            boolPD = sensorFrameList[i]['zAxis'] > thresholdPD
            # print(type(bool))
            if (True in boolPD.unique()):
                PDFoIs.append(i)

            # if(sensorFrameList[i]['zAxis'].mean() < thresholdPU):
            #     PUFoIs.append(i)
            # elif(sensorFrameList[i]['zAxis'].mean() > thresholdPD):
            #     PDFoIs.append(i)


        #PDList, PUList=self.audio_filteration(audioFrameList,foIs)
        return PDFoIs, PUFoIs


        return list_of_groupsNS_Final

        #return final_list

    def moving_avg_filter(self,sensorStream,windowSize):
        sensorStream['magnitude'] = sensorStream['magnitude'].rolling(window=windowSize,win_type='triang',min_periods=1).mean()
        # self['yAxis'] = self['yAxis'].rolling(window=windowSize,win_type='triang',min_periods=1).mean()
        # self['zAxis'] = self['zAxis'].rolling(window=windowSize,win_type='triang',min_periods=1).mean()
        # self['magnitude'] = self['magnitude'].rolling(window=windowSize, win_type='triang', min_periods=1).mean()
        return sensorStream

    def plot_la_gy_magnitude(self,sensorStream):
        sensorStream=self.moving_avg_filter(sensorStream,25)
        axarr=plt.plot(sensorStream['timestamp'], sensorStream['magnitude'], label='magnitude', color='green')

        plt.xlabel('time(ms)')
        plt.ylabel('amplitude')

        plt.legend(loc='upper right')

        #Find local peaks
        nV = 250 # number of points to be checked before and after
        nP = 250
        # Find local peaks
        valleysX = sensorStream['magnitude'].iloc[argrelextrema(sensorStream['magnitude'].values, np.less, order=nV)[0]].index
        peaksX = sensorStream['magnitude'].iloc[argrelextrema(sensorStream['magnitude'].values, np.greater, order=nP)[0]].index

        threshold2=(sensorStream['magnitude'][peaksX].mean() + sensorStream['magnitude'][valleysX].mean() )/2

        print('th2',threshold2)

        #print('peaksX',peaksX)

        plt.scatter(sensorStream['timestamp'][peaksX], sensorStream['magnitude'][peaksX],linewidth=0.3, s=50, c='r')
        plt.scatter(sensorStream['timestamp'][valleysX], sensorStream['magnitude'][valleysX], linewidth=0.3, s=50, c='b')

        plt.axhline(y=threshold2, linewidth=1, color='k')


        plt.show()

    # Get samples bet. left edge and right edge for GY
    def trim_LA_GY_Pencil(self,list_of_groupsNSLA, list_of_groupsNSGY):

        final_listLA = []
        final_listGY = []
        for dataLA, dataGY in zip(list_of_groupsNSLA, list_of_groupsNSGY):
            #print('ttttttttttt',len(dataLA))
            dataLA = dataLA.reset_index()
            dataGY = dataGY.reset_index()
            #dataXLA = dataLA['xAxis']
            dataXLA = dataGY['xAxis']

            n = 75  # number of points to be checked before and after
            valleysX = dataXLA.iloc[argrelextrema(dataXLA.values, np.less, order=n)[0]].index
            peaksX = dataXLA.iloc[argrelextrema(dataXLA.values, np.greater, order=n)[0]].index

            if(len(peaksX)>0):
                # startPoint = peaksX.index[0]
                # endPoint = valleysX.index[-1]
                startPoint = peaksX[0]
            else:
                startPoint = 5
            if(len(valleysX)>0):
                endPoint = valleysX[-1]
            else:
                endPoint = -5

            #print('LA', startPoint, endPoint)
            cLA = dataLA[startPoint + 5:endPoint - 5]
            cGY = dataGY[startPoint + 5:endPoint - 5]
            #print('LALen', len(cLA))
            if (not cLA.empty):
                if(len(cLA)>100):
                    final_listLA.append(cLA)
                else:
                    final_listLA.append(dataLA[5:-5])
            if (not cGY.empty):
                if(len(cGY)>100):
                    final_listGY.append(cGY)
                else:
                    final_listGY.append(dataGY[5:-5])


        return final_listLA, final_listGY
        # print('len',len(list_of_groupsNS))
        # return list_of_groupsNS

    # Sync LA and GY
    def sync_LA_GY(self,charListLA, charListGY):
        # print('LA , GY lengths',len(charListLA),len(charListGY))
        for i in range(len(charListLA)):
            # print('eeeeeeeeeee', len(charListLA[i+10]), len(charListGY[i+10]))
            if (len(charListLA[i]) != 0 and len(charListGY[i]) != 0):
                # Remove from head and tail
                if charListLA[i].iloc[0]['timestamp'] < charListGY[i].iloc[0]['timestamp']:
                    while charListLA[i].iloc[0]['timestamp'] < charListGY[i].iloc[0]['timestamp']:
                        h = charListLA[i].head(1)
                        charListLA[i] = charListLA[i].drop(h.index)
                        # dfLA = pd.concat([h, dfLA], ignore_index=True)

                elif charListGY[i].iloc[0]['timestamp'] < charListLA[i].iloc[0]['timestamp']:
                    while charListGY[i].iloc[0]['timestamp'] < charListLA[i].iloc[0]['timestamp']:
                        h = charListGY[i].head(1)
                        charListGY[i] = charListGY[i].drop(h.index)
                        # dfGY = pd.concat([h, dfGY], ignore_index=True)

                if charListLA[i].iloc[-1]['timestamp'] > charListGY[i].iloc[-1]['timestamp']:
                    while charListLA[i].iloc[-1]['timestamp'] > charListGY[i].iloc[-1]['timestamp']:
                        h = charListLA[i].tail(1)
                        charListLA[i] = charListLA[i].drop(h.index)

                elif charListGY[i].iloc[-1]['timestamp'] > charListLA[i].iloc[-1]['timestamp']:
                    while charListGY[i].iloc[-1]['timestamp'] > charListLA[i].iloc[-1]['timestamp']:
                        h = charListGY[i].tail(1)
                        charListGY[i] = charListGY[i].drop(h.index)

                # Drop duplicate stamps
                charListLA[i].drop_duplicates(subset='timestamp', inplace=True)
                charListGY[i].drop_duplicates(subset='timestamp', inplace=True)

                # Convert timestamp to datetimeIndex
                charListLA[i] = charListLA[i].set_index(['timestamp'])
                charListLA[i].index = pd.to_datetime(charListLA[i].index, unit='ms')
                charListLA[i].index.rename('timestamp', inplace=True)

                charListGY[i] = charListGY[i].set_index(['timestamp'])
                charListGY[i].index = pd.to_datetime(charListGY[i].index, unit='ms')
                charListGY[i].index.rename('timestamp', inplace=True)

                # Resample
                charListLA[i] = charListLA[i].resample('10L').bfill()
                charListGY[i] = charListGY[i].resample('10L').bfill()

                # Revert the datetimeIndex to timestamp column
                charListLA[i].reset_index(level=0, inplace=True)
                charListGY[i].reset_index(level=0, inplace=True)
                charListLA[i]['timestamp'] = (charListLA[i]['timestamp'].astype(np.int64) // 10 ** 6)
                charListGY[i]['timestamp'] = (charListGY[i]['timestamp'].astype(np.int64) // 10 ** 6)

                # print('ttttttttt',len(charListLA[i]),len(charListGY[i]))
        return charListLA, charListGY



    ########Char templates and clusters#############
    def get_random_from_clusters(self, cluster ,nOfSamples):
        # lst=[]
        num_to_select = nOfSamples  # set the number to select here.
        list_of_random_items = random.sample(cluster, num_to_select)

        return  list_of_random_items

    def get_PD_PM_OR_PU(self,single_char, gt_motion):
        # Check whether PDs exists in Label column
        gtMotionTimestampList = []
        # print(single_char)
        if gt_motion in single_char.Label.values:
            # if single_char.Label.str.contains(gt_motion).any():
            # print("Testttttttt")
            # print(single_char.loc[single_char['Label'] == gt_motion]['timestamp'])
            gtMotionTimestampList = list(single_char.loc[single_char['Label'] == gt_motion]['timestamp'])
        # return gtMotionTimestampList
        # print("Ground Truth", gtMotionTimestampList)
        return gtMotionTimestampList
        # else:
        #    return False

    ######### Acoustic Methods ##########

    def get_start_time(self, user_data):
        #print('user_data',user_data.head())
        start_data = user_data.loc[(user_data['Label'] == 'TSS')]
        print('start_data', start_data)
        startTime = int(start_data.iloc[0]['Char'])

        # end_data = user_data.loc[(user_data['Label'] == 'END')]
        # endTime = int(end_data.iloc[0]['timestamp'])

        end_data = user_data.loc[(user_data['Label'] == 'AC')]
        endTime = int(end_data.iloc[-1]['timestamp'])

        # print('start time', startTime)
        # print('end time', endTime)
        return startTime

    def process_audioStream(self,audioStartTime, audioStream):
        # audioTimeIndex=np.arange(audioStartTime, (audioStartTime + ((len(audioStream)-1)*0.02)), 0.02)
        # audioTimeIndex = np.arange(audioStartTime, (audioStartTime + ((len(audioStream)) * 0.022675736)), 0.022675736)
        audioTimeIndex = np.arange(audioStartTime, (audioStartTime + ((len(audioStream)) * (1/22.05))), (1/22.05))
        #audioTimeIndex = np.arange(audioStartTime, (audioStartTime + ((len(audioStream)) * (1 / 44.1))), (1 / 44.1))

        print("time index length: ", len(audioTimeIndex))
        print("stream length: ", len(audioStream))

        audioTimeIndex = audioTimeIndex.astype(np.int64)
        # print("audio s time: ",(audioTimeIndex[0]))
        # print("audio e time: ", audioTimeIndex[-1])

        audioStream = audioStream.assign(timestamp=pd.Series(audioTimeIndex))
        audioStream[['timestamp', 'amplitude']] = audioStream[
            ['timestamp', 'amplitude']].apply(pd.to_numeric)

        # print('AudioStartTime',audioStream['timestamp'].iloc[0])
        # print('AudioEndTime',audioStream['timestamp'].iloc[-1])

        return audioStream

    def get_acoustic_char_list(self,audioStartTime, audioStream, singleCharList):
        # Sample rate 44100Hz
        audioCharList = []

        #print('SensorStartTime', singleCharList[0].iloc[0]['timestamp'])
        #print('SensorEndTime', singleCharList[-1].iloc[-1]['timestamp'])

        # print("audio stream size",len(audioStream))
        for char in (singleCharList):
            charSensorStartTime = char.iloc[0]['timestamp']
            charSensorEndTime = char.iloc[-1]['timestamp']

            # print("charSensorStartTime", charSensorStartTime)
            # print("charSensorStartTime", charSensorEndTime)

            # print(audioStream['timestamp'])
            # t=audioStream['timestamp'].between(charSensorStartTime + 1, charSensorStartTime - 1)

            # print(t[t].index)
            # print(audioStream[audioStream.timestamp == charSensorStartTime].index)

            if (audioStream['timestamp'].iloc[0] < charSensorStartTime):
                if (audioStream['timestamp'].iloc[-1] > charSensorEndTime):
                    charAudioStartTimeIndex = audioStream[audioStream.timestamp == charSensorStartTime].index[0]
                    # print("charAudioStartTime", audioStream[audioStream.timestamp == charSensorStartTime])

                    charAudioEndTimeIndex = audioStream[audioStream.timestamp == charSensorEndTime].index[-1]
                    # print("charAudioEndTime", audioStream[audioStream.timestamp == charSensorEndTime])

                    charAudioStream = audioStream.iloc[charAudioStartTimeIndex:charAudioEndTimeIndex]

                    # print("charSensorStartTime", charSensorStartTime)
                    # print("charAudioStartTime", charAudioStream['timestamp'].iloc[0])

                    # print('charSensorStartTime ',charSensorStartTime)
                    # print('charSensorEndTime', charSensorEndTime)
                    # print('charAudioStartTime ',charAudioStream['timestamp'].iloc[0])
                    # print('charAudioEndtTime ', charAudioStream['timestamp'].iloc[-1])

                    audioCharList.append(charAudioStream)

        return audioCharList
        # pylab.plot(audioCharList[4])
        # print("size",len(audioCharList))
        # print("audio char ",audioCharList[10])

    def get_acoustic_char(self,audioStartTime, audioStream, singleChar):

        #print('SensorStartTime', singleChar.iloc[0]['timestamp'])
        #print('SensorEndTime', singleChar.iloc[-1]['timestamp'])

        # print("audio stream size",len(audioStream))
        charSensorStartTime = singleChar.iloc[0]['timestamp']
        charSensorEndTime = singleChar.iloc[-1]['timestamp']

        # print("charSensorStartTime", charSensorStartTime)
        # print("charSensorStartTime", charSensorEndTime)

        # print(audioStream['timestamp'])
        # t=audioStream['timestamp'].between(charSensorStartTime + 1, charSensorStartTime - 1)

        # print(t[t].index)
        # print(audioStream[audioStream.timestamp == charSensorStartTime].index)

        if (audioStream['timestamp'].iloc[0] < charSensorStartTime):
            if (audioStream['timestamp'].iloc[-1] > charSensorEndTime):
                charAudioStartTimeIndex = audioStream[audioStream.timestamp == charSensorStartTime].index[0]
                # print("charAudioStartTime", audioStream[audioStream.timestamp == charSensorStartTime])

                charAudioEndTimeIndex = audioStream[audioStream.timestamp == charSensorEndTime].index[-1]
                # print("charAudioEndTime", audioStream[audioStream.timestamp == charSensorEndTime])

                charAudioStream = audioStream.iloc[charAudioStartTimeIndex:charAudioEndTimeIndex]

                # print("charSensorStartTime", charSensorStartTime)
                # print("charAudioStartTime", charAudioStream['timestamp'].iloc[0])

                # print('charSensorStartTime ',charSensorStartTime)
                # print('charSensorEndTime', charSensorEndTime)
                #print('charAudioStartTime ',charAudioStream['timestamp'].iloc[0])
                #print('charAudioEndtTime ', charAudioStream['timestamp'].iloc[-1])
                return charAudioStream

        # pylab.plot(audioCharList[4])
        # print("size",len(audioCharList))
        # print("audio char ",audioCharList[10])

