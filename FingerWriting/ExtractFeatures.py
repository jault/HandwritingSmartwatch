import math
import numpy as np
import matplotlib.pyplot as plt

import tsfresh.feature_extraction.feature_calculators as fc
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import butter, sosfilt, lfilter

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
        load_robot_execution_failures
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
import pandas as pd
eps = 0.00000001
countering = 0

class ExtractFeatures():

    def zero_crossing_rate(self, frame):
        zero_crossings = 0
        zero_crossing_rate=0
        for i in range(1,len(frame)):
            if (frame.iloc[i - 1] < 0 and frame.iloc[i] > 0) or \
                    (frame.iloc[i - 1] > 0 and frame.iloc[i] < 0) or \
                    (frame.iloc[i - 1] != 0 and frame.iloc[i] == 0):
                zero_crossings += 1

        if(len(frame)>=2):
            zero_crossing_rate = zero_crossings / float(frame.count() - 1)
        #print('zcr',zero_crossing_rate)
        return zero_crossing_rate



    def frame_filter(self, X):
        # First, design the Buterworth filter
        N = 2  # Filter order
        Wn = 1  # Cutoff frequency
        B, A = signal.butter(N, Wn, output='ba')

        # Second, apply the filter
        if(len(X)>0):
            X = signal.filtfilt(B, A, X,padtype=None)

        return X

    def butter_lowpass(self,cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self,data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y


    def DCMean(self,x):
        x=self.butter_lowpass_filter(x,1,200)
        return fc.mean(x)

    #Same as DCMean but computed over the summation of all the acceleration
    #signals over all axis and accelerometers available (sensors). This feature
    #captures the overall posture information contained in the DC component of
    #the acceleration signals.
    def DCTotalMean(self,frame):
        return (self.DCMean(frame['xAxis'])
                + self.DCMean(frame['yAxis'])
                                            + self.DCMean(frame['zAxis']))/3

    #The area under the signal simply computed by summing the acceleration samples
    # contained in a given window.
    def DCArea(self,x):
        return x.sum()

    #The differences between the mean values of the X-Y, X-Z, and Y-Z
    #acceleration axis per sensor. These three values capture the orientation of
    #the sensor with respect to ground or body posture information. The feature
    #is computed after low-pass filtering the acceleration signals at 1Hz.
    def DCPostureDist(self,frame):
        xy=self.DCMean(frame['xAxis'])-self.DCMean(frame['yAxis'])
        xz = self.DCMean(frame['xAxis']) - self.DCMean(frame['zAxis'])
        yz = self.DCMean(frame['yAxis']) - self.DCMean(frame['zAxis'])

        return xy,xz,yz

    def ACEnergy(self,x):
        x = self.butter_bandpass_filter(x, 0.1, 20, 200)
        # n=len(frame)
        # Y=fft(frame)/n
        Y = abs(fft(x))
        Y = Y / len(Y)
        return fc.abs_energy(Y[range(1, len(Y))])  # ACEnergy

    #The energy is computed from the FFT coefficients computed
    #over the band-pass filtered (0.1-20Hz) accelerometer signals.
    def ACLowEnergy(self, x):
        x = self.butter_bandpass_filter(x, 0.1, 20, 200)
        # n=len(frame)
        # Y=fft(frame)/n
        Y = abs(fft(x))
        Y = Y / len(Y)

        """
        Returns the absolute energy of the time series which is the sum over the squared values
        the sum of the energy contained between frequencies of 0.3 – 3.5Hz.
        .. math::

            E = \\sum_{i=1,\ldots, n} x_i^2

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        #xx=x.between(0.3, 3.5)
        #x=x[xx]
        # if not isinstance(x, (np.ndarray, pd.Series)):
        #     x = np.asarray(x)
        return np.dot(Y, Y)


    #Mean of average over the absolute value of the band-pass filtered (0.1-20Hz) accelerometer signals.
    #Acceleration can have positive and negative values, so computing the absolute value guarantees the mean
    #will not be zero for perforect oscillatory motion with equal positive and negative magnitudes
    def ACAbsMean(self, x):
        x=self.butter_bandpass_filter(x,0.1,20,200)
        return fc.mean(np.absolute(x))

    #The area under the absolute value of the signal computed by simply summing
    # the accelerometer samples inside a given window. The sum is computed after
    # band-pass filtering (0.1-20Hz) the accelerometer signals. Acceleration can
    # have positive and negative values, so computing the absolute value guarantees
    # the area will not be zero for perfect oscillatory motion with equal positive
    # and negative acceleration magnitudes.
    def ACAbsArea(self,x):
        x = self.butter_bandpass_filter(x, 0.1, 20, 200)
        return np.sum(np.absolute(x))

    #Same as ACAbsArea but computed over the summation of all the signals over all axis
    # and accelerometers available (sensors). This feature captures the overall motion
    # experienced by the human body as experienced by all the accelerometers worn.
    def ACTotalAbsArea(self,frame):
        return self.ACAbsArea(frame['xAxis']) + \
               self.ACAbsArea(frame['yAxis']) + \
               self.ACAbsArea(frame['zAxis'])

    #The variance of the accelerometer signal computed over a given window. It
    #is computed after band-pass filtering (0.1-20Hz) the accelerometer signals.
    def ACVar(selfself,frame):
        pass


    #Computed as the ratio of the standard deviation and the mean over each
    #signal window multiplied by 100. This measures the dispersion of the
    #acceleration signal. Acceleration is band-pass filtered (0.1-20Hz) before
    #computing this feature.
    def ACAbsCV(selfself,frame):
        pass

    #Computed as the difference between quartiles three and one (Q3-Q1). This
    #value describes the dispersion of the acceleration signal. This feature is
    #computed after band-pass filtering (0.1-20Hz) the accelerometer signals.
    def ACIQR(self, frame):
        pass

    #Difference between the maximum and minimum values of the
    #accelerometer signal over a given window. This is a measure of peak
    #acceleration or maximum motion inside the window. Accelerometer signals
    #are band-pass filtered (0.1-20Hz) before computing this feature.
    def acrange(selfself,frame):
        pass

    #page 286
    def window_summary_char(self, frame,char):

        return [
            char,
            # Freq
            self.ACEnergy(frame), #ACEnergy
            self.ACLowEnergy(frame),  #ACLowEnergy
            self.DCMean(frame), #DCMean
            self.DCArea(frame), #DCArea
            self.ACAbsMean(frame), #ACAbsMean
            self.ACAbsArea(frame) #ACAbsArea
        ]

    def window_summary_char_2(self, frame):

        return [

            # Freq
            self.ACEnergy(frame), #ACEnergy
            self.ACLowEnergy(frame),  #ACLowEnergy
            self.DCMean(frame), #DCMean
            self.DCArea(frame), #DCArea
            self.ACAbsMean(frame), #ACAbsMean
            self.ACAbsArea(frame)  # ACAbsArea
        ]

    def remap(self, axis):
        mapped = [100+axis[0]]
        for i in range(len(axis)-1):
            mapped.append(mapped[-1]+axis[i+1])
        return np.asarray(mapped)
                
    def resample(self, xaxis, yaxis):
        ymin = yaxis[0]
        ymax = yaxis[0]
        xmin = xaxis[0]
        xmax = xaxis[0]
        for i in range(len(xaxis)):
            if yaxis[i] < ymin:
                ymin = yaxis[i]
            if yaxis[i] > ymax:
                ymax = yaxis[i]
            if xaxis[i] < xmin:
                xmin = xaxis[i]
            if xaxis[i] > xmax:
                xmax = xaxis[i]
        S = math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2) / 40.0
        D = 0
        xresampled = [xaxis[0]]
        yresampled = [yaxis[0]]
        i = 1
        for i in range(1, len(xaxis)):
            d = math.sqrt((xaxis[i]-xaxis[i-1])**2 + (yaxis[i]-yaxis[i-1])**2)
            if D+d >= S:
                qx = xaxis[i-1] + ((S - D)/d) * (xaxis[i] - xaxis[i-1])
                qy = yaxis[i-1] + ((S - D)/d) * (yaxis[i] - yaxis[i-1])
                xresampled.append(qx)
                yresampled.append(qy)
                #stroke.splice(i, 0, q)
                D = 0
            else:
                D = D + d
        return xresampled, yresampled
     
    def rubine_long(self, frame, type):
        features = []
        rexaxis = self.remap(np.array(frame['xAxis']))
        reyaxis = self.remap(np.array(frame['yAxis']))
        xaxis, yaxis = self.resample(rexaxis, reyaxis)
        
        global countering
        if countering < 153:
            char = frame['Char'].iloc[0]
            plt.title(type + ' Unmapped ' + char)
            plt.plot(np.array(frame['xAxis']),np.array(frame['yAxis']), 'o', color='black')
            #plt.show()
            plt.savefig('./imgs/'+char+'_'+type+'_unmapped.png')
            plt.clf()
            plt.title(type + ' Remapped ' + char)
            plt.plot(rexaxis,reyaxis, 'o', color='black')
            #plt.show()
            plt.savefig('./imgs/'+char+'_'+type+'_remapped.png')
            plt.clf()
            plt.title(type + ' Resampled ' + char)
            plt.plot(xaxis,yaxis, 'o', color='black')
            #plt.show()
            plt.savefig('./imgs/'+char+'_'+type+'_resampled.png')
            plt.clf()
            countering += 1
        

        r5 = self.r5(xaxis, yaxis)
        r3, r4, l18 = self.r3_r4_l18(xaxis, yaxis)
        r8 = self.r8(xaxis, yaxis)
        r9, r10, r11, l13 = self.r9_r10_r11_l13(xaxis, yaxis)
        l12 = self.l12(xaxis,yaxis,r4)
        features.append(self.r1(xaxis, yaxis))
        features.append(self.r2(xaxis, yaxis))
        features += [r3, r4, r5]
        features.append(self.r6(xaxis, yaxis, r5))
        features.append(self.r7(xaxis, yaxis, r5))
        features += [r8, r9, r10, r11, l12, l13]
        features.append(self.l14(r9, r8))
        features.append(self.l15(r8, r5))
        features.append(self.l16(r8, r3))
        features.append(self.l17(r5, r3))
        features.append(l18)
        features.append(self.l19(l18))
        features.append(self.l20(r9, r10))
        features.append(self.l21(r8))
        features.append(self.l22(l12))
        
        return features

    def r1(self, xaxis, yaxis):
        n=1
        for i in range(1, len(xaxis)):
            dist = math.sqrt((xaxis[i]-xaxis[0])**2+(yaxis[i]-yaxis[0])**2)
            if dist > 0.00001:
                n = i
                break
        xd = (xaxis[n] - xaxis[0])**2
        yd = (yaxis[n] - yaxis[0])**2
        return xd / math.sqrt(xd+yd)

    def r2(self, xaxis, yaxis):
        n=1
        for i in range(1, len(xaxis)):
            dist = math.sqrt((xaxis[i]-xaxis[0])**2+(yaxis[i]-yaxis[0])**2)
            if dist > 0.00001:
                n = i
                break
        xd = (xaxis[n] - xaxis[0])**2
        yd = (yaxis[n] - yaxis[0])**2
        return yd / math.sqrt(xd+yd)
        
    def r3_r4_l18(self, xaxis, yaxis):
        ymin = yaxis[0]
        ymax = yaxis[0]
        xmin = xaxis[0]
        xmax = xaxis[0]
        for i in range(len(xaxis)):
            if yaxis[i] < ymin:
                ymin = yaxis[i]
            if yaxis[i] > ymax:
                ymax = yaxis[i]
            if xaxis[i] < xmin:
                xmin = xaxis[i]
            if xaxis[i] > xmax:
                xmax = xaxis[i]
        yd2 = (ymax-ymin)**2
        xd2 = (xmax-xmin)**2
        yd = ymax - ymin
        xd = xmax-xmin
        return math.sqrt(yd2+xd2), math.atan(yd/xd), yd*xd

    def r5(self, xaxis, yaxis):
        y=(yaxis[len(yaxis)-1]-yaxis[0])**2
        x=(xaxis[len(xaxis)-1]-xaxis[0])**2
        return math.sqrt(x+y)

    def r6(self, xaxis, yaxis, r5):
        x=(xaxis[len(xaxis)-1]-xaxis[0])
        return x/(r5+0.000001)

    def r7(self, xaxis, yaxis, r5):
        y=(yaxis[len(yaxis)-1]-yaxis[0])
        return y/(r5+0.000001)

    def r8(self, xaxis, yaxis):
        total = 0
        for i in range(1, len(xaxis)):
            xd = (xaxis[i] - xaxis[i-1])**2
            yd = (yaxis[i] - yaxis[i-1])**2
            total += math.sqrt(xd+yd)
        return total
        
    def r9_r10_r11_l13(self, xaxis, yaxis):
        sum, abssum, sqrsum, smallsum = 0, 0, 0, 0
        for i in range (1, len(xaxis)-1):
            xi = xaxis[i+1]-xaxis[i]
            xj = xaxis[i]-xaxis[i-1]
            yi = yaxis[i+1]-yaxis[i]
            yj = yaxis[i]-yaxis[i-1]
            numer = xi*yj - xj*yi
            denom = xi*xj + yi*yj + 0.000001
            angle = math.atan(numer/denom)
            if angle > math.pi:
                angle = angle - math.pi
            if angle < -math.pi:
                angle = angle + math.pi
            sum += angle
            abssum += abs(angle)
            sqrsum += angle**2
            if abs(angle) < 19*(math.pi/180):
                smallsum += abs(angle)
        return [sum, abssum, sqrsum, smallsum]
        
    def l12(self, xaxis, yaxis, r4):
        return abs(math.pi/4 - r4)
        
    def l14(self, r9, r8):
        return r9/(r8+0.000001)
        
    def l15(self, r8, r5):
        return r8/(r5+0.000001)
        
    def l16(self, r8, r3):
        return r8/(r3+0.000001)
        
    def l17(self, r5, r3):
        return r5/(r3+0.000001)
        
    def l19(self, l18):
        return math.log(l18+0.000001)
        
    def l20(self, r9, r10):
        return r9/(r10+0.000001)
        
    def l21(self, r8):
        return math.log(r8+0.000001)
        
    def l22(self, l12):
        return math.log(l12+0.000001)

    #Character Classification

    def char_features(self,char, type):
        #print(char)
        features = []
        if(not char.empty):
            for axis in ['xAxis', 'yAxis', 'zAxis']:
                if axis == 'xAxis':
                        features += self.window_summary_char(char[axis],char['Char'].iloc[0])
                else:
                    features += self.window_summary_char_2(char[axis])

            features.append(self.DCTotalMean(char))
            features+=self.DCPostureDist(char)
            features.append(self.ACTotalAbsArea(char))
            features += self.rubine_long(char, type)
            
        yield features

    def get_feature_set_char(self,charList,type):
        return self.char_features(charList, type)




# http://www.ifs.tuwien.ac.at/~schindler/lectures/MIR_Feature_Extraction.html

# https://github.com/tyiannak/recognizeFitExercise/blob/master/accelerometer.py

#Freq
#https://glowingpython.blogspot.com/2011/08/how-to-plot-frequency-spectrum-with.html

#Freq Extraction in stFeatureExtraction in audioFeatureExtraction.py
#https://github.com/tyiannak/pyAudioAnalysis/blob/master/audioFeatureExtraction.py
