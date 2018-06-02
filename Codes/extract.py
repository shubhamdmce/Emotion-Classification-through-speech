import numpy as np
import scipy.io.wavfile
from feature import spectral_centroid, spectral_flux, spectral_spread, spectral_entropy, energy,zcr, spectral_skew, spectral_kurt
import os
import cmath
import math
import librosa
import librosa.feature
import csv

csvfile = "data.csv"
featureRow = ['emotion','centroid','spread','entropy','skew','kurt','mfcc']

directory = '/home/dnyaneshwar/Desktop/DSP_project/DC'
frame_time = 40
samples_per_frame = 22050
centroid_list = []
flux_list = []
pitch_list = []
x_old = np.zeros((1,samples_per_frame))
x_old2 = np.zeros((1,samples_per_frame))

hop = 4

x = np.random.random((1,samples_per_frame))
count=0
temp=0
with open(csvfile, "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(featureRow)
    for filename in os.listdir(directory):
        count=count+1
        if (count>15):
            count=1
            temp=temp+1
        if filename.endswith('.wav'):
            print(os.path.join(directory,filename))
            data,rate = librosa.load(os.path.join(directory,filename))
            #print(rate)
            #print(data)
            N = int(np.size(data)/samples_per_frame)
            #print(N)
            #break
            for i in range(N-3):
                #print(i)
                for j in range(hop):
                    #print(j)
                    print(int(j/hop)*samples_per_frame)
                    x[0,:] = data[i*samples_per_frame + int(j/hop)*samples_per_frame:(i+1)*samples_per_frame +int(j/hop)*samples_per_frame]
                    #print(data.shape)
                    #print(x[0,:].shape)
                    #print(x.shape)
                    #break
                    centroid = spectral_centroid(x,rate)
                    skew = spectral_skew(x,rate)
                    kurt = spectral_kurt(x,rate)
                    flux = spectral_flux(x,x_old)
                    zc = zcr(x,rate)
                    spread = spectral_spread(x, rate)
                    entropy = spectral_entropy(x)
                    ener = energy(x, rate)
                    coeffs = np.average((librosa.feature.mfcc(x[0,:],sr=rate,hop_lengtht=512)),axis=1)
                    wr.writerow(np.append([temp,centroid,spread,entropy,skew,kurt],coeffs))
                    x_old = x
        else:
            continue
	








    
