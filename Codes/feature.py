import numpy as np  
import os
import cmath
import math
import timeit	

# FILE WITH ALL FEATURE EXTRACTION FUNCTIONS

def DFT(x): 
	x = np.asarray(x,dtype = float)
	N = x.shape[1]
	n = np.arange(N)
	M = np.exp(-2j*np.pi*np.outer(n,n)/N)
	return np.dot(x,M)


def FFT(x):
	x = np.asarray(x, dtype = float)
	N = x.shape[1]
	if N<32:
		return DFT(x)
	else:
		if  np.log2(N)%1 > 0:
			x = np.append(x,np.zeros((1,int(pow(2,np.ceil(np.log(N)/np.log(2))))-N)),axis =1)
			N = x.shape[1]
		X_even = FFT(x[:,::2])
	X_odd = FFT(x[:,1::2])
	factor = np.exp(-2j * np.pi * np.arange(N/2) / N)
	return np.append(X_even + factor * X_odd,X_even - factor * X_odd, axis = 1)


# def DFT_slow(x):
#     """Compute the discrete Fourier Transform of the 1D array x"""
#     x = np.asarray(x, dtype=float)
#     N = x.shape[0]
#     n = np.arange(N)
#     k = n.reshape((N, 1))
#     M = np.exp(-2j * np.pi * k * n / N)
#     return np.dot(M, x)

# def FFT(x):
#     """A recursive implementation of the 1D Cooley-Tukey FFT"""
#     x = np.asarray(x, dtype=float)
#     N = x.shape[0]
    
#     if N % 2 > 0:
#         raise ValueError("size of x must be a power of 2")
#     elif N <= 32:  # this cutoff should be optimized
#         return DFT_slow(x)
#     else:
#         X_even = FFT(x[::2])
#         X_odd = FFT(x[1::2])
#         factor = np.exp(-2j * np.pi * np.arange(N) / N)
#         return np.concatenate([X_even + factor[:N / 2] * X_odd,
#                                X_even + factor[N / 2:] * X_odd])

def fft_freqs(x, rate):
	N = x.shape[1]
	p1 = np.arange(0,(N-1)//2 +1) # adf
	p2 = np.arange(-(N//2),0)#df
	freqs = np.append(p1,p2)#dfa
	return freqs * rate / N

def spectral_centroid(x, rate):
	N = x.shape[1]
	magnitudes = np.abs(FFT(x))[:,:N//2 +1]
	freqs = fft_freqs(x, rate)[:N//2 +1]
	return np.sum(magnitudes*freqs)/np.sum(magnitudes)# return weighted mean

def spectral_flux(x1, x2): # input is in time domain NEED TO pass FFT to opt
	return np.sum(np.abs(FFT(x1)-FFT(x2))**2)
	
def hamming(n):
	return 0.54-0.46*np.cos(2*np.pi*np.arange(n)/n)

def cepstrum(x): # input is in time NEED TO pass FFT to opt
	return np.fft.ifft(np.log(np.abs(FFT(x))))
	
def pitch(x, rate): # passing TIME domain signal
	N = np.size(x)
	w = hamming(N)
	x = w*(x/(1.01*np.abs(np.amax(x)))) # normalizing and applying hamming
	y = cepstrum(x)[:N//2 +1]  # cesptrum is symmetric so we take half
	l = np.zeros((1,np.size(y)))
	l[15:]=1
	y1 = y*l
	y_val = np.real(y1)
	y_loc = np.argmax(y1) # y_loc gives pitch period
	pitch_period = 1 + y_loc
	return (1/pitch_period)*rate #pitch frequency

def zcr(x,rate):
    change=0 
    size = np.size(x)
    x1 = x - np.average(x)  #counts the zero crossings
    for i in range(size-1):
        if(x1[:,i]*x1[:,i+1]<0):
            change+=1
    return  change*rate/(float(size))
	
def spectral_spread(x, rate):
	return np.sum(np.abs(x - spectral_centroid(x,rate))**2)

def spectral_skew(x, rate):
	return np.sum(np.abs(x - spectral_centroid(x,rate))**3)

def spectral_kurt(x, rate):
	return np.sum(np.abs(x - spectral_centroid(x,rate))**4)

def energy(x,rate):
	return float(np.sum(np.abs(x)**2))/float(float(rate)/float(np.size(x)))

def spectral_entropy(x):
	N = np.size(x)
	psd = np.abs(FFT(x)[:,:N//2 +1])**2 # +ve psd as its a real signal
	psd = psd/np.sum(psd + 1e-3) # normalised psd
	return -np.sum(psd*np.log2(psd+ 1e-3))




def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
         nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
         winfunc=lambda x:numpy.ones((x,))):
	energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
	feat = numpy.log(feat)
	feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
	feat = lifter(feat,ceplifter)
	if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
	return feat
'''
def mtFeatureExtraction(signal, Fs, mtWin, mtStep, stWin, stStep): #mid level feautures
    """
    Mid-term feature extraction
    """

    mtWinRatio = int(round(mtWin / stStep))
    mtStepRatio = int(round(mtStep / stStep))

    mtFeatures = []

    stFeatures = stFeatureExtraction(signal, Fs, stWin, stStep)
    numOfFeatures = len(stFeatures)
    numOfStatistics = 2

    mtFeatures = []
    #for i in range(numOfStatistics * numOfFeatures + 1):
    for i in range(numOfStatistics * numOfFeatures):
        mtFeatures.append([])

    for i in range(numOfFeatures):        # for each of the short-term features:
        curPos = 0
        N = len(stFeatures[i])
        while (curPos < N):
            N1 = curPos
            N2 = curPos + mtWinRatio
            if N2 > N:
                N2 = N
            curStFeatures = stFeatures[i][N1:N2]

            mtFeatures[i].append(numpy.mean(curStFeatures))
            mtFeatures[i+numOfFeatures].append(numpy.std(curStFeatures))
            #mtFeatures[i+2*numOfFeatures].append(numpy.std(curStFeatures) / (numpy.mean(curStFeatures)+0.00000010))
            curPos += mtStepRatio

	return numpy.array(mtFeatures), stFeatures
'''	
