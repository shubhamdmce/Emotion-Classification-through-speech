import numpy as np   
import csv
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
import numpy

numpy.random.seed(7)

percentTrain = 0.7
cry = np.genfromtxt("adult_cry2.csv", delimiter=',', skip_header=True)
laugh = np.genfromtxt("adult_laugh2.csv", delimiter=',', skip_header=True)


print (laugh.shape)
print (cry.shape)


# CALCULATING THE MID-LEVEL FEATURES
numStats = 15

nC = cry.shape[0]
nL = laugh.shape[0]
cry2 = np.ndarray((nC,2*cry.shape[1]))
laugh2 = np.ndarray((nL,2*laugh.shape[1]))




for i in range(nC):
	cry2[i,:] = np.append(cry[i,:],np.average(cry[i%nC:(i+numStats)%nC,:],axis = 0))

for i in range(nL):
	laugh2[i,:] = np.append(laugh[i,:],np.average(laugh[i%nC:(i+numStats)%nC,:],axis = 0))

cry = cry2
laugh = laugh2


data = np.append(cry,laugh,axis=0)
data = shuffle(data,random_state=0) # randomized the frames
print (data.shape)

data = np.nan_to_num(data)

'''
le = LabelEncoder()
data[:,0] = le.fit_transform(data[:,0])
print list(le.classes_)
'''

#print data[5:10,:]
#SPLITTING THE DATA INTO TRAIN AND TEST

train = data[:int(percentTrain*data.shape[0]),:]
test = data[int(percentTrain*data.shape[0])+1:,:]


Xtrain = train[:,1:]
Ytrain = train[:,0]
Xtest = test[:,1:]
Ytest = test[:,0]

num_feat = Xtrain.shape[1]
#print Xtrain[5,:], Xtest[5,:], Ytrain[5,:], Ytest[5,:]


#rbf_kernel_svm_clf = Pipeline((("scaler",StandardScaler()),("svm_clf",SVC(kernel='rbf', gamma=1000, C=1000))))

# SVM CLASSFIER
rbf_kernel_svm_clf = SVC(kernel='linear')
scores = cross_val_score(rbf_kernel_svm_clf, Xtrain, Ytrain, cv=2)
print (scores)

#rbf_kernel_svm_clf.fit(Xtrain,Ytrain)
'''
# MLP CLASSIFIER
model = Sequential()
model.add(Dense(25, input_dim= num_feat, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(Xtrain, Ytrain, epochs=50, batch_size=1000)

scores = model.evaluate(Xtrain, Ytrain)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(Xtest, Ytest)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))'''