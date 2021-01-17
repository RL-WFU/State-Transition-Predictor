#LIBRARIES
import numpy as np
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model,load_model
import tensorflow as tf
from keras.utils import to_categorical
from keras.losses import kullback_leibler_divergence
from keras.losses import CategoricalCrossentropy
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix,accuracy_score



# convert an array of values into a timeseries of 3 previous steps matrix
def create_timeseries(data):
    dataX = []
    dataY = []
    for i in range(3,len(data)):
        if i%25 > 1:
            a = np.vstack((data[i - 3], data[i - 2],data[i - 1]))
            dataX.append(a)
            dataY.append(data[i])
    return np.array(dataX), np.array(dataY)


#same results for same model, makes it deterministic
np.random.seed(1234)
tf.random.set_seed(1234)


#reading data
input = np.load("Transition_new.npy", allow_pickle=True)
pre = np.asarray(input[:,0])
a1 = np.asarray(input[:,1])
a2 = np.asarray(input[:,2])
a3 = np.asarray(input[:,3])
post = np.asarray(input[:,4])

#flattens the np arrays
pre = np.concatenate(pre).ravel()
pre = np.reshape(pre, (pre.shape[0]//54,54))
post = np.concatenate(post).ravel()
post = np.reshape(post, (post.shape[0]//54,54))

data = np.column_stack((pre,a1.T,a2.T,a3.T))
print(data.shape)
print(data[0])



#reshapes trainX to be timeseries data with 3 previous timesteps
#LSTM requires time series data, so this reshapes for LSTM purposes
#X has 200000 samples, 3 timestep, 55 features
inputX, inputY = create_timeseries(data)
inputX = inputX.astype('float64')
inputY = inputY.astype('float64')


trainX = inputX[:180000]
trainY = inputY[:180000]
valX = inputX[180000:]
valY = inputY[180000:]


valYpre = valY[:,:54]
valY1 = to_categorical(valY[:,-3])
valY2 = to_categorical(valY[:,-2])
valY3 = to_categorical(valY[:,-1])
trainYpre = trainY[:,:54]
trainY1 = to_categorical(trainY[:,-3])
trainY2 = to_categorical(trainY[:,-2])
trainY3 = to_categorical(trainY[:,-1])



#build functional model
visible =Input(shape=(trainX.shape[1],trainX.shape[2]))
hidden1 = LSTM(100, return_sequences=True)(visible)
hidden2 = LSTM(64,return_sequences=True)(hidden1)
#first agent branch
hiddenAgent1 = LSTM(16, name='firstBranch')(hidden2)
agent1 = Dense(valY1.shape[1],activation='softmax',name='agent1classifier')(hiddenAgent1)
#second agent branch
hiddenAgent2 = LSTM(16, name='secondBranch')(hidden2)
agent2 = Dense(valY2.shape[1],activation='softmax',name='agent2classifier')(hiddenAgent2)
#third agent branch
hiddenAgent3 = LSTM(16, name='thirdBranch')(hidden2)
agent3 = Dense(valY3.shape[1],activation='softmax',name='agent3classifier')(hiddenAgent3)
#observation branch
hiddenObservation = LSTM(64, name='observationBranch')(hidden2)
observation = Dense(valYpre.shape[1], name='observationScalar')(hiddenObservation)



model = Model(inputs=visible,outputs=[agent1,agent2,agent3,observation])

model.compile(optimizer='adam',
              loss={'agent1classifier': 'categorical_crossentropy',
                  'agent2classifier': 'categorical_crossentropy',
                    'agent3classifier': 'categorical_crossentropy',
                    'observationScalar': 'mse'},
              metrics={'agent1classifier': ['acc'],
                       'agent2classifier': ['acc'],
                        'agent3classifier': ['acc'],
                       'observationScalar': ['mae']})
print(model.summary())


history = model.fit(trainX,
                    y={'agent1classifier': trainY1,
                       'agent2classifier':trainY2,
                       'agent3classifier':trainY3,
                       'observationScalar': trainYpre}, epochs=200, batch_size=5000, verbose=2,
                    validation_data = (valX,
                                       {'agent1classifier': valY1,
                                        'agent2classifier': valY2,
                                        'agent3classifier': valY3,
                                        'observationScalar': valYpre}),shuffle=False)

model.save('StateTransitionNetwork1.keras')


#model = load_model("actionMultiClassNetwork.keras")


np.save("state_transition_history1.npy", history.history, allow_pickle=True)
