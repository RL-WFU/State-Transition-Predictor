#LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

history = np.load("obs_transition_history.npy", allow_pickle=True).item()

model = load_model("ObsTransitionNetwork.keras")
model.summary()

#PLOTS
plt.plot(history['loss'], label='Training loss')
plt.plot(history['val_loss'], label='Validation loss')
plt.legend()
plt.show()
plt.clf()   # clear figure

epochs = range(1,len(history['mae'])+1)
plt.plot(epochs, history['mae'], label='Training mae')
plt.plot(epochs, history['val_mae'], label='Validation mae')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(history['val_mae'][-1])