#LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

history = np.load("state_transition_history.npy", allow_pickle=True).item()

model = load_model("StateTransitionNetwork.keras")
model.summary()

#PLOTS
plt.plot(history['agent1classifier_loss'], label='Training Agent1 loss')
plt.plot(history['agent2classifier_loss'], label='Training Agent2 loss')
plt.plot(history['agent3classifier_loss'], label='Training Agent3 loss')
plt.plot(history['val_agent1classifier_loss'], label='Validation Agent1 loss')
plt.plot(history['val_agent2classifier_loss'], label='Validation Agent2 loss')
plt.plot(history['val_agent3classifier_loss'], label='Validation Agent3 loss')
plt.title("State-Transition Loss")
plt.legend()
plt.show()

plt.clf()   # clear figure

plt.plot(history['agent1classifier_acc'], label='Training Agent1 acc')
plt.plot(history['agent2classifier_acc'], label='Training Agent2 acc')
plt.plot(history['agent3classifier_acc'], label='Training Agent3 acc')
plt.plot(history['val_agent1classifier_acc'], label='Validation Agent1 acc')
plt.plot(history['val_agent2classifier_acc'], label='Validation Agent2 acc')
plt.plot(history['val_agent3classifier_acc'], label='Validation Agent3 acc')
plt.title("State-Transition Accuracy")
plt.legend()
plt.show()

plt.clf()   # clear figure

plt.plot(history['observationScalar_mae'], label='Training Observation MAE')
plt.plot(history['val_observationScalar_mae'], label='Validation Observation MAE')
plt.title("Observation MAE")
plt.legend()
plt.show()


print(history['val_agent3classifier_acc'][-1])
print(history['val_agent2classifier_acc'][-1])
print(history['val_agent1classifier_acc'][-1])
print(history['val_observationScalar_mae'][-1])