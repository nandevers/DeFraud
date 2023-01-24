import yaml
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from sklearn.model_selection import KFold




def build_model(state_size, action_size, learning_rate):
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(action_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))
    return model

