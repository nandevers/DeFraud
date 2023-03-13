import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

# from kerastuner import Hyperband, HyperParameter, HyperModel
# from sklearn.model_selection import KFold, cross_val_score

tf.random.set_seed(2)


loss_functions = [
    "mse",
    "mae",
    "mape",
    "binary_crossentropy",
    "categorical_crossentropy",
    "sparse_categorical_crossentropy",
    "kullback_leibler_divergence",
    "poisson",
    "cosine_proximity",
    "huber_loss",
]

LOSS = {k: k for k in loss_functions}


def build_baseline_model(state_size, action_size, learning_rate):
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation="linear"))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(action_size, activation="linear"))
    model.compile(
        loss="mae",
        optimizer=Adam(learning_rate=learning_rate),
    )
    return model


def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, "float"), axis=0)
    tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), "float"), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, "float"), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), "float"), axis=0)

    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return tf.reduce_mean(f1)


def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # define the number of positive cases
    pos_num = tf.reduce_sum(y_true)
    neg_num = tf.reduce_sum(1 - y_true)

    # calculate false positive rate
    fp = tf.reduce_sum((1 - y_true) * y_pred) / neg_num

    # calculate false negative rate
    fn = tf.reduce_sum(y_true * (1 - y_pred)) / pos_num

    # combine false positive and false negative rates
    loss = fp + fn

    return loss


LOSS["custom"] = custom_loss
LOSS["f1_score"] = f1_score


# class MyHyperModel(HyperModel):
#    def __init__(self, x_train, y_train):
#        self.x_train = x_train
#        self.y_train = y_train
#        # self.loss = LOSS[loss]
#
#    def build_model(self, hp):
#        units = hp.Int("units", min_value=32, max_value=512, step=32)
#        activation = hp.Choice("activation", ["relu", "tanh"])
#        dropout = hp.Boolean("dropout")
#        lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
#        state_size = self.x_train.shape[1]
#        action_size = self.y_train.shape[1]
#        model = Sequential()
#        model.add(Dense(24, input_dim=state_size, activation="relu"))
#        model.add(Dense(units=units, activation=activation))
#        if dropout:
#            model.add(Dropout(rate=0.25))
#        model.add(Dense(action_size, activation="sigmoid"))
#
#        model.compile(
#            optimizer=Adam(learning_rate=lr), loss="mse", metrics=["accuracy"]
#        )
#        return model
#
#    def run_tuner(self, epochs=100, objective="val_accuracy"):
#        self.tuner = Hyperband(
#            self.build_model,
#            objective=objective,
#            max_epochs=epochs,
#            factor=3,
#            directory="logs",
#            project_name="defraud",
#        )
#
#        self.tuner.search_space_summary()
#        self.tuner.search(self.x_train, self.y_train, epochs=epochs, verbose=1)
#
#        best_model = self.tuner.get_best_models(num_models=1)[0]
#        best_hyperparameters = self.tuner.get_best_hyperparameters(num_models=1)[0]
#        print("Best Hyperparameters:", best_hyperparameters.values)
#        return best_model
