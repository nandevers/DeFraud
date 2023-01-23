from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

optimizer_dict = {'adam': Adam, 'sgd': SGD}

def build_model(config_file, X, y):
    # Read YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Parse YAML
    learning_rate = config['learning_rate']
    loss = config['loss']
    optimizer_name = config['optimizer']
    optimizer_params = config['optimizer_params']
    layers = config['layers']
    early_stopping = config['early_stopping']
    cross_val = config['cross_val']

    # Define a dictionary of layer types and corresponding functions
    layer_dict = {'dense': Dense, 'batch_normalization': BatchNormalization, 'dropout': Dropout, 'lambda': Lambda}

    # Build network
    model = Sequential()
    for layer in layers:
        layer_type = layer.pop('type')
        layer_func = layer_dict[layer_type]
        model.add(layer_func(**layer))

    optimizer = optimizer_dict[optimizer_name](learning_rate=learning_rate, **optimizer_params)
    model.compile(loss=loss, optimizer=optimizer)

