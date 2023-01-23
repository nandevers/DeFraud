import random
import itertools
import yaml
from sklearn.model_selection import KFold

# Define the range of possible architectures
layer_types = ["dense", "batch_normalization", "dropout"]
n_layers = [2, 5, 8, 10 , 50]
units = [24, 32, 64, 128]
activations = ["relu", "tanh"]
dropout_rates = [0.2, 0.5]
losses = ["mse", "binary_crossentropy"]
optimizers = ["adam", "sgd"]
optimizer_params = [{"momentum": 0.9}, {"beta_1": 0.9, "beta_2": 0.999}]
early_stopping = [
    {"monitor": "val_loss", "patience": 10},
    {"monitor": "val_acc", "patience": 10},
]
learning_rate = [0.001, 0.01, 0.1]
cross_val = [None, 5, 7, 10]

sample_size = 5

main_iter = itertools.product(
        n_layers,
        layer_types,
        units,
        activations,
        dropout_rates,
        losses,
        optimizers,
        optimizer_params,
        early_stopping,
        cross_val,
        learning_rate
    )
sample = random.sample(list(main_iter), sample_size)

architectures=[]
for n, layer_type, unit, activation, dropout_rate, loss, optimizer, optimizer_param, early_stop, cv, lr in sample:
    layers = []
    for i in range(n):
        layer = {"type": layer_type, "units": unit, "activation": activation}
        if layer_type == "dropout":
            layer["rate"] = dropout_rate
        layers.append(layer)
        if layer_type == "dense":
            layers.append({"type": "batch_normalization"})
    architecture = {
        "nn":{
            "layers": layers,
            "learning_rate": lr,
            "loss": loss,
            "optimizer": optimizer,
            "optimizer_params": optimizer_param
        },
        "tunning":{
            "early_stopping": early_stop,
            "cross_val": cv
        }
    }
    architectures.append(architecture)


# Write architectures to YAML files
for i, architecture in enumerate(architectures):
    with open(f"experiments/architectures/architecture_{i}.yaml", "w") as f:
        yaml.dump(architecture, f)
