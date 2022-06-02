from keras import models, layers, metrics
from tensorflow.keras import optimizers

def create_model(input_shape, dropout=None, lr=0.001):
    inp = layers.Input(input_shape)
    x = layers.Flatten()(inp)
    x = layers.Dense(512, activation='relu')(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(1024, activation='relu')(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(2048, activation='relu')(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(5112, activation='relu')(x)
    x = layers.Reshape((71, 72))(x)

    optimizer = optimizers.Adam(learning_rate=lr)

    model = models.Model(inp, x)
    model.compile(
        loss='mse', 
        optimizer=optimizer, 
        metrics=['mse', 'mae', metrics.RootMeanSquaredError()]
    )
    return model

