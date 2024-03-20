from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_mlp(dim, output_dim):
    # define our MLP network
    model = Sequential()
    model.add(Dense(output_dim, input_dim=dim, activation="relu"))
    for _ in range(10):
        model.add(Dense(output_dim, activation="relu"))
    # return our model
    return model


def create_256_mlp(dim, output_dim):
    # define our MLP network
    model = Sequential()
    model.add(Dense(4, input_dim=dim, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(output_dim))
    # return our model
    return model


if __name__ == "__main__":
    create_mlp(2)
