import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from ast import literal_eval
from sklearn.model_selection import train_test_split

from yads.predictors.Neural_Networks.create_mlp import create_256_mlp
from yads.predictors.Neural_Networks.create_cnn import create_cnn_1D
from tensorflow.keras.layers import (
    concatenate,
    Dense,
    Conv1DTranspose,
    Activation,
    BatchNormalization,
    MaxPooling1D,
    Flatten,
    Dropout,
)
from tensorflow.keras.models import Model
import tensorflow as tf


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 10),
):
    """
    from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Loss")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        scoring="neg_mean_squared_error",
    )
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = -np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")
    # axes[0].set_yscale('symlog')
    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Performance of the model")
    return plt


def create_multi_input_model():
    filters = (32, 64, 128, 256)
    mlp_model = create_cnn_1D(1, map_shape[1], 128)
    cnn_model = create_256_mlp(2, 128)

    print("MLP:", mlp_model.input.shape, trainAttrX.shape)
    print("CNN:", cnn_model.input.shape, S0_train.shape)

    concatenated = concatenate([mlp_model.output, cnn_model.output])
    concatenated = tf.reshape(concatenated, [-1, 1, concatenated.shape[1]])
    chanDim = -1
    x = None
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        # CONV => RELU => BN => POOL
        if i == 0:
            x = concatenated
        x = Conv1DTranspose(f, kernel_size=3, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling1D(pool_size=2, strides=1, padding="same")(x)
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    return Model(inputs=[mlp_model.input, cnn_model.input], outputs=x)


def single_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 10),
):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=False,
        scoring="neg_mean_squared_error",
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = -np.std(test_scores, axis=1)

    fig = plt.figure(figsize=(8, 8))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    # Plot learning curve
    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.legend(loc="best")
    plt.ylabel("MSE")
    plt.title(title)
    plt.xlabel("Training examples")
    return fig


if __name__ == "__main__":
    df = pd.read_csv(
        "data/third_approach_2000.csv",
        converters={"S": literal_eval, "P": literal_eval, "S0": literal_eval},
    )
    df["log_q"] = -np.log10(-df["q"])

    X = df[["log_q", "dt", "S0"]]
    y = df[["S"]]

    map_shape = (y.shape[0], len(df["S0"].loc[0]))

    (trainX, testX, trainY, testY) = train_test_split(
        X, y, test_size=0.01, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(trainX[["log_q", "dt"]])
    trainX[["log_q", "dt"]], testX[["log_q", "dt"]] = (
        scaler.transform(trainX[["log_q", "dt"]]),
        scaler.transform(testX[["log_q", "dt"]]),
    )

    trainAttrX, testAttrX = (
        trainX[["log_q", "dt"]].to_numpy(),
        testX[["log_q", "dt"]].to_numpy(),
    )
    trainY, testY = np.array(list(trainY["S"])), np.array(list(testY["S"]))

    trainY_reshape = trainY.reshape((trainY.shape[0], 1, trainY.shape[1]))
    testY_reshape = testY.reshape((testY.shape[0], 1, testY.shape[1]))

    S0_train, S0_test = np.array(list(trainX["S0"])), np.array(list(testX["S0"]))

    S0_train = S0_train.reshape((S0_train.shape[0], 1, S0_train.shape[1]))
    S0_test = S0_test.reshape((S0_test.shape[0], 1, S0_test.shape[1]))

    trainAttrX = trainAttrX.reshape((trainAttrX.shape[1], trainAttrX.shape[0]))
    title = r"Learning Curves (MLP)"
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    model = KerasRegressor(
        build_fn=create_multi_input_model, epochs=30, batch_size=10, verbose=0
    )
    single_learning_curve(
        model,
        title,
        X=[S0_train, trainAttrX],
        y=trainY_reshape,
        ylim=None,
        cv=cv,
        n_jobs=4,
        train_sizes=np.linspace(0.01, 0.30, 30),
    )

    plt.show()
