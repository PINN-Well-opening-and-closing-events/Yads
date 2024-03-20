import tensorflow as tf
from yads.thesis_approaches.data_generation import check_conservative
from yads.wells import Well
import json


def mse_conservative(grid, save_path, q, dt):
    with open(save_path, "r") as jsf:
        json_obj = json.load(jsf)

    metadata = json_obj[0][1]["metadata"]

    P = np.array(metadata["P0"])
    Pb = metadata["Pb"]
    Sb_dict = metadata["Sb_dict"]
    phi = np.array(metadata["Phi"])
    K = np.array(metadata["K"])
    T = np.array(metadata["T"])
    mu_g = metadata["mu_g"]
    mu_w = metadata["mu_w"]
    kr_model = metadata["kr_model"]
    wells = []
    for well in metadata["wells"]:
        if "Neumann" in well["control"].keys():
            wells.append(
                Well(
                    name=well["name"],
                    cell_group=well["cell_group"],
                    radius=well["radius"],
                    control={"Neumann": q},
                    s_inj=well["s_inj"],
                    schedule=well["schedule"],
                    mode=well["mode"],
                )
            )
        else:
            wells.append(
                Well(
                    name=well["name"],
                    cell_group=well["cell_group"],
                    radius=well["radius"],
                    control=well["control"],
                    s_inj=well["s_inj"],
                    schedule=well["schedule"],
                    mode=well["mode"],
                )
            )

    dt_init = dt
    del metadata
    del json_obj

    def loss(y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        mse = tf.reduce_mean(squared_difference, axis=-1)
        print(y_pred)
        """
        P_guess = implicit_pressure_solver_with_wells(grid=grid,
                                                      K=K,
                                                      T=T,
                                                      P=P,
                                                      S=y_pred,
                                                      Pb=Pb,
                                                      Sb_dict=Sb_dict,
                                                      mu_g=mu_g,
                                                      mu_w=mu_w,
                                                      wells=wells,
                                                      kr_model=kr_model)
        """
        S_i = np.zeros(y_pred.shape)
        print(S_i.shape, y_pred.shape)
        r = check_conservative(
            grid=grid,
            S_i=S_i,
            Pb=Pb,
            Sb_dict=Sb_dict,
            phi=phi,
            K=K,
            T=T,
            mu_g=mu_g,
            mu_w=mu_w,
            dt_init=dt_init,
            wells=wells,
            kr_model=kr_model,
            P_guess=P,
            S_guess=y_pred,
        )

        return mse + tf.reduce_mean(tf.square(r), axis=-1)

    return loss


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from ast import literal_eval
    from tensorflow.keras.optimizers import Adam
    from sklearn.model_selection import train_test_split

    from yads.predictors.Neural_Networks.create_mlp import create_256_mlp
    import yads.mesh as ym

    csv_path = "easy_points_predictors/first_approach/data/getting_sirious.csv"
    json_path = "easy_points_predictors/first_approach/data/getting_sirious.json"

    df = pd.read_csv(csv_path, converters={"S": literal_eval, "P": literal_eval})
    X = df[["q", "total_time"]]
    y = df[["S"]]

    map_shape = (y.shape[0], len(df["S"].loc[0]))

    (trainX, testX, trainY, testY) = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    trainAttrX, testAttrX = (
        trainX[["q", "total_time"]].to_numpy(),
        testX[["q", "total_time"]].to_numpy(),
    )
    trainY, testY = np.array(list(trainY["S"])), np.array(list(testY["S"]))

    model = create_256_mlp(2, 256)

    opt = Adam(learning_rate=1e-3, decay=1e-3 / 300)
    grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 256, 1)
    model.compile(
        loss=mse_conservative(
            grid=grid, save_path=json_path, q=trainAttrX[:, 0], dt=trainAttrX[:, 1]
        ),
        optimizer=opt,
        run_eagerly=True,
    )

    history = model.fit(
        trainAttrX, trainY, validation_data=(testAttrX, testY), epochs=300, batch_size=1
    )
