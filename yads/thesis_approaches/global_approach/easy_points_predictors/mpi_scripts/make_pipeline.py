from yads.predictors.easy_points_predictors.first_approach import (
    dict_to_args,
    data_dict_to_combi,
    generate_wells_from_q,
    json_to_df,
)
from yads.thesis_approaches.data_generation import raw_solss_1_iter
import json


def make_pipeline_mpi(var_dict, data_dict, savepath: str):
    mapping = {}
    var_list = []
    var_list_serializable = []
    for i, key in enumerate(var_dict.keys()):
        if key in data_dict.keys():
            mapping[key] = i
            var_list.append(var_dict[key])
            var_list_serializable.append(var_dict[key])
        # q -> well flux
        elif key in ["q"]:
            if key == "q":
                well_ref = None
                other_wells = []
                for well in data_dict["wells"]:
                    if well.name == "well co2":
                        well_ref = well
                    else:
                        other_wells.append(well)
                assert well_ref is not None
                mapping["wells"] = i
                non_ser_wells = generate_wells_from_q(
                    var_dict["q"], well_ref, other_wells
                )
                var_list.append(non_ser_wells)
                var_list_serializable.append(
                    [[well.well_to_dict() for well in combi] for combi in non_ser_wells]
                )
        else:
            print("key error")
            return

    combinations = [(var_list[0][i], var_list[1][i]) for i in range(len(var_dict["q"]))]

    combinations_serializable = [
        (var_list_serializable[0][i], var_list_serializable[1][i])
        for i in range(len(var_dict["q"]))
    ]
    ### Parallel computation ###
    from joblib import delayed, Parallel

    def launch_sim(data_dict2, combination, mapping2, combination_ser, rank):
        data_dict2 = data_dict_to_combi(data_dict2, combination, mapping2)
        args = dict_to_args(data_dict2)
        sim_state = raw_solss_1_iter(*args)
        return [combination_ser, sim_state]

    simulation_dataset = Parallel(n_jobs=-1)(
        delayed(launch_sim)(
            data_dict, combinations[i], mapping, combinations_serializable[i], i
        )
        for i in range(len(combinations))
    )

    # save everything to json
    with open(savepath + ".json", "w") as fp:
        json.dump(simulation_dataset, fp)

    # save features of interest in csv
    df = json_to_df(simulation_dataset)
    df.to_csv(savepath + ".csv")
    return


if __name__ == "__main__":
    pass
