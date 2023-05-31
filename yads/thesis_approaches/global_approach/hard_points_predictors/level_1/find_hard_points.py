import copy

from yads.thesis_approaches.data_generation import (
    raw_solss_1_iter,
    raw_solss_n_iter,
    raw_solss_1_hard_iter,
)
from yads.predictors.hard_points_predictors.level_1.utils import (
    dict_to_args,
    dict_to_args_hard,
    data_dict_to_combi,
    data_dict_to_combi_hard,
    generate_wells_from_q,
    hard_data_to_df,
    giga_hard_data_to_df,
    data_to_df,
)


def find_hard_points(var_dict, data_dict, savepath: str):
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

    ############### GLOBAL SIM PART ##########################

    def launch_sim(data_dict2, combination, mapping2, combination_ser, step):
        print(f"launching simulation number {step + 1}/{len(combinations)}")
        data_dict2 = data_dict_to_combi(data_dict2, combination, mapping2)
        args = dict_to_args(data_dict2)

        sim_state = raw_solss_1_iter(*args)
        print(f"simulation number {step + 1}/{len(combinations)} finished")
        return [combination_ser, sim_state]

    print("LAUNCHING SIMULATION ON ALL POINTS")
    simulation_dataset = Parallel(n_jobs=-1)(
        delayed(launch_sim)(
            data_dict, combinations[i], mapping, combinations_serializable[i], i
        )
        for i in range(len(combinations))
    )
    simulation_dataset = list(filter(lambda ele: ele is not None, simulation_dataset))
    print("END OF SIMULATION ON ALL POINTS")
    ############################# HARD SIM PART #######################

    hard_var_dict = {"dt_init": [], "q": []}
    hard_combinations = []
    hard_combinations_ser = []
    for i, (comb, state) in enumerate(simulation_dataset):
        for tot_t in state["data"].keys():
            nb_newton = state["data"][tot_t]["nb_newton"]
            if nb_newton > state["metadata"]["max_newton_iter"]:
                hard_combinations.append(combinations[i])
                hard_combinations_ser.append(combinations_serializable[i])
                hard_var_dict["dt_init"].append(var_dict["dt_init"][i])
                hard_var_dict["q"].append(var_dict["q"][i])

    hard_data_dict = copy.deepcopy(data_dict)
    # launching sim on points that have newton > max iter (10) with max iter = 100 this time
    hard_data_dict["max_newton_iter"] = 100

    def launch_hard_sim(data_dict2, combination, mapping2, combination_ser, step):
        print(f"launching simulation number {step + 1}/{len(hard_combinations)}")
        data_dict2 = data_dict_to_combi(data_dict2, combination, mapping2)
        args = dict_to_args(data_dict2)

        sim_state = raw_solss_1_hard_iter(*args)
        print(f"simulation number {step + 1}/{len(hard_combinations)} finished")
        return [combination_ser, sim_state]

    print("LAUNCHING SIMULATION ON HARD POINTS")
    hard_simulation_dataset = []
    if len(hard_combinations) > 0:
        hard_simulation_dataset = Parallel(n_jobs=-1)(
            delayed(launch_hard_sim)(
                hard_data_dict,
                hard_combinations[i],
                mapping,
                hard_combinations_ser[i],
                i,
            )
            for i in range(len(hard_combinations))
        )
    hard_simulation_dataset = list(
        filter(lambda ele: ele is not None, hard_simulation_dataset)
    )
    print("END OF SIMULATION ON HARD POINTS")

    ##### GIGA HARD SIM PART ###################
    # Newton > 100
    giga_hard_var_dict = {"dt_init": [], "q": []}
    giga_hard_combinations = []
    giga_hard_combinations_ser = []
    ens = []
    for i, (comb, state) in enumerate(hard_simulation_dataset):
        for tot_t in state["data"].keys():
            nb_newton = state["data"][tot_t]["nb_newton"]
            if nb_newton == 100:
                # not optimal but secure: n = 10, let's try first n = 3
                # n = nb_newton // 10
                n = 3
                ens.append(n)
                giga_hard_combinations.append(hard_combinations[i])
                giga_hard_combinations_ser.append(hard_combinations_ser[i])
                giga_hard_var_dict["dt_init"].append(hard_var_dict["dt_init"][i])
                giga_hard_var_dict["q"].append(hard_var_dict["q"][i])

    giga_hard_data_dict = copy.deepcopy(hard_data_dict)
    giga_hard_data_dict["max_newton_iter"] = 10

    def launch_giga_hard_sim(
        data_dict2, combination, mapping2, combination_ser, n, step
    ):
        print(f"launching simulation number {step + 1}/{len(giga_hard_combinations)}")
        data_dict2 = data_dict_to_combi_hard(data_dict2, combination, mapping2, n)
        args = dict_to_args_hard(data_dict2)

        sim_state = raw_solss_n_iter(*args)
        print(f"simulation number {step + 1}/{len(giga_hard_combinations)} finished")
        return [combination_ser, sim_state]

    print("LAUNCHING SIMULATION ON GIGA HARD POINTS")
    giga_hard_simulation_dataset = []
    if len(ens) > 0:
        giga_hard_simulation_dataset = Parallel(n_jobs=-1)(
            delayed(launch_giga_hard_sim)(
                giga_hard_data_dict,
                giga_hard_combinations[i],
                mapping,
                giga_hard_combinations_ser[i],
                ens[i],
                i,
            )
            for i in range(len(giga_hard_combinations))
        )
    giga_hard_simulation_dataset = list(
        filter(lambda ele: ele is not None, giga_hard_simulation_dataset)
    )
    print("END OF SIMULATION ON GIGA HARD POINTS")
    #### UPDATE HARD ####
    for i, (comb, state) in enumerate(hard_simulation_dataset):
        for tot_t in state["data"].keys():
            nb_newton = state["data"][tot_t]["nb_newton"]
            if nb_newton == 100:
                hard_simulation_dataset[i] = -1
    hard_simulation_dataset = list(
        filter(lambda ele: ele != -1, hard_simulation_dataset)
    )

    ##### UPDATE ALL POINTS FOR CLASSIFICATION #######
    # print(hard_combinations)
    for i, (comb, state) in enumerate(simulation_dataset):
        case = "undetermined"
        for tot_t in state["data"].keys():
            nb_newton = state["data"][tot_t]["nb_newton"]
            if nb_newton <= state["metadata"]["max_newton_iter"]:
                case = "gentle"
            else:
                if comb in giga_hard_combinations_ser:
                    case = "giga_hard"
                elif comb in hard_combinations_ser:
                    case = "hard"
        simulation_dataset[i].append(case)
    # save features of interest in csv
    df_sim = data_to_df(simulation_dataset)
    df_sim.to_csv(savepath + ".csv")
    df_hard = hard_data_to_df(hard_simulation_dataset)
    df_hard.to_csv(savepath + "_hard.csv")
    df_giga_hard = giga_hard_data_to_df(giga_hard_simulation_dataset)
    df_giga_hard.to_csv(savepath + "_giga_hard.csv")
    return hard_simulation_dataset, giga_hard_simulation_dataset


def find_points(var_dict, data_dict, savepath: str):
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

    ############### GLOBAL SIM PART ##########################

    def launch_sim(data_dict2, combination, mapping2, combination_ser, step):
        print(f"launching simulation number {step + 1}/{len(combinations)}")
        data_dict2 = data_dict_to_combi(data_dict2, combination, mapping2)
        args = dict_to_args(data_dict2)

        sim_state = raw_solss_1_iter(*args)
        print(f"simulation number {step + 1}/{len(combinations)} finished")
        return [combination_ser, sim_state]

    print("LAUNCHING SIMULATION ON ALL POINTS")
    simulation_dataset = Parallel(n_jobs=-1)(
        delayed(launch_sim)(
            data_dict, combinations[i], mapping, combinations_serializable[i], i
        )
        for i in range(len(combinations))
    )
    simulation_dataset = list(filter(lambda ele: ele is not None, simulation_dataset))
    print("END OF SIMULATION ON ALL POINTS")
    ############################# HARD SIM PART #######################

    # save features of interest in csv
    df_sim = data_to_df(simulation_dataset)
    df_sim.to_csv(savepath + ".csv")
    return
