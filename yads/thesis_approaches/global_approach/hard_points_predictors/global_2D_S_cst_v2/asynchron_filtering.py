import os
import pandas as pd


def quality_check(path: str):
    if not os.path.isdir(path):
        print(f"given path does not exists or is not a directory (given path: {path})")
        return False
    sub_dirs = os.listdir(path)
    for d in sub_dirs:
        if d == "all_sim":
            # get filenames
            files_path = [
                os.path.relpath(x)
                for x in os.listdir(path + "/all_sim")
                if os.path.abspath(x).endswith(".csv")
            ]
            # check if length corresponds to the number of sim given in a filename
            # number of simulation is the first number after sim: all_sim_X
            parsing = files_path[0].split("_")
            num_sim = int(parsing[parsing.index("sim") + 1])
            if num_sim != len(files_path):
                print(
                    f"Simulation number and file number do not match ({num_sim} and {len(files_path)})"
                )
                return False
        elif d == "hard_sim":
            # no quality check yet
            pass
        elif d == "giga_hard_sim":
            # no quality check yet
            pass
    return True


def gather(path: str):
    sub_dirs = os.listdir(path)
    for d in sub_dirs:
        if d == "all_sim":
            # get filenames
            files_path = [
                os.path.relpath(x)
                for x in os.listdir(path + "/all_sim")
                if os.path.abspath(x).endswith(".csv")
            ]
            list_of_df = []
            for file in files_path:
                list_of_df.append(pd.read_csv(path + "/all_sim/" + file, sep="\t"))
            df = pd.concat(list_of_df, axis=0)
            df.to_csv(path + "/" + path + "_all_sim.csv", sep="\t", index=False)
        elif d == "hard_sim":
            # get filenames
            files_path = [
                os.path.relpath(x)
                for x in os.listdir(path + "/hard_sim")
                if os.path.abspath(x).endswith(".csv")
            ]
            list_of_df = []
            if files_path:
                for file in files_path:
                    list_of_df.append(pd.read_csv(path + "/hard_sim/" + file, sep="\t"))
                df = pd.concat(list_of_df, axis=0)
                df.to_csv(path + "/" + path + "_hard_sim.csv", sep="\t", index=False)
        elif d == "giga_hard_sim":
            # get filenames
            files_path = [
                os.path.relpath(x)
                for x in os.listdir(path + "/giga_hard_sim")
                if os.path.abspath(x).endswith(".csv")
            ]
            if files_path:
                list_of_df = []
                for file in files_path:
                    list_of_df.append(
                        pd.read_csv(path + "/giga_hard_sim/" + file, sep="\t")
                    )
                df = pd.concat(list_of_df, axis=0)
                df.to_csv(
                    path + "/" + path + "_giga_hard_sim.csv", sep="\t", index=False
                )
    return


def main(path):
    if quality_check(path):
        gather(path)


if __name__ == "__main__":
    data_path = "test"
    main(path=data_path)
    test_df = pd.read_csv(data_path + "/test_all_sim.csv", sep="\t")
    # print(test_df.describe())
    # print(test_df.columns)
    assert len(test_df.columns) == 10
