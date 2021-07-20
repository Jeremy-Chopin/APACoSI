import os
import numpy as np
import pandas as pd

percentages = [100,75,50]

for percentage in percentages:

    percentage_path = os.path.join("sub_datasets_brain", str(percentage))

    mean_result_path = os.path.join(percentage_path, "global_mean_edt_cc.csv")
    std_result_path = os.path.join(percentage_path, "global_std_edt_cc.csv")

    nb_classes = 14
    arrays = []
    s = 0

    iterations = os.listdir(percentage_path)

    for iteration in iterations:

        iteration_path = os.path.join(percentage_path, iteration)

        if os.path.isdir(os.path.join(iteration_path, "results", "edt")):
            patients = os.listdir(os.path.join(iteration_path, "results", "edt"))
        
            for patient in patients:
        
                # On lit le fichier
                df_path = os.path.join(iteration_path, "results", "edt", patient, "result_CC.csv")
                if os.path.isfile(df_path):
                    df_result = pd.read_csv(df_path)
                    
                    # on recupère les index et colonnes
                    columns = df_result.columns

                    index = df_result[columns[0]].values[0:nb_classes]
                    df_result.drop(columns= columns[0])

                    columns = columns.values[1:]

                    # on recupère les données
                    array = df_result.to_numpy()[:,1:].astype(np.float32)

                    array = array[0:nb_classes,:]
                    
                    print(array)

                    arrays.append(array)

    if len(arrays) > 0:

        arrays.pop(0)

        all_array = np.stack(arrays, axis = 0)

        mean = np.mean(all_array, axis = 0)
        std = np.std(all_array, axis = 0)

        mean_avg = np.mean(mean, axis=0)
        mean_all = np.vstack((mean, mean_avg))

        std_avg = np.mean(std, axis=0)
        std_all = np.vstack((std, std_avg))

        index = np.hstack((index, np.array(["Mean"])))

        df_mean = pd.DataFrame(mean_all, index = index, columns = columns)
        print(df_mean)
        df_mean.to_csv(mean_result_path, sep=";")

        df_std = pd.DataFrame(std_all, index = index, columns = columns)
        print(df_std)
        df_std.to_csv(std_result_path, sep=";")