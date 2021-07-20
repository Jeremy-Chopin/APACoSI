import os
import numpy as np
import pandas as pd

percentages = ["100", "75", "50"]
classes = ["Tha.L", "Tha.R", "Cau.L", "Cau.R", "Put.L", "Put.R", "Pal.L", "Pal.R", "Hip.L", "Hip.R", "Amy.L","Amy.R", "Acc.L", "Acc.R", "Mean"]

dict = {}

for i in range(0, 15):
    dict[i] = []

data_path = ("sub_datasets_brain")

for percentage in percentages:

    results_path = os.path.join(data_path, percentage)

    mean_result_path = os.path.join(results_path, "global_mean_result.csv")
    std_result_path = os.path.join(results_path, "global_std_result.csv")

    mean_df = np.round(pd.read_csv(mean_result_path, sep=";"),2)
    std_df = np.round(pd.read_csv(std_result_path, sep=";"),2)

    labels = mean_df[mean_df.columns[0]].values

    cnn_dice = np.round(mean_df["dice.1"].values, 2)
    cnn_std_dice = np.round(std_df["dice.1"].values,2)

    cnn_hd = np.round(mean_df["hausdorff.1"].values, 2)
    cnn_std_hd = np.round(std_df["hausdorff.1"].values,2)

    refi_dice = np.round(mean_df["dice.4"].values, 2)
    refi_std_dice = np.round(std_df["dice.4"].values, 2)

    refi_hd = np.round(mean_df["hausdorff.4"].values, 2)
    refi_std_hd = np.round(std_df["hausdorff.4"].values, 2)

    for i in range(0, len(labels)):
        if len(dict[i]) == 0:
            dict[i].append(labels[i])
        dict[i].append(cnn_dice[i])
        dict[i].append(cnn_std_dice[i])

        dict[i].append(cnn_hd[i])
        dict[i].append(cnn_std_hd[i])

        dict[i].append(refi_dice[i])
        dict[i].append(refi_std_dice[i])

        dict[i].append(refi_hd[i])
        dict[i].append(refi_std_hd[i])

string = ""
for i in range(0, 15):
    string += classes[i]
    for j in range(1, len(dict[i])):
        if j % 2 == 0:
            string += "\pm" + str(dict[i][j]) + "$" 
        else:
            string += " & $" + str(dict[i][j])
    string+= "\\\\" + "\n"

tab_file_path = os.path.join(data_path, "tab_file.txt")
textfile = open(tab_file_path, "w")
a = textfile.write(string)
textfile.close()


"""df_mean = pd.DataFrame(mean, index = index, columns = columns)
print(df_mean)
df_mean.to_csv(mean_result_path, sep=";")

df_std = pd.DataFrame(std, index = index, columns = columns)
print(df_std)
df_std.to_csv(std_result_path, sep=";")"""