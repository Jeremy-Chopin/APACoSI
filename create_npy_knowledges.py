import numpy as np
import os


KNOWLEDGES_PATH_MIN = os.path.join("Knowledges", "2D", "face_EDT_signed_min_new.csv")
KNOWLEDGES_PATH_MAX = os.path.join("Knowledges", "2D", "face_EDT_signed_max_new.csv")

min_k = np.genfromtxt(KNOWLEDGES_PATH_MIN, delimiter=";")
max_k = np.genfromtxt(KNOWLEDGES_PATH_MAX, delimiter=";")

final = np.stack((min_k, max_k), axis=0)

np.save(os.path.join("Knowledges", "2D", "edt_min.npy"), final)

print("ok")