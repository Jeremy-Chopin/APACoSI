import os
import numpy as np
from skimage.measure import regionprops, label
import cc3d
import time
from matplotlib import pyplot as plt

"""t1 = 0
t2 = 0

for i in range(0, 100):
    array = np.random.rand(256,128, 256)

    array = np.where(array >= 0.75, 1, 0)

    temp = time.time()
    lbl = label(array, connectivity=2)
    t1 += time.time() - temp

    temp = time.time()
    lbl2 = cc3d.connected_components(array)
    t2 += time.time() - temp

    if lbl.all() != lbl2.all():
        break

print("T1 : ", t1/100)
print("T2 : ", t2/100)"""

path_to_irm = os.path.join("sub_datasets_brain", "100", "0", "segmentation", "0pr.npy")

indices = []
counts_global = []

image = np.load(path_to_irm)

#image = image[1:15,:,:,:]

#u = np.unique(np.argmax(image, axis=0))

ar_sort = np.sort(image, axis=0)

a_max1 = ar_sort[14,:,:,:]
a_max2 = ar_sort[13,:,:,:]

diff = a_max1 - a_max2

for i in range(1, 99):

    indice = i/100

    arrf = np.where(diff < indice, 1, 0)

    counts  = np.sum(arrf) / image.size

    counts_global.append(counts)
    indices.append(indice)


plt.plot(np.array(indices), np.array(counts_global))
plt.show()