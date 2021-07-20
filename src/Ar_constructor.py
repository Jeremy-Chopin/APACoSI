import numpy as np
from src import utils

class Ar_constructor(object):
	def __init__(self):
		self.constructor=Functor()
	
	def construct_Ar(self, specifier, labelled_image, regions, nodes):
		return self.constructor(specifier, labelled_image, regions, nodes)

class Functor(object):
			
	def __call__(self, specifier, labelled_image, regions, nodes) :

		if specifier == "centroid":
			return self. __Centroid(labelled_image, regions, nodes)
		elif specifier == "edt_min":
			return self. __EDT_min(labelled_image, regions, nodes)
		else :
			return self.__Error()

	def __Centroid(self, labelled_image, regions, nodes):
		Ar = np.zeros((len(nodes), len(nodes), len(labelled_image.shape)))

		for i in range(0, len(nodes)):
			ids1 = nodes[i].ids -1
			zc1, yc1, xc1 = regions[ids1].centroid
			for j in range(i+1, len(nodes)):
				ids2=nodes[j].ids -1
				zc2, yc2, xc2 = regions[ids2].centroid
				
				vector = np.asarray([xc2 - xc1, yc2 - yc1, zc2 - zc1])

				for dim in range(0, len(labelled_image.shape)):
					Ar[i][j][dim] = vector[dim]
					Ar[j][i][dim] = -vector[dim]

		Ar = np.transpose(Ar, (2,0,1))
		return Ar

	def __EDT_min(self, labelled_image, regions, nodes):
		
		Ar = np.zeros((2, len(nodes), len(nodes)))

		for i in range(0, len(nodes)):
			ids1 = nodes[i].ids -1
			mask = np.where(labelled_image == regions[ids1].label, 1, 0)
			dist = utils.signed_transform(mask)
			
			for j in range(0, len(nodes)):
				if i !=j:
					ids2 = nodes[j].ids -1
					mask2 = np.where(labelled_image == regions[ids2].label, 1, 0)
					res = mask2 * dist

					Ar[0][i][j] = np.min(res[np.nonzero(res)])
					Ar[1][i][j] = np.max(res[np.nonzero(res)])

		#Ar = np.transpose(Ar, (2,0,1))

		return Ar
		
	def __Error(self):
		print ("Data is not correct")