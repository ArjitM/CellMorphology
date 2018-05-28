from multiprocessing.dummy import Pool
import numpy as np 
#from __future__ import division

import skimage
from skimage import io
from skimage import external
from skimage import morphology
import math



class Compartment:
	def __init__(self, left, right, top, bottom, image_values):
		self.values = []
		for i in range(top, bottom):
			for j in range(left, right):
				self.values.append(float(image_values[i][j]))
		self.coordinates = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
		self.noise_compartment = False
		self.hasBorder = True

	def __str__(self):
		return self.coordinates

	def std(self):
		if self.values == []:
			print(self.coordinates)
			return 5
		return np.std(self.values, dtype=np.float64)

	def mean(self):
		return np.mean(self.values)

	def borderPercent(self, binary):
		border = 0
		for i in range(self.coordinates['top'], self.coordinates['bottom']):
			for j in range(self.coordinates['left'], self.coordinates['right']):
				if binary[i][j] == 0:
					border += 1
		area = (self.coordinates['right'] - self.coordinates['left']) * (self.coordinates['bottom'] - self.coordinates['top'])
		return border / area


	def set_noise_compartment(self, value):
		self.noise_compartment = bool(value)

	def avgBorder(self, binary):
		k, border = 0, []
		for i in range(self.coordinates['top'], self.coordinates['bottom']):
			for j in range(self.coordinates['left'], self.coordinates['right']):
				if binary[i][j] == 0:
					border.append(self.values[k])
				k += 1
		if border != []:
			return np.mean(border)

		self.hasBorder = False
		print(self.str())
		return None


class Compartmentalize:
	def __init__(self, image_values, size):
		self.compartments = []
		i_max, j_max, ki, kj = len(image_values), len(image_values[0]), 0, 0

		while ki < i_max:
			while kj < j_max:

				self.compartments.append(Compartment(kj, min(kj+size, j_max), ki, min(ki+size, i_max), image_values))
				kj += size
			kj = 0
			ki+= size
		self.size = size
		self.i_max = i_max
		self.j_max = j_max

	def getCompartment(self, i, j):
		return self.compartments[(j // self.size) + (i // self.size) * (self.j_max // self.size)]

	def setNoiseCompartments(self, binary, border_thresh):
		for c in self.compartments:
			k = c.borderPercent(binary)
			if k > border_thresh:
				c.set_noise_compartment(True)
			else:
				c.set_noise_compartment(False)

	def getAvgBorder(self, binary):
		not_Noise = filter(lambda c: not bool(c.noise_compartment), self.compartments)		
		return np.mean([c.avgBorder(binary) for c in not_Noise if c.hasBorder])


class Noise:

	def __init__(self, image_values, regions=None, iterations=3, binary=False):
		if not binary:
			assert isinstance(regions, Compartmentalize), "Compartments are required for noise editing unless binary image"
			self.regions = regions
		self.image_values = image_values
		self.iterations = iterations
		self.binary = binary
		self.detectNoise = self.detectNoise_BIN if binary else self.detectNoise_OG

	def detectNoise_OG(self, i, j, neighbors):
		cut = regions.getCompartment(i, j).std() / 3
		num = len(neighbors)
		hits = 0
		for (y, x) in neighbors:
			if abs(float(self.image_values[i][j]) - float(self.image_values[y][x])) > cut:
				hits += 1
		return [hits > (num * 0.5), 0]

	def detectNoise_BIN(self, i, j, neighbors):
		num = len(neighbors)
		hits = 0
		hit_vals = []
		for (y, x) in neighbors:
			if self.image_values[i][j] != self.image_values[y][x]:
				hits += 1
				hit_vals.append(1)
			else:
				hit_vals.append(0)
		return [hits > (num * 0.5), WHITE] if self.image_values[i][j] == 0 else [hits > (num * 0.5), 0]

	def smooth(self):
		for i in range(len(self.image_values)):
			for j in range(len(self.image_values)):
				change, value = self.detectNoise(i, j, getNeighborIndices(self.image_values, i, j))
				if change:
					self.image_values[i][j] = value

	def reduce(self):
		for _ in range(self.iterations):
			self.smooth()


def getNeighborIndices(image_values, i, j):
	neighbors = []
	for ki in range(-1, 2):
		for kj in range(-1, 2):
			if i + ki >= 0 and j + kj >= 0 and i + ki < len(image_values) and j + kj < len(image_values[0]) and not (ki==0 and kj==0):
				neighbors.append((i+ki, j+kj))
	return neighbors



def basicEdge(pic_array, out_array, regions):
	for i in range(len(pic_array) - 1):
		for j in range(len(pic_array[0]) - 1):

			left, right = float(pic_array[i][j]), float(pic_array[i][j+1])
			bottom, diagonal = float(pic_array[i+1][j]), float(pic_array[i+1][j+1])

			''' May require revision '''
			# Make borders thinner! if cut change within 5 pixels ?
			# find average color of borders and find missing border pixels ... closing may be needed thereafter
			cut = regions.getCompartment(i, j).std() / 3  # 3 basic, 5 sensitive

			if abs(left - right) > cut:
				out_array[i][j] = 0
			elif abs(left - bottom) > cut:
				out_array[i][j] = 0
			elif abs(left - diagonal) > cut:
				out_array[i][j] = 0
			else:
				out_array[i][j] = WHITE

def enhanceEdges(pic_array, out_array, regions):
	global borderMean
	borderMean = regions.getAvgBorder(out_array)
	tolerance = 0.07 * WHITE # +/- 10 % of total image range
	for i in range(len(out_array)):
		for j in range(len(out_array[0])):
			if borderMean - tolerance <= pic_array[i][j] and borderMean + tolerance >= pic_array[i][j]:
				out_array[i][j] = 0
			elif pic_array[i][j] > borderMean:
				out_array[i][j] = 0 #blackens blood vessels
			elif pic_array[i][j] < borderMean and regions.getCompartment(i, j).noise_compartment == False:
				out_array[i][j] = WHITE
			#if regions.getCompartment(i, j).noise_compartment == True:
			#	out_array[i][j] = WHITE // 2

def internalBorder(pic_array, out_array, regions):
	pass


class Cluster:

	clusters = []

	def __init__(self, binary, pivot):

		for c in Cluster.clusters:
			if c.contains(pivot):
				raise ValueError('pivot already in cluster')
		assert type(pivot) == tuple and len(pivot) == 2, 'represent points as (i, j)'

		self.points = []
		self.add_Point(pivot)
		Cluster.clusters.append(self)
		self.binary = binary
		self.boundary = []

	def add_Point(self, point):
		self.points.append(point)

	def propagate(self, start):
		#print("in propagate")
		if not self.contains(start):
			self.points.append(start)
		neighbors = list(filter(lambda p: self.binary[p[0]][p[1]] == WHITE, getNeighborIndices(self.binary, start[0], start[1])))
		if len(neighbors) != 8:
			self.boundary.append(start)
		for n in filter(lambda p: not (self.contains(p)), neighbors):
			self.propagate(n)

	def contains(self, point):
		return point in self.points



		
def makeClusters(binary, regions):
	#using regions speeds up search
	for c in regions.compartments:
		if c.noise_compartment:
			continue
		i, j = c.coordinates['top'], c.coordinates['left']
		cluster = None
		while i < c.coordinates['bottom']:
			while j < c.coordinates['right']:
				if binary[i][j] == WHITE:
					try:
						cluster = Cluster(binary, (i, j))
					except ValueError:
						pass
					else:
						cluster.propagate((i, j))
						cluster.points.sort()
					finally:
						if cluster is not None:
							row = list(filter(lambda p: p[0] == i, cluster.points))
							right = row[-1][1] #j coordinate of last point in the row
							j = right
				else:
					j += 1
			i += 3 #cluster must be at least 3 pixels high








def process_image(inFile):

	with skimage.external.tifffile.TiffFile(inFile) as pic:

		pic_array = pic.asarray();
		out_array = pic.asarray(); #copy dimensions
		global WHITE
		WHITE = skimage.dtype_limits(pic_array, True)[1]
		regions = Compartmentalize(pic_array, 32)

		basicEdge(pic_array, out_array, regions) # out_array is modified
		skimage.external.tifffile.imsave(inFile.replace('.tif', '_edgeBasic.tif'), out_array)

		regions.setNoiseCompartments(pic_array, 0.95)

		enhanceEdges(pic_array, out_array, regions)
		skimage.external.tifffile.imsave(inFile.replace('.tif', '_edgeEnhance.tif'), out_array)


		noise_handler = Noise(out_array, iterations=3, binary=True)
		noise_handler.reduce()

		skimage.external.tifffile.imsave(inFile.replace('.tif', '_Binary.tif'), out_array)

		print("***made binary")

		makeClusters(out_array, regions)
		print(Cluster.clusters, len(Cluster.clusters))



'''prefixes = ['/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p1f2_normal/eye1-',
'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p1f3_normal/eye1-',
'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p2f1_normal/eye1-',
'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p2f2_normal/eye1-',
'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p2f3_normal/eye1-',
'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye2p1f1_normal/eye2-',
'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye2p1f2_normal/eye2-',
'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye2p1f3_normal/eye2-']'''

prefix = '/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p1f1_normal/eye1-'

def parallel(prefix):
	x = 24
	while True:
		try:
			process_image(prefix + str(x).rjust(4, '0') + '.tif')
		except IOError:
			break
		else:
			print(prefix)
			x += 1

#with Pool(5) as p:
#	p.map(parallel, prefixes)
parallel(prefix)






