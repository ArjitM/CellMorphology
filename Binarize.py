import numpy as np
import skimage
from skimage import io
from skimage import morphology
from skimage import filters
from skimage import color
import math

global WHITE
WHITE = None  # should be set by main main script

global bin_WHITE
bin_WHITE = WHITE  # same as input rendition changing to uint8 requires new threshold parameters!!


class Compartment:
    def __init__(self, left, right, top, bottom, image_values):
        self.values = []
        for i in range(top, bottom):
            for j in range(left, right):
                self.values.append(float(image_values[i][j]))
        self.coordinates = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        self.noise_compartment = False
        self.hasBorder = True
        self.neighbors = None
        self.border_mean = 0

    def __str__(self):
        return self.coordinates

    def std(self):
        if self.values == []:
            print(self.coordinates)
            return 0
        return np.std(self.values, dtype=np.float64)

    def mean(self):
        return np.mean(self.values)

    def borderPercent(self, binary):
        border = 0
        for i in range(self.coordinates['top'], self.coordinates['bottom']):
            for j in range(self.coordinates['left'], self.coordinates['right']):
                if binary[i][j] == 0:
                    border += 1
        area = (self.coordinates['right'] - self.coordinates['left']) * (
                    self.coordinates['bottom'] - self.coordinates['top'])
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
            self.border_mean = np.mean(border)
            return np.mean(border)

        self.hasBorder = False
        # print(self.str())
        return None


class Compartmentalize:
    def __init__(self, image_values, size):
        self.compartments = []
        i_max, j_max, ki, kj = len(image_values), len(image_values[0]), 0, 0

        while ki < i_max:
            while kj < j_max:
                self.compartments.append(
                    Compartment(kj, min(kj + size, j_max), ki, min(ki + size, i_max), image_values))
                kj += size
            kj = 0
            ki += size
        self.size = size
        self.i_max = i_max
        self.j_max = j_max

    def getCompartment(self, i, j):
        index = (j // self.size) + (i // self.size) * (self.j_max // self.size)
        assert i >= 0 and j >= 0 and index < len(self.compartments)
        return self.compartments[index]

    def setNeighborCompartments(self, compartment):
        neighbors = []
        neighbor_points = [(compartment.coordinates['top'], compartment.coordinates['left'] - 1),
                           (compartment.coordinates['top'] - 1, compartment.coordinates['left'] - 1),
                           (compartment.coordinates['top'] - 1, compartment.coordinates['left']),
                           (compartment.coordinates['bottom'], compartment.coordinates['right'] + 1),
                           (compartment.coordinates['bottom'] + 1, compartment.coordinates['right'] + 1),
                           (compartment.coordinates['bottom'] + 1, compartment.coordinates['right'])]

        for np in neighbor_points:
            try:
                neighbors.append(self.getCompartment(np[0], np[1]))
            except AssertionError:
                pass
        compartment.neighbors = neighbors

    def setNoiseCompartments(self, binary, border_thresh):
        for c in self.compartments:
            k = c.borderPercent(binary)
            if k > border_thresh:
                c.set_noise_compartment(True)
            else:
                c.set_noise_compartment(False)
        self.reviseNoiseCompartments()

    def reviseNoiseCompartments(self):
        for c in self.compartments:
            self.setNeighborCompartments(c)
        for c in self.compartments:
            hits = 0
            for n in c.neighbors:
                if n.noise_compartment:
                    hits += 1
            if hits > (len(c.neighbors) // 2):
                c.set_noise_compartment(True)
            else:
                c.set_noise_compartment(False)

    def getAvgBorder(self, binary):
        not_Noise = filter(lambda c: not bool(c.noise_compartment), self.compartments)
        avgs = [c.avgBorder(binary) for c in not_Noise if c.hasBorder]
        return np.nanmean([a for a in avgs if a])

    def getAvgBorders(self, binary):
        not_Noise = filter(lambda c: not bool(c.noise_compartment), self.compartments)
        return [c.avgBorder(binary) for c in not_Noise if c.hasBorder]


class Noise:

    def __init__(self, image_values, regions=None, iterations=3, binary=False):
        if not binary:
            assert isinstance(regions,
                              Compartmentalize), "Compartments are required for noise editing unless binary image"
            self.regions = regions
        self.image_values = image_values
        self.iterations = iterations
        self.binary = binary
        self.detectNoise = self.detectNoise_BIN if binary else self.detectNoise_OG

    def detectNoise_OG(self, i, j, neighbors, largeNoise):
        cut = self.regions.getCompartment(i, j).std() / 3
        num = len(neighbors)
        hits = 0
        for (y, x) in neighbors:
            if abs(float(self.image_values[i][j]) - float(self.image_values[y][x])) > cut:
                hits += 1
        return [hits > (num * 0.5), 0]

    def detectNoise_BIN(self, i, j, neighbors, largeNoise):
        num = len(neighbors)
        hits = 0
        hit_vals = []
        for (y, x) in neighbors:
            if self.image_values[i][j] != self.image_values[y][x]:
                hits += 1
                hit_vals.append(1)
            else:
                hit_vals.append(0)
        if largeNoise:
            return [hits > (num * 0.5), 0]
        else:
            return [hits > (num * 0.5), bin_WHITE] if self.image_values[i][j] == 0 else [hits > (num * 0.5), 0]

    def smooth(self, largeNoise):
        for i in range(len(self.image_values)):
            for j in range(len(self.image_values)):
                change, value = self.detectNoise(i, j, list(getNeighborIndices(self.image_values, i, j)), largeNoise)
                if change:
                    self.image_values[i][j] = value

    def reduce(self, largeNoise=False):
        for _ in range(self.iterations):
            self.smooth(largeNoise)


def getNeighborIndices(image_values, i, j):
    for ki in range(-1, 2):
        for kj in range(-1, 2):
            if i + ki >= 0 and j + kj >= 0 and i + ki < len(image_values) and j + kj < len(image_values[0]) and not (
                    ki == 0 and kj == 0):
                yield ((i + ki, j + kj))


def getForwardNeighbors(image, previous, point):
    assert previous != point, "Previous point cannot be the same point"
    delta_i = point[0] - previous[0]
    delta_j = point[1] - previous[1]
    not_diagonal = abs(delta_i) ^ abs(delta_j)
    if abs(not_diagonal) >= 2:
        print("Previous ", previous, " | Point ", point, " | Diagonal ", not_diagonal)
        raise AssertionError("previous point must be neighbor, diagonal: ", not_diagonal)

    if not_diagonal:
        if delta_i:
            possible = [(point[0] + delta_i, point[1] + k) for k in [-1, 0, 1]]
        else:
            possible = [(point[0] + k, point[1] + delta_j) for k in [-1, 0, 1]]
    else:
        possible = [(point[0] + delta_i, point[1]), (point[0] + delta_i, point[1] + delta_j),
                    (point[0], point[1] + delta_j)]

    return [p for p in possible if p[0] < len(image) and p[1] < len(image[0])]


def basicEdge(pic_array, out_array, regions):
    for i in range(len(pic_array) - 1):
        for j in range(len(pic_array[0]) - 1):

            if out_array[i, j] == WHITE:
                continue
            left, right = float(pic_array[i][j]), float(pic_array[i][j + 1])
            bottom, diagonal = float(pic_array[i + 1][j]), float(pic_array[i + 1][j + 1])

            ''' May require revision '''
            cut = regions.getCompartment(i, j).std() / 3  # 3 basic, 5 sensitive
            # cut = np.std(out_array) // 3

            if abs(left - right) > cut:
                out_array[i][j] = 0
            elif abs(left - bottom) > cut:
                out_array[i][j] = 0
            elif abs(left - diagonal) > cut:
                out_array[i][j] = 0
            elif left == WHITE:
                out_array[i][j] = 0  # remove saturated pixels
            else:
                out_array[i][j] = WHITE  # bin_WHITE //using boolean representation to save memory
    return out_array


def enhanceEdges(pic_array, out_array, regions, nucleusMode=False):
    global borderMean
    borderMean = regions.getAvgBorder(out_array)
    tolerance = 0.085 * WHITE  # +/- 7 % of total image range
    bg = bin_WHITE if nucleusMode else 0
    fg = 0 if nucleusMode else bin_WHITE

    for i in range(len(out_array)):
        for j in range(len(out_array[0])):
            local_border_avg = regions.getCompartment(i, j).border_mean
            if local_border_avg - tolerance <= pic_array[i][j] and local_border_avg + tolerance >= pic_array[i][j]:
                out_array[i][j] = bg
            elif pic_array[i][j] > local_border_avg:
                out_array[i][j] = bg  # remove blood vessels
            elif pic_array[i][j] < local_border_avg and regions.getCompartment(i, j).noise_compartment == False:
                out_array[i][j] = fg  # foreground
            # if regions.getCompartment(i, j).noise_compartment == True:
            # out_array[i][j] = WHITE // 2


def removeBloodVessels(binary, pic_array):
    pass


def findBoundaryPoints(binary, segmented=False):
    boundary = []
    if not segmented:
        for i in range(len(binary)):
            for j in range(len(binary[0])):
                n_touching = len(
                    list(filter(lambda p: binary[p[0]][p[1]] == bin_WHITE, getNeighborIndices(binary, i, j))))
                # number of neighbors that are within a potential cell
                if binary[i][j] == bin_WHITE and n_touching != 8:
                    boundary.append((i, j))  # , n_touching > 5)) #last argument is cusp boolean

    else:
        return findSegmentedBoundary(binary)
    return boundary


def findSegmentedBoundary(segmented):
    segmentedBoundary = {}  # this is a dictionary with keys object numbers, values boundaries
    for i in range(len(segmented)):
        for j in range(len(segmented[0])):
            if segmented[i, j] != 0:

                if i == 0 or j == 0 or i == (len(segmented) - 1) or j == (len(segmented[0]) - 1):
                    try:
                        segmentedBoundary[segmented[i, j]].append((i, j))
                    except KeyError:
                        segmentedBoundary[segmented[i, j]] = [(i, j)]

                else:
                    for npoint in getNeighborIndices(segmented, i, j):
                        if segmented[npoint[0], npoint[1]] != segmented[i, j]:
                            try:
                                segmentedBoundary[segmented[i, j]].append((i, j))
                            except KeyError:
                                segmentedBoundary[segmented[i, j]] = [(i, j)]

    return segmentedBoundary


def internalBorderTest(pic_array, out_array, boundary):
    internalBorder = out_array[:]  # pseudo-deep copy
    boundaryIndices = [(b[0], b[1]) for b in boundary]  # no cusp boolean
    tolerance = 0.15 * WHITE  # lenient
    for i in range(len(pic_array)):
        for j in range(len(pic_array[0])):
            if pic_array[i][j] > borderMean - tolerance and (i, j) not in boundaryIndices:
                internalBorder[i][j] = 0
    return internalBorder


def growCellBoundaries(pic_array, out_array):
    # Use only with Soma stain, NOT nucleus; assumes dark foreground
    bg_vals = []
    for i in range(len(pic_array)):
        for j in range(len(pic_array[0])):
            if out_array[i, j] == 0:
                bg_vals.append(pic_array[i, j])
    bg_vals.sort()
    bg_cutOff = bg_vals[int(0.6 * len(bg_vals))]  # top 40% pixel values in bg
    fg_cutOff = bg_vals[0]
    for i in range(len(pic_array)):
        for j in range(len(pic_array[0])):
            if out_array[i, j] == 0 and pic_array[i, j] > bg_cutOff:
                out_array[i, j] = 0
            elif out_array[i, j] == WHITE and pic_array[i, j] >= fg_cutOff:
                out_array[i, j] = 0
            else:
                out_array[i, j] = pic_array[i, j]

    return out_array


def getLoadedPixels(pic_array, out_array):
    # Use only with Soma stain, NOT nucleus; assumes dark foreground
    out_rgb = skimage.color.gray2rgb(out_array)
    fg_vals = []
    for i in range(len(pic_array)):
        for j in range(len(pic_array[0])):
            if out_array[i, j] != 0:
                fg_vals.append(pic_array[i, j])
    fg_vals.sort()
    fg_cutOff = fg_vals[int(0.6 * len(fg_vals))]  # top 40% pixel values in fg

    for i in range(len(pic_array)):
        for j in range(len(pic_array[0])):
            if out_array[i, j] == WHITE and pic_array[i, j] >= fg_cutOff:
                out_rgb[i, j] = [pic_array[i, j], 0, 0]
            elif out_array[i, j] == WHITE:
                out_rgb[i, j] = [0, pic_array[i, j], 0]

    return out_rgb
