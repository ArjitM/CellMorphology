import math
import numpy as np
import skimage
from skimage import io
from skimage import external
from skimage import morphology
from skimage import filters
from skimage import color
from Binarize import *
import scipy.ndimage
from scipy.signal import argrelextrema
from itertools import chain
import copy
import functools

global WHITE
WHITE = None


class Cusp:

    def __init__(self, point, left_deriv, right_deriv, angle, angle_trend):
        self.point = point
        self.left_deriv = left_deriv
        self.right_deriv = right_deriv
        self.angle = abs(angle)
        self.angle_trend = angle_trend

    #def angle(self):
    #    return abs( math.atan(self.left_deriv) - math.atan(self.right_deriv) ) #in radians!!

class Edge:

    def __init__(self, edge):
        self.start = edge.pop(0)
        self.end = edge.pop()
        self.internalEdge = edge

def erase(pivot, binary):
    pivot.sort()
    pivot2D = [ list(filter(lambda p: p[0] == y, pivot)) for y in range(pivot[0][0], pivot[-1][0] + 1)]
    for row in pivot2D:
        if len(row) == 1:       
            for i,j in row: 
                binary[i,j] = WHITE
        else:
            start = row[0]
            end = row[-1]
            for k in range(start[1], end[1] + 1):
                binary[start[0],k] = WHITE


def createPivots(pivots, binary):
    pruned = []
    for pivot in pivots:
        #if len(pivot) < 12:
        pruned.append((int(np.mean([p[0] for p in pivot])), int(np.mean([p[1] for p in pivot]))))
        erase(pivot, binary)
    return pruned

def showPivots(binary, clusters):
    visualizedPivots = copy.deepcopy(binary) #deep copy is shallow!!!
    # visualizedPivots = []
    # for i in range(len(binary)):
    #     visualizedPivots.append([])
    #     for j in range(len(binary[0])):
    #         visualizedPivots[-1].append(binary[i][j])

    for c in clusters:
        for p in c.pivots:
            visualizedPivots[p[0]][p[1]] = 0
    return np.array(visualizedPivots)

class Cluster:

    clusters = []

    def __init__(self, binary, boundary, stack_slice, pivots=None, internalEdges=None): #, segmented=None, pic=None):

        self.boundary = list(set(boundary)) #DO NOT SORT THIS! Order is important
        #self.boundary2D = self.getBoundary2D()
        self.binary = binary
        self.cells = []
        self.cusps = []
        self.center = (int(np.mean([p[0] for p in self.boundary])), int(np.mean([p[1] for p in self.boundary])))
        self.pivots = createPivots(pivots, binary) if pivots is not None else None
        if (not self.pivots) or len(self.pivots) < 6:
            self.constriction_points = []
            self.internalEdges = internalEdges
            self.stack_slice = stack_slice
            #self.object_number = None
            self.internalEdges = [] if (internalEdges is None) else internalEdges #using [] as default argument is problematic; old is appended to default
            if not isinstance(self, Cell):
                Cluster.clusters.append(self)

    def getTrueCusps(self, segmentLen=8, arc=None):
        #arc must be None if using gradient-only declumping
        ##arc defined as important regions as per kmeans-results
        cusps = []

        if arc is None:
            arc = self.boundary
            assert len(self.boundary) > segmentLen * 3, 'boundary is too short. consider killing cluster'

        if len(self.boundary) < segmentLen * 2:
            return None

        for point in arc: 
            k = self.boundary.index(point)

            angles = []
            notCusp = False

            for segmentPoint in range((segmentLen // 4) * 3, 1 + segmentLen):
                
                try:
                    before = self.boundary[k - segmentPoint]
                except IndexError:
                    print(k-segmentPoint)
                    #after = self.boundary[k + segmentPoint - len(self.boundary)]
                after = self.boundary[(k + segmentPoint) % len(self.boundary)]

                midpt = (math.floor(np.mean([p[0] for p in [before, after]])), math.floor(np.mean([p[1] for p in [before, after]])))
                if self.binary[midpt[0]][midpt[1]] != 0:
                    #notCusp = True
                    continue 

                ldy = - (point[0] - before[0]); ldx = (point[1] - before[1])
                if ldx == 0:
                    left_deriv = math.inf if ldy > 0 else (- math.inf) 
                else:
                    left_deriv = ldy / ldx

                rdy = - (after[0] - point[0]); rdx = (after[1] - point[1])
                if rdx == 0:
                    right_deriv = math.inf if rdy > 0 else (- math.inf) 
                else:
                    right_deriv = rdy / rdx
                angles.append(abs( math.atan(left_deriv) - math.atan(right_deriv) ))

            if notCusp or angles == []:
                continue
            angle = np.nanmean(angles)
            if angle < 0.6 * math.pi:
                cusps.append(Cusp(point, left_deriv, right_deriv, angle, angles))

        if arc is None: #is not being used with kmeans algorithm, i.e. declumping is gradient ONLY
            self.cusps = cusps

        return cusps

    def getCuspsKMeans(self, kmean_labels, segmentLen=8):
        if len(self.boundary) < segmentLen * 3:
            return None

        assert len(kmean_labels) == len(self.boundary), 'labels and boundary points must be 1:1'
        k = 0
        group_change_points = []
        while k < len(self.boundary):
            if kmean_labels[k] != kmean_labels[k-1]:
                group_change_points.append(k)
            k +=1

        cleave_points = []
        for grp in group_change_points:
            group_change_region = [ self.boundary[i % len(self.boundary)] for i in range(grp-5, grp+6) ]
            cusps = self.getTrueCusps(segmentLen=segmentLen, arc=group_change_region)
            if cusps is None or len(cusps) == 0:
                continue
            cleave_points.append(min(cusps, key=lambda c: c.angle))
        return cleave_points

    def pruneCusps(self):
        # cusps = [c.point for c in self.cusps]
        arcs = []
        while self.cusps:
            seq = []
            previous = self.cusps[0]
            k = 0
            #arc is a sequence of contiguous (max distance 1 pixel) cusp-points
            while k < len(self.cusps) and max(abs(self.cusps[k].point[0] - previous.point[0]), abs(self.cusps[k].point[1] - previous.point[1])) <= 1:
                seq.append(self.cusps[k])
                previous = self.cusps[k]
                k += 1
            arcs.append(seq)
            del self.cusps[:k]

        self.arcs = [arc for arc in arcs if len(arc) >= 1] #removing arcs with len < 3 DOES NOT work!
        return self.arcs

    def showCusps(self):
        #for c in self.constriction_points:
        # for arc in self.arcs:
        #     for c in arc:
        #         self.binary[c.point[0]][c.point[1]] = WHITE // 2
        for c in self.constriction_points:
            for n in getNeighborIndices(self.binary, c.point[0], c.point[1]):
                self.binary[n[0]][n[1]] = WHITE // 2

    def splitBentCells(self):
        pass

    def transformToCell(self):
        nc = Cluster.makeCell(self.stack_slice, self.binary, self.boundary, [])
        self.stack_slice.addCell(nc)
        return nc

    @staticmethod
    def makeCell(stack_slice, binary, cell_interior, internalEdges):
        cell = Cell(stack_slice, binary, cell_interior, internalEdges)
        return cell

    def growPivots(self):
        from math import sqrt
        distance_sequences = []
        minima_sequences = []
        for p in self.pivots:
            distances = []
            for b in self.boundary:
                distances.append(sqrt( (p[0] - b[0])**2 + (p[1] - b[1])**2) )
            #print("-------: ", distances)
            distance_sequences.append(distances)
            distances = np.array(distances)
            minima = argrelextrema(distances, np.less, mode='wrap')[0].tolist() #returns indices not actual minima
            #Ensure at least 40 pixel separation between minima indices
            too_keep = []
            l = len(minima)
            i=0
            while i < l:
                if ((i+1) < l and minima[(i+1)] - minima[i] < 40) or ((i+1) > l and minima[(i+1) % l] + l - minima[i] < 40): #modulus allows wrapping
                    group = [i % l, (i+1) % l]
                    k = 2
                    while ((i+k) < l and minima[(i+k)] - minima[i] < 40) or ((i+k) > l and minima[(i+k) % l] + l - minima[i] < 40):
                        group.append((i+k) % l)
                        k += 1
                    best = min(group, key=lambda x: distances[minima[x]])
                    too_keep.append(best)
                    i = i+k
                else:
                    too_keep.append(minima[i])
                i+=1

            minima_sequences.append(too_keep)
            for m in minima: #too_keep:
                point = self.boundary[m]
                for n in getNeighborIndices(self.binary, point[0], point[1]):
                    self.binary[n[0]][n[1]] = WHITE // 2

            ##Identified minima should be in the vicinity of cusps identified by gradient
            ###Merge pivots in close vicinity ??
        #return list(chain.from_iterable(minima_sequences))

    def propagateInternalBoundaries(self, cleave_points=None):
      
        # if len(self.arcs) <= 1:
        #     return None

        self.constriction_points = []
        
        #cleave_points = [ min(arc, key=lambda c: c.angle) for arc in self.arcs]
        ## these are the points where internal boundaries start/stop. Find by looking for cusp region points with the least (most constricted) angle
        if cleave_points is None:
            cleave_points = [arc[len(arc) // 2] for arc in self.arcs]
        completed_pairs = []
        #boundaries that have already been made, avoid duplication

        for cp in cleave_points:

            self.constriction_points.append(cp)
            orientation = lambda p: abs(np.mean( [(math.atan(p.left_deriv) - math.atan(cp.left_deriv)), (math.atan(p.right_deriv) - math.atan(cp.right_deriv))] ))
            viable = filter(lambda p: abs(orientation(p)) > math.pi * 0.5, cleave_points)
            viable_no_duplicate = filter(lambda p: (p, cp) not in completed_pairs and (cp, p) not in completed_pairs, viable)
            try:
                pair = min(viable_no_duplicate, key = lambda p: (cp.point[0] - p.point[0])**2 + (cp.point[1] - p.point[1])**2 )
            except ValueError:
                continue
            #print("Pair", pair.point)
            completed_pairs.append((pair, cp))

            delta_i = cp.point[0] - pair.point[0]
            ki = -1 if delta_i > 0 else 1
            delta_i = abs(delta_i)

            delta_j = cp.point[1] - pair.point[1]
            kj = -1 if delta_j > 0 else 1
            delta_j = abs(delta_j)

            # to propagate boundaries from one constriction site to another, assume a simple diagonal line (m=1)
            ## and use periodic shifts to complete the boundary
            edge = []
            edge.append(cp.point)
            #print("Start: ", cp.point)

            if delta_i > delta_j:

                #print("delta i > j")

                #vert_shifts = delta_i - delta_j
                if delta_j != 0:
                    shift_period = delta_i // delta_j #vert_shifts #need a shift every length period of i 
                    shift = True

                else:
                    shift_period = 100
                    shift = False


                #print("horizontal shift period", shift_period)

                x, y = cp.point[1], cp.point[0]
                k = 0
                x_rem, y_rem = math.inf, math.inf
                while min(x_rem, y_rem) > 1:

                    #print("k", k)
                    k += 1
                    y += ki
                    
                    if shift and k % shift_period == 0:
                        x += kj #horizontal shift
                    edge.append((y, x))

                    if k > 500:
                        break
                    #print((y,x))

                    x_rem = abs(x - pair.point[1])
                    y_rem = abs(y - pair.point[0])

                assert x_rem <= 1 or y_rem <= 1, 'your code is trash'

                if x_rem > 1:
                    for rem in range(x, pair.point[1], kj):
                        edge.append((y, rem))
                        #print((y, rem))

                else:
                    for rem in range(y, pair.point[0], ki):
                        edge.append((rem, x))
            elif delta_j > delta_i:

                #print('delta j > i')
                #horiz_shifts = delta_j - delta_i

                if delta_i != 0:
                    shift = True
                    shift_period = delta_j // delta_i #need a shift every length period of j 

                else:
                    shift_period = 100
                    shift = False

                x, y = cp.point[1], cp.point[0]
                k = 0
                x_rem, y_rem = math.inf, math.inf
                while min(x_rem, y_rem) > 1:

                    #print('k', k)
                    k += 1
                    x += kj

                    if shift and k % shift_period == 0:
                        y += ki #vertical shift
                    edge.append((y, x))
                    
                    if k > 500:
                        break
                    #print((y,x))

                    x_rem = abs(x - pair.point[1])
                    y_rem = abs(y - pair.point[0])

                assert x_rem <= 1 or y_rem <= 1, 'your code is trash'

                if x_rem > 1:
                    for rem in range(x, pair.point[1], kj):
                        edge.append((y, rem))
                        #print((y, rem))

                else:
                    for rem in range(y, pair.point[0], ki):
                        edge.append((rem, x))
                        #print((rem, x))
            
            else:

                #print("i = j")
                x, y = cp.point[1], cp.point[0]
                k = 0
                while min(abs(x - pair.point[1]), abs(y - pair.point[0])) > 1:
                    #print('k', k)
                    k += 1
                    y += ki
                    x += kj
                    edge.append((y, x))

                    if k>500:
                        break
                    #print((y,x))
            edge.append(pair.point)

            for p in edge:
                self.binary[p[0]][p[1]] = 0 

            #print("New edge done")

            self.internalEdges.append(Edge(edge))

        #for ep in edges:
        #    self.binary[ep[0]][ep[1]] = 0

    def splitByEdges(self):
        if self.internalEdges == []:
            newCell = Cluster.makeCell(self.stack_slice, self.binary, self.boundary, [])
            if len(newCell.boundary) > 10:
                self.stack_slice.addCell(newCell)
        else:
            if self.internalEdges:
                divider = self.internalEdges.pop()
                try:
                    start_index = self.boundary.index(divider.start)
                except ValueError: #intersecting internal edges
                    return
                
                try:
                    end_index = self.boundary.index(divider.end)
                except ValueError:
                    return

                if start_index > end_index:
                    cell_1_bound = list(reversed(divider.internalEdge)) + self.boundary[start_index : end_index : -1] + [divider.end]
                    cell_2_bound = list(reversed(divider.internalEdge)) + self.boundary[start_index:] + self.boundary[:end_index+1]

                else:
                    cell_1_bound = list(reversed(divider.internalEdge)) + self.boundary[start_index : end_index : 1] + [divider.end]
                    cell_2_bound = list(reversed(divider.internalEdge)) + self.boundary[end_index::1] + self.boundary[:start_index+1:1]

                cell_1 = Cluster.makeCell(self.stack_slice, self.binary, cell_1_bound, self.internalEdges)
                cell_2 = Cluster.makeCell(self.stack_slice, self.binary, cell_2_bound, self.internalEdges)

                cell_1.splitByEdges()
                cell_2.splitByEdges()

    def splitConglomerates(self, kmean_labels):
        assert len(self.interior) == len(kmean_labels), 'each interior point must be labeled'
        cells = {}
        for si, kl in zip(self.interior, kmean_labels):
            if kl in cells.keys():
                cells[kl].append(si)
            else:
                cells[kl] = [si]
        for k in cells.keys():
            Cluster.makeCell(self.stack_slice, self.binary, cells[k], [])


    def getBoundary2D(self):
        sortedBound = self.boundary[:]
        sortedBound.sort() #default key sort tuples (i, j) by i then j
        return [ list(filter(lambda p: p[0] == y, sortedBound)) for y in range(sortedBound[0][0], sortedBound[-1][0] + 1)]


    def addInternalBoundaryHit(self):
        self.internalBoundaryHits += 1

    def get_var_pairs(self, var_bounds, index, area=False): #if area is true check for controversial region
        if len(var_bounds) <= 1:
            return []
        k = 0
        var_pairs = []
        while k < len(var_bounds) - 1:
            interrupted = False

            start_index = var_bounds[k][index]
            for btwn in range(start_index, var_bounds[k+1][index]):
                #toggle = int(not(bool(index))) #toggle btwn 0 and 1
                if index == 1:
                    if self.binary[var_bounds[k][0], btwn] == 0:
                    #if Cluster.segmented[var_bounds[k][0]][btwn] == 0:
                        interrupted = True
                        var_pairs.append([(var_bounds[k][0], start_index), (var_bounds[k][0], btwn - 1) ])
                        if area and self.binary[var_bounds[k][0]][btwn] == 0:
                            self.addInternalBoundaryHit()

                        z = 1
                        while btwn + z < var_bounds[k+1][index] and self.binary[var_bounds[k][0], btwn+z] == 0:
                            z += 1
                        start_index = btwn + z #pickup from after the internal boundary or region
                        btwn = start_index
                        #break
                else:
                    if self.binary[btwn, var_bounds[k][1]] == 0: #incorrectly accounts for contained cells
                    #if Cluster.segmented[btwn][var_bounds[k][1]] == 0:
                        interrupted = True
                        var_pairs.append([(start_index, var_bounds[k][1]), (btwn - 1, var_bounds[k][1]) ])
                        z=1
                        while btwn + z < var_bounds[k+1][index] and self.binary[btwn+z, var_bounds[k][1]] == 0:
                            z += 1
                        start_index = btwn + z #pickup from after the internal boundary or region
                        btwn = start_index
                        #break
            if not interrupted:
                var_pairs.append([var_bounds[k], var_bounds[k+1]])
            k += 1
        return var_pairs

    def contains_or_overlaps(self, other_cell):
        k = len(other_cell.boundary)
        hits = 0
        for other_p in other_cell.boundary:
            if self.pointWithin(other_p):
                hits += 1        
        return hits >= k // 3 and self.pointWithin(other_cell.center), hits >= k // 6

    @property
    @functools.lru_cache()
    def area(self):
        self.interior = []
        self.internalBoundaryHits = 0
        bounds = self.getBoundary2D() #returns row-wise 2d matrix
        area = 0
        for b in bounds: #boundary points in each row
            x_pairs = self.get_var_pairs(b, 1, True)
            if x_pairs == []:
                area += 1
            else:
                for xp in x_pairs:
                    area += (xp[1][1] - xp[0][1] + 1)
                    self.interior.extend([(xp[0][0], k) for k in range(xp[0][1], xp[1][1]+1)])
        return area

    def kill(self):
        Cluster.clusters.remove(self)

class Cell(Cluster):

    def __init__(self, stack_slice, binary, interior, internalEdges=[]): #boundary X interior

        self.interior = interior
        self.boundary = self.getBoundary(binary, interior)
        if len(self.boundary) > 25:
            self.center = (int(np.mean([p[0] for p in self.boundary])), int(np.mean([p[1] for p in self.boundary])))
            self.gridSquare = Cluster.current_stack.grid.getGridSquare(self.center)
            self.gridSquare.addCell(self)
            super().__init__(binary, self.boundary, stack_slice, internalEdges=internalEdges)
            self.stack_slice.addCell(self)

    def getBoundary(self, binary, interior):
        boundary = []
        for si in interior:
            added = False
            k=0
            for n in getNeighborIndices(binary, si[0], si[1]):
                k+=1
                if n not in interior:
                    boundary.append(si)
                    added = True
                    break
            if (not added) and k != 8:
                boundary.append(si)
        return boundary

    def pointWithin(self, point):
        return point in self.interior


    @property
    @functools.lru_cache()
    def roundness(self):
        circum = len(self.boundary)
        pp_score = 4 * math.pi * self.area / (circum**2)
        return pp_score
        # ideal = circum / (2 * math.pi)

        # def distance(p1, p2):
        #     d = math.sqrt( (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        #     return d

        # # normalized_distances = list(np.sqrt([ ((distance(b, self.center) - ideal)/ideal)**2 for b in self.boundary]))
        # distance_sequence = [distance(b, self.center) for b in self.boundary]
        # #print(distance_sequence)
        # return 1- np.std(distance_sequence) / ideal
    @property
    def area(self):
        if len(self.interior) > 0:
            return len(self.interior)
        return Cluster.area(self)
    
    @property
    @functools.lru_cache()
    def isLoaded(self):
        x = 0
        for (i,j) in self.interior:
            if Cluster.pic[i,j] > 0.8 * WHITE:
                x += 1
        return x > 0.8 * len(self.interior)

    def kill(self):
        if self in self.stack_slice.cells:
            self.stack_slice.removeCell(self)
