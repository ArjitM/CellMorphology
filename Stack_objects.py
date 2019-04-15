from Cell_objects import *

class GridSquare:

    def __init__(self, left, right, top, bottom):
        self.coordinates = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        self._cells = []
        self.neighbors = []
        self.large_Cells = []

    def addCell(self, cell):
        self._cells.append(cell)


class Grid:
    def __init__(self, xmax, ymax, squareSize):
        self.compartments = []
        i_max, j_max, ki, kj = ymax, xmax, 0, 0
        while ki < i_max:
            while kj < j_max:
                self.compartments.append(GridSquare(kj, min(kj+squareSize, j_max), ki, min(ki+squareSize, i_max)))
                kj += squareSize
            kj = 0
            ki+= squareSize
        self.squareSize = squareSize
        self.i_max = i_max
        self.j_max = j_max
        for c in self.compartments:
            self.setNeighborGridSquares(c)

    def getGridSquare(self, p):
        i, j = p
        index = (j // self.squareSize) + (i // self.squareSize) * (self.j_max // self.squareSize)
        assert i >= 0 and j>= 0 and index < len(self.compartments)
        return self.compartments[index]

    def setNeighborGridSquares(self, compartment):
        neighbors = []
        neighbor_points = [(compartment.coordinates['top'], compartment.coordinates['left'] - 1),
        (compartment.coordinates['top'] - 1, compartment.coordinates['left'] - 1),
        (compartment.coordinates['top'] - 1, compartment.coordinates['left']),
        (compartment.coordinates['bottom'], compartment.coordinates['right'] + 1),
        (compartment.coordinates['bottom'] + 1, compartment.coordinates['right'] + 1),
        (compartment.coordinates['bottom'] + 1, compartment.coordinates['right'])]
        for npoint in neighbor_points:
            try:
                neighbors.append(self.getGridSquare(npoint))
            except AssertionError:
                pass
        compartment.neighbors = neighbors


class Stack_slice:

    def __init__(self, number, cells):
        self.cells = cells
        self.number = number
        self.finalizedCellSlice = Stack_slice_largest(number, None)
        #reasons for cell candidate rejection
        self.roundness_rejected_Cells = []
        self.contained_Cells = []
        self.split_Cells = []
        self.overlapping_Cells = []
        self.size_rejected_Cells =[]

    def addCell(self, cell):
        if isinstance(cell, Cell):
            self.cells.append(cell)
        else:
            print("not a cell instance")

    def removeCell(self, cell):
        self.cells.remove(cell)

    def pruneCells(self, roundness_thresh=0.75):
        
        for c in self.cells:
            if c.roundness < roundness_thresh:
                self.roundness_rejected_Cells.append(c)
                #self.cells.remove(c)
                continue
            if c.area < 90:# or c.area > 120: #55 corresponds to <17 microns squared. RGCs are 20-30 um^2
                self.size_rejected_Cells.append(c)
                #self.cells.remove(c)
        for c in self.roundness_rejected_Cells:
            self.removeCell(c)
        for c in self.size_rejected_Cells:
            self.removeCell(c)

class Stack_slice_largest(Stack_slice):

    def __init__(self, number, cells):
        self.cells = cells if cells is not None else []
        self.number = number


class Stack:

    def __init__(self, xmax, ymax, squareSize, stack_slices=[]):
        self.stack_slices = list(set(stack_slices))
        self.large_Cells = []
        self.grid = Grid(xmax, ymax, squareSize)

    def addSlice(self, stack_slice):
        self.stack_slices.append(stack_slice)

    def collate_slices(self, nucleusMode):
        for stack_slice in self.stack_slices:
            for cell in stack_slice.cells:
                hits = 0
                neighbor_cells = []
                for ngs in cell.gridSquare.neighbors:
                    neighbor_cells.extend(ngs.large_Cells)
                for large_Cell in cell.gridSquare.large_Cells + neighbor_cells:
                    if large_Cell is cell:
                        continue
                    if cell.contains_or_overlaps(large_Cell)[0]:
                        large_replace = large_Cell
                        hits += 1
                    if hits > 1:
                        break
                if hits == 1:
                    self.large_Cells.remove(large_replace)
                    large_replace.gridSquare.large_Cells.remove(large_replace)
                    large_replace.stack_slice.contained_Cells.append(large_replace)
                    self.large_Cells.append(cell)
                    cell.gridSquare.large_Cells.append(cell)
                else:
                    large_replace = set()
                    new_cell = True
                    for large_Cell in cell.gridSquare.large_Cells + neighbor_cells:
                        if large_Cell is cell:
                            continue
                        assert large_Cell in self.large_Cells, 'grid large should add up'
                        (contained, overlapping) = large_Cell.contains_or_overlaps(cell)
                        if contained: #contained
                            new_cell = False
                            cell.stack_slice.contained_Cells.append(cell)
                            break
                        elif overlapping: #overlapping
                            if large_Cell.area > cell.area:# and large_Cell.roundness > (cell.roundness * 1.2):
                                new_cell = False
                                cell.stack_slice.overlapping_Cells.append(cell)
                                break
                            else:
                                large_replace.add(large_Cell)
                                large_Cell.stack_slice.overlapping_Cells.append(large_Cell)

                    if new_cell:
                        self.large_Cells.append(cell)
                        cell.gridSquare.large_Cells.append(cell)
                    if large_replace:
                        for lr in large_replace:
                            assert lr in self.large_Cells, 'large cells changed?{0}'.format(large_replace)
                            self.large_Cells.remove(lr)
                            lr.gridSquare.large_Cells.remove(lr)
        for lg in self.large_Cells:
            lg.stack_slice.finalizedCellSlice.addCell(lg)
 


