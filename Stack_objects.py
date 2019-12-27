from Cell_objects import *

AREA_CUTOFF = 90


class GridSquare:

    def __init__(self, left, right, top, bottom):
        self.coordinates = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        self._cells = set()
        self.neighbors = set()
        self.large_Cells = set()

    def __hash__(self):
        return self.coordinates.__hash__()

    def addCell(self, cell):
        self._cells.add(cell)


class Grid:
    def __init__(self, xmax, ymax, squareSize):
        self.compartments = set()
        i_max, j_max, ki, kj = ymax, xmax, 0, 0
        while ki < i_max:
            while kj < j_max:
                self.compartments.add(GridSquare(kj, min(kj + squareSize, j_max), ki, min(ki + squareSize, i_max)))
                kj += squareSize
            kj = 0
            ki += squareSize
        self.squareSize = squareSize
        self.i_max = i_max
        self.j_max = j_max
        for c in self.compartments:
            self.setNeighborGridSquares(c)

    def getGridSquare(self, p):
        i, j = p
        index = (j // self.squareSize) + (i // self.squareSize) * (self.j_max // self.squareSize)
        assert i >= 0 and j >= 0 and index < len(self.compartments)
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

    def __init__(self, number, stack, cells):
        self.cells = cells
        self.number = number
        self.finalizedCells = set()
        self.threeDstack = stack  # reasons for cell candidate rejection
        self.roundness_rejected_Cells = set()
        self.contained_Cells = set()
        self.split_Cells = set()
        self.overlapping_Cells = set()
        self.size_rejected_Cells = set()

    def __hash__(self):
        return self.number

    def addCell(self, cell):
        if isinstance(cell, Cell):
            self.cells.append(cell)
        else:
            print("not a cell instance")

    def removeCell(self, cell):
        self.cells.remove(cell)

    def addContainedCell(self, cell):
        self.contained_Cells.append(cell)

    def pruneCells(self, roundness_thresh=0.75):

        for c in self.cells:
            if c.roundness < roundness_thresh:
                self.roundness_rejected_Cells.add(c)
                continue
            if c.area < AREA_CUTOFF:  # 55 corresponds to <17 microns squared. RGCs are 20-30 um^2
                self.size_rejected_Cells.add(c)

        for c in self.roundness_rejected_Cells:
            self.removeCell(c)
        for c in self.size_rejected_Cells:
            self.removeCell(c)



class Stack:

    def __init__(self, xmax, ymax, squareSize):
        self.stack_slices = []
        self.large_Cells = set()
        self.grid = Grid(xmax, ymax, squareSize)

    def addSlice(self, stack_slice):
        self.stack_slices.append(stack_slice)

    def collate_slices(self, nucleusMode):

        for stack_slice in self.stack_slices:

            for cell in stack_slice.cells:

                hits = 0
                large_replace = None

                neighbor_cells = []  # all cells in neighboring grid squares
                for ngs in cell.gridSquare.neighbors:
                    neighbor_cells.extend(ngs.large_Cells)

                for large_Cell in cell.gridSquare.large_Cells + neighbor_cells:
                    if large_Cell is cell:
                        continue
                    if cell.contains_or_overlaps(large_Cell)[0]:
                        large_replace = large_Cell
                        hits += 1
                    if hits > 1:
                        large_replace.stack_slice.split_Cells.add(large_replace)
                        large_replace.grid_square.large_Cells.remove(large_replace)
                        break  # do not add cell to self large cells

                if hits == 1:
                    self.large_Cells.remove(large_replace)
                    large_replace.gridSquare.large_Cells.remove(large_replace)
                    large_replace.stack_slice.contained_Cells.add(large_replace)
                    self.large_Cells.add(cell)
                    cell.gridSquare.large_Cells.add(cell)

                else:
                    large_replace = set()
                    new_cell = True
                    for large_Cell in cell.gridSquare.large_Cells + neighbor_cells:
                        if large_Cell is cell:
                            raise ValueError("Overlap method is not working")
                        assert large_Cell in self.large_Cells, 'grid large should add up'
                        (contained, overlapping) = large_Cell.contains_or_overlaps(cell)
                        if contained:  # contained
                            new_cell = False
                            cell.stack_slice.contained_Cells.add(cell)
                            break
                        elif overlapping:  # overlapping
                            if large_Cell.area > cell.area:
                                new_cell = False
                                cell.stack_slice.overlapping_Cells.add(cell)
                                break
                            else:
                                large_replace.add(large_Cell)
                                large_Cell.stack_slice.overlapping_Cells.add(large_Cell)

                    if new_cell:
                        self.large_Cells.add(cell)
                        cell.gridSquare.large_Cells.add(cell)
                    if large_replace:
                        for lr in large_replace:
                            assert lr in self.large_Cells, 'large cells changed?{0}'.format(large_replace)
                            self.large_Cells.remove(lr)
                            lr.gridSquare.large_Cells.remove(lr)

        for lg in self.large_Cells:
            lg.stack_slice.finalizedCells.add(lg)
