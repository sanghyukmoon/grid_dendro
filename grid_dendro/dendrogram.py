"""Implements Dendrogram class"""

import numpy as np
import xarray as xr
from scipy.ndimage import minimum_filter
from grid_dendro import boundary


class Dendrogram:
    """Dendrogram representing hierarchical structure in 3D data

    Attributes:
        nodes: dictionary mapping all nodes in dendrogram hierarchy to the
          cells belong to them.
          Access pattern: {node: cells}
        parent: numpy array mapping all cells to their parent node. When a cell
          is a node-generating cell and therefore itself a node, its parent is
          not itself, but the parent node of it.
          Access pattern: {cells: node}
        children: dictionary mapping all nodes to their child nodes. Note that
          leaf nodes have no children.
          Access pattern: {node: list of nodes}
        ancestor: dictionary mapping all nodes to their ancestor node.
          Access pattern: {node: node}
        descendants: dictionary mapping all nodes to their descendant nodes.
          Access pattern: {node: list of nodes}
        leaves: dictionary mapping all leaf nodes to the cells belong to them.
          Access pattern: {node: cells}
        minima: indices of the cell at the local minima of input array.
    """

    def __init__(self, arr, boundary_flag='periodic'):
        """Initializes the instance based on spam preference.

        Find minima in constructor and do not store input array to save memory.

        Args:
          arr: numpy.ndarray instance representing the input data.
          boundary_flag: string representing the boundary condition, optional.
        """
        self._arr_shape = arr.shape
        self._boundary_flag = boundary_flag

        # Create leaf nodes by finding all local minima.
        if self._boundary_flag == 'periodic':
            filter_mode = 'wrap'
        else:
            msg = f"Boundary flag {self.boundary_flag} is not supported"
            raise ValueError(msg)
        arr_min_filtered = minimum_filter(arr, size=3, mode=filter_mode)
        arr = arr.flatten()
        self.minima = np.where(arr == arr_min_filtered.flatten())[0]

        # Sort flat indices in an ascending order of arr.
        self._cells_ordered = arr.argsort()
        self._num_cells = len(self._cells_ordered)

    def construct(self):
        """Construct dendrogram tree

        Initialize nodes, parent, children, ancestor, descendants, and leaves
        """

        # Initialize node, parent, children, descendants, and ancestor
        self.nodes = {nd: [nd] for nd in self.minima}
        self.parent = np.full(self._num_cells, -1, dtype=int)
        self.parent[self.minima] = self.minima
        self.children = {nd: [] for nd in self.minima}
        self.ancestor = {nd: nd for nd in self.minima}
        self.descendants = {nd: [] for nd in self.minima}

        # Load neighbor dictionary.
        my_neighbors = boundary.precompute_neighbor(self._arr_shape,
                                                    self._boundary_flag,
                                                    corner=True)

        print("Start climbing up the tree from the leaf nodes.\t"
              f"Number of nodes = {len(self.nodes)}")
        # Climb up the potential and construct dendrogram.
        for cell in iter(self._cells_ordered):
            if cell in self.minima:
                continue
            # Find parents of neighboring cells.
            parents = set(self.parent[my_neighbors[cell]])
            parents.discard(-1)

            # Find ancestors of their parents, which can be themselves.
            neighboring_nodes = set([self.ancestor[prnt] for prnt in parents])
            num_nghbr_nodes = len(neighboring_nodes)

            if num_nghbr_nodes == 0:
                raise ValueError("Should not reach here")
            elif num_nghbr_nodes == 1:
                # Add this cell to the existing node
                nd = neighboring_nodes.pop()
                self.parent[cell] = nd
                self.nodes[nd].append(cell)
            elif num_nghbr_nodes >= 2:
                # This cell is at the critical point; create new node.
                self.nodes[cell] = [cell]
                self.parent[cell] = cell
                self.ancestor[cell] = cell
                self.children[cell] = list(neighboring_nodes)
                self.descendants[cell] = list(neighboring_nodes)
                for child in self.children[cell]:
                    # This node becomes a parent of its immediate children
                    self.parent[child] = cell
                    # inherit all descendants of children
                    self.descendants[cell] += self.descendants[child]
                for child in self.descendants[cell]:
                    # This node becomes a new ancestor of all its
                    # descendants
                    self.ancestor[child] = cell
                print("Added a new node at the critical point.\t\t"
                      f"Number of nodes = {len(self.nodes)}")
                if set(self.minima).issubset(set(self.descendants[cell])):
                    print("We have reached the trunk. Stop climbing up")
                    break
        self._find_leaves()

    def prune(self, ncells_min=27):
        """Prune the buds by applying minimum number of cell criterion

        Args:
          ncells_min: minimum number of cells to be a leaf, optional.
        """
        bud = self._find_bud(ncells_min)
        while bud is not None:
            parent = self.parent[bud]
            siblings = set(self.children[parent])
            sibling_buds = {nd for nd in siblings if nd in self.leaves
                            and len(self.nodes[nd]) < ncells_min}
            sibling_branches = siblings - sibling_buds

            if len(sibling_branches) >= 2:
                # There are at least two branches at this node.
                # Simply cut the buds.
                print("There are at least two branches. Simply cut the buds")
                for bud in sibling_buds:
                    print(f"Cutting the bud {bud}")
                    self._cut_bud(bud)
            elif len(sibling_branches) == 1:
                # There are only one branch.
                # Subsume buds to the branch and remove the resulting knag.
                branch = sibling_branches.pop()
                print(f"Subsume buds {sibling_buds} into a branch {branch}")
                self._subsume_buds(sibling_buds, branch)
            else:
                # Subsume short buds to the longest bud and remove the knag.
                shorter_buds = sibling_buds.copy()
                longest_bud = sibling_buds.pop()
                rank_min = np.where(self._cells_ordered == longest_bud)[0][0]
                while len(sibling_buds) > 0:
                    nd = sibling_buds.pop()
                    rank = np.where(self._cells_ordered == nd)[0][0]
                    if rank < rank_min:
                        longest_bud = nd
                        rank_min = rank
                shorter_buds.remove(longest_bud)
                print("There are only buds; "
                      "merge shorter buds {} to longest bud {}".format(
                       shorter_buds, longest_bud))
                self._subsume_buds(shorter_buds, longest_bud)
            self._find_leaves()
            bud = self._find_bud(ncells_min)

    def _cut_bud(self, bud):
        if len(self.children[bud]) > 0:
            raise ValueError("This is not a bud")
        parent_node = self.parent[bud]
        ancestor_node = self.ancestor[bud]
        if parent_node == bud:
            raise ValueError("Cannot delete trunk")

        # climb up the family tree and remove this node from family register.
        self.children[parent_node].remove(bud)
        while True:
            self.descendants[parent_node].remove(bud)
            if parent_node == ancestor_node:
                break
            else:
                parent_node = self.parent[parent_node]
        parent_node = self.parent[bud]

        # Remove this node
        orphans = self.nodes[bud]
        for orphan in orphans:
            self.parent[orphan] = -1

        self.nodes.pop(bud)
        self.children.pop(bud)
        self.descendants.pop(bud)
        self.ancestor.pop(bud)

        return orphans

    def _subsume_buds(self, buds, branch):
        """Subsume selected buds into a branch.

        Reassign all cells contained in buds to branch and delete
        buds from the tree.

        Args:
          buds: buds to be subsumed into other branch.
          branch: destination branch that subsumes bud.
        """
        parents = {self.parent[bud] for bud in buds}
        parents.add(self.parent[branch])
        if len(parents) != 1:
            raise ValueError("Subsume operation can only be done within the "
                             "same generation.")
        knag = parents.pop()
        orphans = []
        for bud in buds:
            orphans += self._cut_bud(bud)
        for orphan in orphans:
            self.parent[orphan] = branch
            self.nodes[branch].append(orphan)

        self._remove_knag(knag)

    def _remove_knag(self, knag):
        """Remove a knag that can results from subsume operation.

        Knag is a node that have only one child. Knag forms when buds are
        subsumed into a branch (or a longest bud when there are only buds).
        """
        if len(self.children[knag]) != 1:
            raise ValueError("This is not a knag.")
        child_node = self.children[knag][0]
        parent_node = self.parent[knag]
        ancestor_node = self.ancestor[knag]

        if parent_node == knag:
            # This knag was a trunk. Now, child_node becomes a new trunk.
            self.parent[child_node] = child_node
            for nd in self.descendants[knag]:
                self.ancestor[nd] = child_node
        else:
            # climb up the family tree and reset the family register.
            self.children[parent_node].remove(knag)
            self.children[parent_node].append(child_node)
            while True:
                self.descendants[parent_node].remove(knag)
                if parent_node == ancestor_node:
                    break
                else:
                    parent_node = self.parent[parent_node]
            parent_node = self.parent[knag]

            # climb down the family tree and reset the family register.
            self.parent[child_node] = parent_node

        # Remove this node
        orphans = self.nodes[knag]
        for orphan in orphans:
            self.parent[orphan] = child_node
            self.nodes[child_node].append(orphan)

        self.nodes.pop(knag)
        self.children.pop(knag)
        self.descendants.pop(knag)
        self.ancestor.pop(knag)

    def _find_leaves(self):
        """Find leaf nodes."""
        self.leaves = {}
        for nd in self.nodes:
            if len(self.children[nd]) == 0:
                self.leaves[nd] = self.nodes[nd]

    def _find_bud(self, ncells_min):
        """Loop through all leaves and return the first occurence of a bud.

        Args:
          ncells_min: minimum number of cells to be a leaf, optional.

        Returns:
          leaf: bud node.
        """
        for leaf in self.leaves:
            ncells = len(self.nodes[leaf])
            if ncells < ncells_min:
                return leaf
        return None


def filter_by_node(dat, nodes=None, nodes_select=None, cells_select=None,
                   fill_value=np.nan):
    """Mask DataArray using FISO dictionary or the flattened indexes.

    Args:
        dat: input array to be filtered.
          Supported data types: xarray.DataArray, numpy.ndarray
        nodes: grid_dendro nodes dictionary, optional.
        nodes_select: int or sequence of ints representing the selected nodes,
          optional.
        cells_select: flat indices of selected cells. If given, overrides nodes
          and nodes_select, optional
        fill_value: value to fill outside of the filtered region, optional.
                    Default value is np.nan.

    Returns:
        out: Filtered array matching the input array type
    """
    if isinstance(dat, xr.DataArray):
        dtype = 'xarray'
        coords = dat.coords
        dims = dat.dims
        dat = dat.to_numpy()
    elif isinstance(dat, np.ndarray):
        dtype = 'numpy'
    else:
        raise TypeError("type {} is not supported".format(type(dat)))

    # retreive flat indices of selected cells
    if nodes is None and nodes_select is None and cells_select is None:
        # nothing to do
        return dat
    elif nodes is not None and cells_select is None:
        cells_select = []
        if nodes_select is None:
            # select all cells
            for v in nodes.values():
                cells_select += list(v)
        elif isinstance(nodes_select, (int, np.int64, np.int32)):
            cells_select += nodes[nodes_select]
        else:
            for node in nodes_select:
                cells_select += nodes[node]

    dat1d = dat.flatten()
    out = np.full(len(dat1d), fill_value)
    out[cells_select] = dat1d[cells_select]
    out = out.reshape(dat.shape)
    if dtype == 'xarray':
        out = xr.DataArray(data=out, coords=coords, dims=dims)
    return out
