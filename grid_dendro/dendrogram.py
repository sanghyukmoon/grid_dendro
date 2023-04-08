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

        # Sort flat indices in an ascending order of arr.
        arr_flat = arr.flatten()
        self._cells_ordered = arr_flat.argsort()
        self._num_cells = len(self._cells_ordered)

        # Create leaf nodes by finding all local minima.
        if self._boundary_flag == 'periodic':
            filter_mode = 'wrap'
        else:
            msg = f"Boundary flag {self.boundary_flag} is not supported"
            raise ValueError(msg)
        arr_min_filtered = minimum_filter(arr, size=3, mode=filter_mode)
        self.minima = np.where(arr_flat == arr_min_filtered.flatten())[0]

    def construct(self):
        """Construct dendrogram to set node, parent, children, and descendants"""

        # Initialize node, parent, children, descendants, and ancestor
        self.nodes = {nd: [nd] for nd in self.minima}
        self.parent = np.full(self._num_cells, -1, dtype=int)
        self.parent[self.minima] = self.minima
        self.children = {nd: [] for nd in self.minima}
        self.descendants = {nd: [] for nd in self.minima}
        self.ancestor = {nd: nd for nd in self.minima}

        # Load neighbor dictionary.
        my_neighbors = boundary.precompute_neighbor(self._arr_shape,
                                                    self._boundary_flag,
                                                    corner=True)

        # Climb up the potential and construct dendrogram.
        num_remaining_nodes = len(self.minima) - 1
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
                num_remaining_nodes -= 1
                msg = ("Added a new node at the critical point. "
                       f"number of remaining nodes = {num_remaining_nodes}")
                print(msg)
                if num_remaining_nodes == 0:
                    print("We have reached the trunk. Stop climbing up")
                    break
        self._find_leaf()

    def prune(self, ncells_min=27):
        """Prune the buds by applying minimum number of cell criterion"""
        for leaf in self.leaves:
            ncells = len(self.nodes[leaf])
            if ncells < ncells_min:
                # this leaf is a bud.
                my_parent = self.parent[leaf]
                my_grandparent = self.parent[my_parent]
                sibling = self.children[my_parent]
                sibling.remove(leaf)
                sibling = sibling[0]
                if sibling in self.leaves and len(self.leaves[sibling]) < ncells_min:
                    # TODO(Should prune the smaller bud)
                    print("WARNING: sibling is also a bud")
                    print("leaf = ", leaf)
                    print("sibling = ", sibling)

                if (my_parent == my_grandparent):
                    # This is a bud at the trunk. Cut it and define new trunk
                    orphaned_cells = (self.nodes[my_parent]
                                      + self.nodes[leaf]
                                      + self.nodes[sibling])
                    for cell in orphaned_cells:
                        self.parent[cell] = -1
                    # sibling becomes the trunk node
                    self.nodes[sibling] = [sibling,]
                    self.parent[sibling] = sibling
                    # Remove orphaned node
                    for nd in [my_parent, leaf]:
                        self.nodes.pop(nd)
                        self.children.pop(nd)
                        self.descendants.pop(nd)
                else:
                    # Reset parent
                    orphaned_cells = (self.nodes[my_parent]
                               + self.nodes[leaf])
                    for cell in orphaned_cells:
                        self.parent[cell] = sibling
                        self.nodes[sibling].append(cell)
                    self.parent[sibling] = my_grandparent
                    # Reset children
                    self.children[my_grandparent].remove(my_parent)
                    self.children[my_grandparent].append(sibling)
                    # Reset descendants
                    self.descendants[my_grandparent].remove(my_parent)
                    self.descendants[my_grandparent].remove(leaf)
                    # Remove orphaned node
                    for nd in [my_parent, leaf]:
                        self.nodes.pop(nd)
                        self.children.pop(nd)
                        self.descendants.pop(nd)
        self._find_leaf()

    def delete_node(self, nd):
        if len(self.children[nd]) > 0:
            raise ValueError("Inner node deletion is not yet implemented")
        parent_node = self.parent[nd]
        ancestor_node = self.ancestor[nd]
        for cell in self.nodes[nd]:
            self.parent[cell] = -1
        self.nodes.pop(nd)
        self.children.pop(nd)
        self.descendants.pop(nd)
        self.ancestor.pop(nd)
        self.children[parent_node].remove(nd)
        self.descendants[parent_node].remove(nd)
        # climb up the family tree and remove me from family register
        while parent_node != ancestor_node:
            parent_node = self.parent[parent_node]
            self.descendants[parent_node].remove(nd)

    def check_sanity(self):
        for nd in self.nodes:
            if not (self._num_children(nd) == 2 or self._num_children(nd) == 0):
                raise ValueError("number of children is not 2")
        print("Sane.")

    def _find_leaf(self):
        self.leaves = {}
        for nd in self.nodes:
            if self._num_children(nd) == 0:
                self.leaves[nd] = self.nodes[nd]

    def _num_children(self, nd):
        return len(self.children[nd])

    def _subsume(self, src_nodes, dst_node):
        """Subsume selected nodes into a destination node.

        Reassign all cells contained in src_nodes to dst_node and delete
        src_nodes from the tree.

        Args:
          src_nodes: nodes to be subsumed into other branch.
          dst_node: destination node that subsumes src_nodes.
        """
        parents = {self.parent[nd] for nd in src_nodes}
        parents.add(self.parent[dst_node])
        if len(parents) != 1:
            raise ValueError("Subsume operation can only be done at the same"
                              "level in a dendrogram hierarchy")
        parent = parents.pop()
        orphaned_cells = []
        for nd in src_nodes:
            orphaned_cells += self.nodes[nd]
            self.delete_node(nd)
        for cell in orphaned_cells:
            self.parent[cell] = dst_node
            self.nodes[dst_node].append(cell)


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
        dtype='xarray'
        coords = dat.coords
        dims = dat.dims
        dat = dat.to_numpy()
    elif isinstance(dat, np.ndarray):
        dtype='numpy'
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
