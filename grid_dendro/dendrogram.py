"""Implements Dendrogram class"""

import numpy as np
import xarray as xr
from scipy.ndimage import minimum_filter
from grid_dendro import boundary


class Dendrogram:
    """Dendrogram representing hierarchical structure in 3D data

    Attributes
    ----------
    nodes : dict
        Maps a node to flat indices of its member cells.
        {node: cells}
    parent : dict
        Maps a node to its parent node.
        {node: node}
    children : dict
        Maps a node to its child nodes.
        {node: list of nodes}
    ancestor : dict
        Maps a node to its ancestor.
        {node: node}
    descendants : dict
        Maps a node to its descendant nodes.
        {node: list of nodes}
    leaves : list
        List of nodes that do not have children.
    trunk : int
        Id of the trunk node.
    minima : set
        Flat indices at the local potential minima.
    """

    def __init__(self, arr, boundary_flag='periodic', verbose=True):
        """Read data and find local miniima

        Parameters
        ----------
        arr : numpy.ndarray
            Input array.
        boundary_flag : str, optional
            String representing boundary condition.
        """
        self._arr_shape = arr.shape
        self._boundary_flag = boundary_flag
        self._verbose = verbose

        # Create leaf nodes by finding all local minima.
        if self._boundary_flag == 'periodic':
            filter_mode = 'wrap'
        else:
            msg = f"Boundary flag {self.boundary_flag} is not supported"
            raise ValueError(msg)
        arr_min_filtered = minimum_filter(arr, size=3, mode=filter_mode
                                          ).flatten()
        arr = arr.flatten()
        self.minima = set((arr == arr_min_filtered).nonzero()[0])

        # Sort flat indices in an ascending order of arr.
        self.cells_ordered = arr.argsort()
        self.num_cells = len(self.cells_ordered)

        # Set data type based on the size of array
        if self.num_cells < 2**31:
            self._dtype = np.int32
        elif self.num_cells < 2**63:
            self._dtype = np.int64
        else:
            raise MemoryError("Input array size is too large")

    def construct(self):
        """Construct dendrogram

        Constructs dendrogram dictionaries: nodes, parent, children,
        ancestor, and descendants. Then finds leaf nodes and trunk node.

        Notes
        -----
        Use int64 in a loop which seems to be faster in 64 bit system, and
        then cast to int32 or int64 depending on the size of the array
        after construction is done, in order to save memory. Similarly, use
        list instead of numpy array for efficient append operation, then
        cast to numpy array after construction is done to save memory.
        These memory optimization is only done to "nodes" and
        "cells_ordered", which consume most of the memory.

        Examples
        --------
        >>> import pyathena as pa
        >>> s = pa.LoadSim("/scratch/smoon/M5J2P0N512")
        >>> ds = s.load_hdf5(50, load_method='pyathena')
        >>> gd = dendrogram.Dendrogram(ds.phigas.to_numpy())
        >>> gd.construct()
        >>> gd.prune()  # Remove buds
        """

        # Initialize node, parent, children, descendants, and ancestor
        self.nodes = {nd: [nd] for nd in self.minima}
        self.children = {nd: [] for nd in self.minima}
        self.ancestor = {nd: nd for nd in self.minima}
        self.descendants = {nd: [] for nd in self.minima}

        # parent_array is a numpy array that maps a cell to its "parent" node.
        # When a cell is itself a node-generating point, than it is mapped to
        # its parent node.
        parent_array = np.full(self.num_cells, -1, dtype=int)
        parent_array[list(self.minima)] = list(self.minima)

        # Load neighbor dictionary.
        my_neighbors = boundary.precompute_neighbor(self._arr_shape,
                                                    self._boundary_flag,
                                                    corner=True)
        if self._verbose:
            print("Start climbing up the tree from the leaf nodes.\t"
                  f"Number of nodes = {len(self.nodes)}")
        # Climb up the potential and construct dendrogram.
        for cell in iter(self.cells_ordered):
            if cell in self.minima:
                # Performance critical to have type(self.minima) = set for
                # efficient "in" operation.
                continue
            # Find parents of neighboring cells.
            parents = set(parent_array[my_neighbors[cell]])
            parents.discard(-1)

            # Find ancestors of their parents, which can be themselves.
            neighboring_nodes = set([self.ancestor[prnt] for prnt in parents])
            num_nghbr_nodes = len(neighboring_nodes)

            if num_nghbr_nodes == 0:
                raise ValueError("Should not reach here")
            elif num_nghbr_nodes == 1:
                # Add this cell to the existing node
                nd = neighboring_nodes.pop()
                parent_array[cell] = nd
                self.nodes[nd].append(cell)
            elif num_nghbr_nodes >= 2:
                # This cell is at the critical point; create new node.
                self.nodes[cell] = [cell]
                parent_array[cell] = cell
                self.ancestor[cell] = cell
                self.children[cell] = list(neighboring_nodes)
                self.descendants[cell] = list(neighboring_nodes)
                for child in self.children[cell]:
                    # This node becomes a parent of its immediate children
                    parent_array[child] = cell
                    # inherit all descendants of children
                    self.descendants[cell] += self.descendants[child]
                for child in self.descendants[cell]:
                    # This node becomes a new ancestor of all its
                    # descendants
                    self.ancestor[child] = cell
                if self._verbose:
                    print("Added a new node at the critical point.\t\t"
                          f"Number of nodes = {len(self.nodes)}")
                if set(self.minima).issubset(set(self.descendants[cell])):
                    if self._verbose:
                        print("We have reached the trunk. Stop climbing up")
                    break

        # Construct parent dictionary from parent_array
        self.parent = {}
        for k, v in self.nodes.items():
            # Cast to save memory
            self.parent[k] = parent_array[k].astype(self._dtype)
            self.nodes[k] = np.array(v).astype(self._dtype)
        self.cells_ordered = self.cells_ordered.astype(self._dtype)

        # Find leaves and trunk
        self._find_leaves()
        self._find_trunk()

    def prune(self, ncells_min=27):
        """Prune the buds by applying minimum number of cell criterion

        Parameters
        ----------
        ncells_min : int, optional
            Minimum number of cells of a leaf node.
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
                if self._verbose:
                    print("There are at least two branches. Simply cut the buds")
                for bud in sibling_buds:
                    if self._verbose:
                        print(f"Cutting the bud {bud}")
                    self._cut_bud(bud)
            elif len(sibling_branches) == 1:
                # There are only one branch.
                # Subsume buds to the branch and remove the resulting knag.
                branch = sibling_branches.pop()
                if self._verbose:
                    print(f"Subsume buds {sibling_buds} into a branch {branch}")
                self._subsume_buds(sibling_buds, branch)
            else:
                # Subsume short buds to the longest bud and remove the knag.
                shorter_buds = sibling_buds.copy()
                longest_bud = sibling_buds.pop()
                rank_min = np.where(self.cells_ordered == longest_bud)[0][0]
                while len(sibling_buds) > 0:
                    nd = sibling_buds.pop()
                    rank = np.where(self.cells_ordered == nd)[0][0]
                    if rank < rank_min:
                        longest_bud = nd
                        rank_min = rank
                shorter_buds.remove(longest_bud)
                if self._verbose:
                    print("There are only buds; "
                          "merge shorter buds {} to longest bud {}".format(
                               shorter_buds, longest_bud))
                self._subsume_buds(shorter_buds, longest_bud)

            self._find_leaves()
            bud = self._find_bud(ncells_min)
        self._find_trunk()

    def get_all_descendant_cells(self, node):
        """Return all member cells of the node, including descendant nodes

        Parameters
        ----------
        node : int
            Id of a selected node.

        Returns
        -------
        cells : array
            Flattned indices of all member cells of this node.
        """
        cells = self.nodes[node].copy()
        for nd in self.descendants[node]:
            cells = np.concatenate((cells, self.nodes[nd]))
        return cells

    def len(self, node):
        """
        Returns the number of cells in a node (including descendants)

        Parameters
        ----------
        node : int
            ID of a selected node.

        Returns
        -------
        int
            Number of cells in this node
        """
        ncells = len(self.nodes[node])
        for nd in self.children[node]:
            ncells += self.len(nd)
        return ncells

    def sibling(self, node):
        """Returns my sibling

        Parameter
        ---------
        node : int
            ID of a selected node.

        Returns
        -------
        int
            ID of my sibling node
        """
        if node == self.trunk:
            raise ValueError("Trunk has no sibling")
        parent = self.parent[node]
        sibling = self.children[parent].copy()
        sibling.remove(node)
        return sibling[0]

    def filter_data(self, dat, nodes, fill_value=np.nan, drop=False):
        """Filter data by node, including all descendant nodes.

        Get all member cells of the nodes and their descendants and return
        the masked data.

        Parameters
        ----------
        dat : xarray.DataArray or numpy.ndarray
            Input array to be filtered.
        nodes : int or array of ints
            Selected nodes.
        fill_value : float, optional
            The value to fill outside of the filtered region. Default to nan.
        drop : bool
            If true, return faltten data that only include filtered cells.

        Returns
        -------
        out : xarray.DataArray or numpy.ndarray
            Filtered array matching the input array type

        See Also
        --------
        dendrogram.filter_by_dict

        Notes
        -----
        If `dat` is already 1-D flattend array, return 1-D array that only
        contains filtered region. Otherwise, return as the original shape
        and data type unless `drop`=True.

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
        if isinstance(nodes, (int, np.int64, np.int32)):
            cells = self.get_all_descendant_cells(nodes)
        else:
            cells = []
            for node in nodes:
                cells += list(self.get_all_descendant_cells(node))

        shape = dat.shape
        dim = len(shape)
        if dim == 1:
            # If dat is already flattend, return filtered 1d array
            out = dat[cells]
            return out
        else:
            dat = dat.flatten()
            if drop:
                out = dat[cells]
                return out
            else:
                out = np.full(len(dat), fill_value, dtype=dat.dtype)
                out[cells] = dat[cells]
                out = out.reshape(shape)
                if dtype == 'xarray':
                    out = xr.DataArray(data=out, coords=coords, dims=dims)
                return out

    def reindex(self, start_indices, shape):
        """Transform global indices to local indices

        When only a part of the entire domain is loaded, the indices of the cell
        becomes different from the global indices. This function do the
        required reindexing. The node IDs are not transformed. Assumes the
        index ordering is (k, j, i).

        Parameters
        ----------
        start_indices : array
            (ks, js, is) of the local domain
        shape : array
            (nz, ny, nx) of the local domain
        """
        for k, v in self.nodes.items():
            local_indices = np.unravel_index(v, self._arr_shape) - start_indices[:,None]
            self.nodes[k] = np.unique(np.ravel_multi_index(local_indices, shape, mode='clip'))

    def _cut_bud(self, bud):
        """Remove bud node and return its member cells

        Parameters
        ----------
        bud : int
            Id of a selected node.

        Returns
        -------
        cells : array
            Flattned indices of all member cells of this node.

        See Also
        --------
        _remove_knag
        """
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

        # Remove this node
        orphans = self.nodes[bud]
        self.nodes.pop(bud)
        self.parent.pop(bud)
        self.children.pop(bud)
        self.ancestor.pop(bud)
        self.descendants.pop(bud)

        return orphans

    def _subsume_buds(self, buds, branch):
        """Subsume selected buds into a branch.

        Reassign all cells contained in buds to branch and delete
        buds from the tree.

        Parameters
        ----------
        buds : array of ints
            IDs of the bud nodes to be subsumed into other branch.
        branch : int
            Destination branch that subsumes bud.
        """
        parents = {self.parent[bud] for bud in buds}
        parents.add(self.parent[branch])
        if len(parents) != 1:
            raise ValueError("Subsume operation can only be done within the "
                             "same generation.")
        for bud in buds:
            self.nodes[branch] = np.concatenate((self.nodes[branch],
                                                 self._cut_bud(bud)))

        # Remove knag node resulting from subsume.
        # knag node is the pseudo-node that have only one child.
        knag = parents.pop()
        self._remove_knag(knag)

    def _remove_knag(self, knag):
        """Remove a knag that can results from subsume operation.

        Knag is a node that have only one child. Knag forms when buds are
        subsumed into a branch (or into a longest bud when there are only
        buds).

        Parameters
        ----------
        knag : int
            Id of a selected node.

        See Also
        --------
        _cut_bud
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
        self.nodes[child_node] = np.concatenate((self.nodes[child_node],
                                                 self.nodes[knag]))
        self.nodes.pop(knag)
        self.parent.pop(knag)
        self.children.pop(knag)
        self.ancestor.pop(knag)
        self.descendants.pop(knag)

    def _find_leaves(self):
        """Find leaf nodes."""
        self.leaves = []
        for nd in self.nodes:
            if len(self.children[nd]) == 0:
                self.leaves.append(nd)

    def _find_bud(self, ncells_min):
        """Loop through all leaves and return the first occurence of a bud.

        Parameters
        ----------
        ncells_min : int
            Minimum number of cells of a leaf node.

        Returns
        -------
        leaf : int
            ID of the bud node.
        """
        for leaf in self.leaves:
            ncells = len(self.nodes[leaf])
            if ncells < ncells_min:
                return leaf
        return None

    def _find_trunk(self):
        # TODO use set instead of np.unique
        trunk = np.unique(list(self.ancestor.values()))
        if len(trunk) != 1:
            raise Exception("There are more than one trunk."
                            " Something must be wrong")
        else:
            self.trunk = trunk[0]


def filter_by_dict(dat, node_dict=None, cells=None,
                   fill_value=np.nan, drop=False):
    """Filter data by node dictionary or array of cell indices.

    This is a stand-alone filtering function that offers more flexibility
    to Dendrogram.filter_data method.

    Parameters
    ----------
    dat : xarray.DataArray or numpy.ndarray
        Input array to be filtered.
    node_dict : dict, optional
        Dictionary that maps node id to flat indices of member cells.
    cells : array, optional
        Flat indices of selected cells. Overrides node_dict.
    fill_value : float, optional
        Value to fill outside of the filtered region. Default is np.nan.
    drop : bool, optional
        If true, return flatten dat that only include selected cells.

    Returns
    -------
    out : xarray.DataArray or numpy.ndarray
        Filtered array matching the input array type

    See Also
    --------
    Dendrogram.filter_data
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
    if cells is None:
        cells = []
        # select all cells
        for v in node_dict.values():
            cells += list(v)

    dat1d = dat.flatten()
    if drop:
        out = dat1d[cells]
        return out
    else:
        out = np.full(len(dat1d), fill_value)
        out[cells] = dat1d[cells]
        out = out.reshape(dat.shape)
        if dtype == 'xarray':
            out = xr.DataArray(data=out, coords=coords, dims=dims)
        return out
