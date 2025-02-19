import itertools
import numpy as np


def get_edge_cells(cells, pcn):
    """Find edge cells of the given region

    Parameters
    ----------
    cells : array_like
        flattened indices that defines some region
    pcn : array_like
        precomputed neighbors.

    Return
    ------
    edge_cells : array_like
        flattened indices that defines edge cells of the given region
    """
    cells = np.array(cells)
    # For each cell, are there neighboring cells which is not contained
    # in the given region? That is, is there any element of "pcn" which is
    # not contained in "cells"?
    neighbors = [pcn[i] for i in cells]
    adjacent_exterior = np.isin(neighbors, cells, invert=True)
    # If any of N_neighbor cells fall in exterior region, mark True.
    edge_mask = np.any(adjacent_exterior, axis=1)
    edge_cells = cells[edge_mask]
    return edge_cells


def precompute_neighbor(shape, boundary_flag, corner=True):
    """Precompute neighbor indices

    Parameters
    ----------
    shape : tuple
        shape of the input data
    boundary_flag : str
        Flag for boundary condition. Affects how to set neighbors of the
        edge cells.
    corner : Boolean, optional
        If true, the corner cells are counted as neighbors
        (26 neighbors in total)

    Returns
    -------
    pcn : pcnDict
        This is a pseudo-array of the shape (Ncells, N_neighbors), such that
        pcn[1][0] is the flattened index of the (-1,-1,-1) neighbor of the
        (k,j,i) = (0,0,1) cell, which is
        (k,j,i) = (-1, -1, 0) = (Nz-1, Ny-1, 0)
        for periodic BC. See docstring of get_offsets for the ordering of
        neighbor directions.
        The actual data is stored in memory only along the boundary points
        (e.g., i=0, j=32, k=32 for 64^3 mesh), such that the dictionary
        returns the value bpcn when given a key bi; otherwise, it computes
        the neighbor indices on-the-fly, by index+displacements.
    """

    # memory efficient version of pcn, by precomputing only the boundary,
    # computing interior points on the fly.
    Ncells = np.prod(shape)
    # Save on memory when applicable
    if Ncells < 2**31:
        dtype = np.int32
    else:
        dtype = np.int64
    offset = _get_offsets(len(shape), corner)

    # Calculate flattened indices of boundary cells
    bndry_idx = _get_boundary_indices(shape, dtype)
    bndry_idx_3d = np.array(np.unravel_index(bndry_idx, shape), dtype=dtype)

    # shape of bpcn = [N_boundary_cells, N_neighbors(=26 for corner=True)]
    if boundary_flag == 'periodic':
        mode = 'wrap'
    elif boundary_flag == 'outflow':
        mode = 'clip'
    else:
        raise Exception("unknown boundary mode")
    nghbr_idx = bndry_idx_3d[:, :, None] + offset[:, None, :]
    nghbr_idx = np.ravel_multi_index(nghbr_idx, shape, mode=mode).astype(dtype)

    p0 = np.array([1, 1, 1])
    displacements = (np.ravel_multi_index(p0[:,None] + offset, shape, mode='raise'
                                          ).astype(dtype)
                     - np.ravel_multi_index(p0, shape, mode='raise').astype(dtype))

    # Caution: allow out-of-bound index for performance.
    class pcnDict(dict):
        def __getitem__(self, index):
            return self.get(index, index+displacements)
    pcn = pcnDict(zip(bndry_idx, nghbr_idx))
    return pcn


def _get_offsets(dim, corner=True):
    """Compute 1-D flattened array offsets corresponding to neighbors

    Parameters
    ----------
    dim : int
        dimension of the input data.
    corner : Boolean, optional
        If true, the corner cells are counted as neighbors
        (26 neighbors in total).

    Returns
    -------
    offsets : array-like
        The shape of this array is (N_neighbors, dim). Each rows are integer
        directions that points to the neighbor cells. For example, in 3D, the
        offsets will look like:
        (-1, -1, -1)
        (-1, -1,  0)
        (-1, -1,  1)
        (-1,  0, -1)
        (-1,  0,  0)
        (-1,  0,  1)
        (-1,  1, -1)
        (-1,  1,  0)
        (-1,  1,  1)
        ( 0, -1, -1)
        ...
    """
    offs = [-1, 0, 1]
    offsets = list(itertools.product(offs, repeat=dim))
    if corner:
        offsets.remove((0,)*dim)
    else:
        offsets = [i for i in offsets if i.count(0) == 2]
    offsets = np.array(offsets, dtype=np.int32)
    return offsets.T


def _get_boundary_indices(shape, dtype):
    """get boundary indices from shape"""
    shape = list(shape)
    bi = []
    ls = len(shape)
    basel = ls*[None]  # array slice none to extend array dimension
    idx = range(ls)  # index for loops
    # dni is the coords for dimension "i"
    dni = ls*[None]
    for i in idx:
        dni[i] = np.arange(shape[i], dtype=dtype)

    # for boundary dimensions i set indices j != i, setting index i to be
    # 0 or end
    for i in idx:

        ndnis = ls*[None]
        shapei = shape[:]  # copy shape
        shapei[i] = 1      # set dimension i to 1 (flat boundary)
        nzs = np.zeros(shapei, dtype=dtype)  # initialize boundary to 0
        for j in idx:
            if j == i:
                continue
                # make coord j using the np.arange (dni) with nzs of
                # desired shape
            selj = basel[:]
            selj[j] = slice(None)
            # slicing on index j makes dni[j] vary on index j and copy on other
            # dimensions with desired shape nzs
            ndnis[j] = dni[j][tuple(selj)] + nzs
        ndnis[i] = 0
        bi += list(np.ravel_multi_index(ndnis, shape).astype(dtype).flatten())
        ndnis[i] = shape[i]-1
        bi += list(np.ravel_multi_index(ndnis, shape).astype(dtype).flatten())
    return bi
