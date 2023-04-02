"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Terminology:
    index (indices): flattend array index of each cell.
    seed: The index of the cell that has a critical value of the input array,
          above and below which the nodes merge or split.
    flesh: indices of all cells that belong to a node, including the seed.
    node: Any structure in dendrogram hierarchy (e.g., leaf, branch, trunk).
          Stored in dictionary structure such that nodes[seed] = flesh.
     ___________
    |xxxxxxxxxxx|
    |xxxxx xxxxx|    . seed
    |xxxx . xxxx|    x flesh
    |xxxxx xxxxx|
    |xxxxxxxxxxx|
     -----------
"""

import numpy as np
from scipy.ndimage import minimum_filter
from fhiso import boundary


def construct_dendrogram(arr, boundary_flag='periodic'):
    """Construct isocontour tree

    Args:
        arr: numpy.ndarray instance representing the input data.
        boundary_flag: string representing the boundary condition, optional.

    Returns:
        nodes: dictionary in which each key represents the flat index of the
          seed cell, and the corresponding values contain the flat indices of
          flesh cells of this node.
        child_nodes: dictionary containing child seeds of each seed (not recursive).
    """
    # sort flat indices in an ascending order of arr
    arr_flat = arr.flatten()
    indices_ordered = arr_flat.argsort()
    num_cells = len(indices_ordered)

    # Create leaf nodes by finding all local minima.
    # Note that local minima are seeds of the leaf nodes.
    if boundary_flag == 'periodic':
        filter_mode = 'wrap'
    else:
        raise ValueError(f"Boundary flag {boundary_flag} is not supported")
    arr_min_filtered = minimum_filter(arr, size=3, mode=filter_mode)
    leaf_nodes = np.where(arr_flat == arr_min_filtered.flatten())[0]
    num_leaves = len(leaf_nodes)
    nodes = {idx: [idx,] for idx in leaf_nodes}
    child_nodes = {idx: set() for idx in leaf_nodes}
    print("Found {} minima".format(num_leaves))

    # Create my_seed list and add seeds of themselves.
    # my_seed is indexed by flattened indices. Note that my_seed is not sorted
    # cells with my_seed = -1 has not been added to any nodes.
    my_seed = np.full(num_cells, -1, dtype=int)
    my_seed[leaf_nodes] = leaf_nodes

    # Load neighbor dictionary
    my_neighbors = boundary.precompute_neighbor(arr.shape, boundary_flag,
                                                corner=True)

    # Climb up the potential and construct nodes and their children
    nmerge = 0
    for idx in indices_ordered:
        if idx in leaf_nodes:
            continue
        ngb_seeds = set(my_seed[my_neighbors[idx]])
        ngb_seeds.discard(-1)
        num_ngb_seeds = len(ngb_seeds)
        if num_ngb_seeds == 0:
            raise ValueError("Should not reach here")
        elif num_ngb_seeds == 1:
            # This cell is a flesh of an existing node.
            seed = ngb_seeds.pop()
            my_seed[idx] = seed
            nodes[seed].append(idx)
        elif num_ngb_seeds == 2:
            # This cell is at the critical point, thus becomes a new seed;
            # create new node.
            my_seed[idx] = idx
            nodes[idx] = [idx,]
            child_nodes[idx] = ngb_seeds
            flesh = recursive_members(nodes, child_nodes, idx)
            my_seed[flesh] = idx
            nmerge += 1
            num_remaining_nodes = num_leaves - 1 - nmerge
            print("Reaching critical point. number of remaining nodes = {}"
                  .format(num_remaining_nodes))
            if num_remaining_nodes == 0:
                print("We have reached the trunk. Stop climbing up")
                break
    return nodes, child_nodes


def recursive_members(nodes, child_nodes, idx):
    flesh = []
    flesh += nodes[idx]
    for child in child_nodes[idx]:
        flesh += recursive_members(nodes, child_nodes, child)
    return flesh
