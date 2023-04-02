"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Terminology:
    index (indices): flattend array index of each cell.
    node: Any structure in dendrogram hierarchy (e.g., leaf, branch, trunk).
          Stored in dictionary structure.
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
        nodes: dictionary containing {node: list(member cells)}
        child_nodes: dictionary containing {node: set(child nodes)}
    """
    # sort flat indices in an ascending order of arr
    arr_flat = arr.flatten()
    indices_ordered = arr_flat.argsort()
    num_cells = len(indices_ordered)

    # Create leaf nodes by finding all local minima.
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

    # initially, all the cells have no parent
    parent_node = np.full(num_cells, -1, dtype=int)
    parent_node[leaf_nodes] = leaf_nodes
    descendant_node = {idx: [idx,] for idx in leaf_nodes}
    ancestor_node = {idx: idx for idx in leaf_nodes}

    # Load neighbor dictionary
    my_neighbors = boundary.precompute_neighbor(arr.shape, boundary_flag,
                                                corner=True)

    # Climb up the potential and construct nodes and their children
    nmerge = 0
    for idx in indices_ordered:
        if idx in leaf_nodes:
            continue
        neighbors = my_neighbors[idx]
        mask = parent_node[neighbors] != -1
        neighbors = neighbors[mask]
        parents = parent_node[neighbors]
        ancestors = set([ancestor_node[prnt] for prnt in parents])
        num_parents = len(ancestors)
        if num_parents == 0:
            raise ValueError("Should not reach here")
        elif num_parents == 1:
            # This cell is a member of an existing node.
            prnt = ancestors.pop()
            parent_node[idx] = prnt
            nodes[prnt].append(idx)
        elif num_parents == 2:
            # This cell is at the critical point, thus becomes a new parent node;
            nodes[idx] = [idx,]
            parent_node[idx] = idx
            child_nodes[idx] = ancestors
            ancestor_node[idx] = idx
            descendant_node[idx] = [idx,]
            for child in child_nodes[idx]:
                descendant_node[idx] += descendant_node[child]
                for descendant in descendant_node[child]:
                    ancestor_node[descendant] = idx
            nmerge += 1
            num_remaining_nodes = num_leaves - 1 - nmerge
            print("Reaching critical point. number of remaining nodes = {}"
                  .format(num_remaining_nodes))
            if num_remaining_nodes == 0:
                print("We have reached the trunk. Stop climbing up")
                break
        else:
            raise ValueError("This cell have more than two neighboring parent.")
    return nodes, child_nodes
