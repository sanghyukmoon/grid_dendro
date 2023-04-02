"""Constructs dendrogram.

Terminology:
    node: Any structure in dendrogram hierarchy (e.g., leaf, branch, trunk).
    parent: The node that a cell belongs to is its parent. Also, when two nodes
            merge to create a new node, it is a parent node of the two merging
            nodes.
    child: When two nodes merge to create a new node, they are child nodes of
           the newly created node. Leaf nodes have no children.
    ancestor: Ancestor of a node is the most distant parent node, up along the
              dendrogram hierarchy. Ancestor of any given node changes in the
              course of dendrogram construction, whenever a new node is
              created.
    descendant: When children of a node have their own children, and so on and
                so forth, all child nodes down to the leaf nodes are
                the descendants of the node.
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
        node: dictionary containing {node: list(member cells)}
        child_node: dictionary containing {node: set(child nodes)}
    """
    # Sort flat indices in an ascending order of arr.
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
    print("Found {} minima".format(num_leaves))

    # Initialize node, parent, child, ancestor, and descendant.
    node = {nd: [nd] for nd in leaf_nodes}
    parent_node = np.full(num_cells, -1, dtype=int)
    parent_node[leaf_nodes] = leaf_nodes
    child_node = {nd: [] for nd in leaf_nodes}
    ancestor_node = {nd: nd for nd in leaf_nodes}
    descendant_node = {nd: [nd] for nd in leaf_nodes}

    # Load neighbor dictionary.
    my_neighbors = boundary.precompute_neighbor(arr.shape, boundary_flag,
                                                corner=True)

    # Climb up the potential and construct dendrogram.
    num_remaining_nodes = num_leaves - 1
    for idx in iter(indices_ordered):
        if idx in leaf_nodes:
            continue
        # Find parents of neighboring cells.
        parents = set(parent_node[my_neighbors[idx]])
        parents.discard(-1)

        # Find ancestors of their parents, which can be themselves.
        ancestors = set([ancestor_node[prnt] for prnt in parents])
        num_ancestors = len(ancestors)

        if num_ancestors == 0:
            raise ValueError("Should not reach here")
        elif num_ancestors == 1:
            # Add this cell to the existing node
            nd = ancestors.pop()
            parent_node[idx] = nd
            node[nd].append(idx)
        elif num_ancestors == 2:
            # This cell is at the critical point; create new node.
            node[idx] = [idx]
            parent_node[idx] = idx
            child_node[idx] = list(ancestors)
            ancestor_node[idx] = idx
            descendant_node[idx] = [idx]
            for child in child_node[idx]:
                # This node becomes a parent of its immediate children
                parent_node[child] = idx
                # inherit all descendants of children
                descendant_node[idx] += descendant_node[child]
                for descendant in descendant_node[child]:
                    # This node becomes a new ancestor of all its descendants
                    ancestor_node[descendant] = idx
            num_remaining_nodes -= 1
            msg = ("Added a new node at the critical point. "
                   f"number of remaining nodes = {num_remaining_nodes}")
            print(msg)
            if num_remaining_nodes == 0:
                print("We have reached the trunk. Stop climbing up")
                break
        else:
            raise ValueError("Should not reach here")
    return node, child_node, parent_node, descendant_node
