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
from gridden import boundary


class Dendrogram:
    """Dendrogram representing hierarchical structure in 3D data

    Attributes:
    """

    def __init__(self, arr, boundary_flag = 'periodic'):
        """Initializes the instance based on spam preference.

        Args:
          arr: numpy.ndarray instance representing the input data.
          boundary_flag: string representing the boundary condition, optional.
        """
        self.arr = arr
        self.boundary_flag = boundary_flag

    def construct(self):
        """Construct dendrogram

        Returns:
          node: dictionary containing {node index: list(member cells)}
          child: dictionary containing {node index: set(child nodes)}
          parent: numpy.ndarray containing {index: parent node index}
          descendant: dictionary containing {node index: descendant node indices}
        """
        # Sort flat indices in an ascending order of arr.
        arr_flat = self.arr.flatten()
        indices_ordered = arr_flat.argsort()
        num_cells = len(indices_ordered)

        # Create leaf nodes by finding all local minima.
        if self.boundary_flag == 'periodic':
            filter_mode = 'wrap'
        else:
            raise ValueError(f"Boundary flag {self.boundary_flag} is not supported")
        arr_min_filtered = minimum_filter(self.arr, size=3, mode=filter_mode)
        leaf_nodes = np.where(arr_flat == arr_min_filtered.flatten())[0]
        num_leaves = len(leaf_nodes)
        print("Found {} minima".format(num_leaves))

        # Initialize node, parent, child, ancestor, and descendant.
        self.node = {nd: [nd] for nd in leaf_nodes}
        self.parent = np.full(num_cells, -1, dtype=int)
        self.parent[leaf_nodes] = leaf_nodes
        self.child = {nd: [] for nd in leaf_nodes}
        self.descendant = {nd: [nd] for nd in leaf_nodes}
        ancestor = {nd: nd for nd in leaf_nodes}

        # Load neighbor dictionary.
        my_neighbors = boundary.precompute_neighbor(self.arr.shape, self.boundary_flag,
                                                    corner=True)

        # Climb up the potential and construct dendrogram.
        num_remaining_nodes = num_leaves - 1
        for idx in iter(indices_ordered):
            if idx in leaf_nodes:
                continue
            # Find parents of neighboring cells.
            parents = set(self.parent[my_neighbors[idx]])
            parents.discard(-1)

            # Find ancestors of their parents, which can be themselves.
            neighboring_nodes = set([ancestor[prnt] for prnt in parents])
            num_nghbr_nodes = len(neighboring_nodes)

            if num_nghbr_nodes == 0:
                raise ValueError("Should not reach here")
            elif num_nghbr_nodes == 1:
                # Add this cell to the existing node
                nd = neighboring_nodes.pop()
                self.parent[idx] = nd
                self.node[nd].append(idx)
            elif num_nghbr_nodes == 2:
                # This cell is at the critical point; create new node.
                self.node[idx] = [idx]
                self.parent[idx] = idx
                self.child[idx] = list(neighboring_nodes)
                ancestor[idx] = idx
                self.descendant[idx] = [idx]
                for nd in self.child[idx]:
                    # This node becomes a parent of its immediate children
                    self.parent[nd] = idx
                    # inherit all descendants of children
                    self.descendant[idx] += self.descendant[nd]
                    for nd in self.descendant[nd]:
                        # This node becomes a new ancestor of all its descendants
                        ancestor[nd] = idx
                num_remaining_nodes -= 1
                msg = ("Added a new node at the critical point. "
                       f"number of remaining nodes = {num_remaining_nodes}")
                print(msg)
                if num_remaining_nodes == 0:
                    print("We have reached the trunk. Stop climbing up")
                    break
            else:
                raise ValueError("Should not reach here")
