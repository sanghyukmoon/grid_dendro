"""Implements Dendrogram class"""

import numpy as np
from scipy.ndimage import minimum_filter
from grid_dendro import boundary


class Dendrogram:
    """Dendrogram representing hierarchical structure in 3D data

    Attributes:
        cells_in_node: {node: cells}
          All nodes in dendrogram hierarchy and the cells belong to them.
        cells_in_leaf: {node: cells}
          Leaf nodes and the cells belong to them.
        parent: {cells: node}
          All cells and their parent node. Note that when a cell is a
          node-generating cell and therefore itself a node, its parent is its
          parent node instead of itself.
        child: {node: node}
          All nodes and their child nodes. When two nodes merge to create a new
          node, they are child nodes of the newly created node. Leaf nodes have
          no children.
        descendant: {node: node}
          All nodes and their descendant nodes. When children of a node have
          their own children, all child nodes down to the leaf nodes are the
          descendants of the node.
    """

    def __init__(self, arr, boundary_flag='periodic'):
        """Initializes the instance based on spam preference.

        Args:
          arr: numpy.ndarray instance representing the input data.
          boundary_flag: string representing the boundary condition, optional.
        """
        self.arr = arr
        self.boundary_flag = boundary_flag

    def construct(self):
        """Construct dendrogram to set node, parent, child, and descendant"""
        # Sort flat indices in an ascending order of arr.
        arr_flat = self.arr.flatten()
        cells_ordered = arr_flat.argsort()
        num_cells = len(cells_ordered)

        # Create leaf nodes by finding all local minima.
        if self.boundary_flag == 'periodic':
            filter_mode = 'wrap'
        else:
            msg = f"Boundary flag {self.boundary_flag} is not supported"
            raise ValueError(msg)
        arr_min_filtered = minimum_filter(self.arr, size=3, mode=filter_mode)
        leaf_nodes = np.where(arr_flat == arr_min_filtered.flatten())[0]
        num_leaves = len(leaf_nodes)
        print("Found {} minima".format(num_leaves))

        # Initialize node, parent, child, descendant, and ancestor
        self.cells_in_node = {nd: [nd] for nd in leaf_nodes}
        self.parent = np.full(num_cells, -1, dtype=int)
        self.parent[leaf_nodes] = leaf_nodes
        self.child = {nd: [] for nd in leaf_nodes}
        self.descendant = {nd: [nd] for nd in leaf_nodes}
        # Ancestor of a node is the most distant parent node, up along the
        # dendrogram hierarchy. Ancestor of any given node changes in the
        # course of dendrogram construction, whenever a new node is created.
        ancestor = {nd: nd for nd in leaf_nodes}

        # Load neighbor dictionary.
        my_neighbors = boundary.precompute_neighbor(self.arr.shape,
                                                    self.boundary_flag,
                                                    corner=True)

        # Climb up the potential and construct dendrogram.
        num_remaining_nodes = num_leaves - 1
        for cell in cells_ordered:
            if cell in leaf_nodes:
                continue
            # Find parents of neighboring cells.
            parents = set(self.parent[my_neighbors[cell]])
            parents.discard(-1)

            # Find ancestors of their parents, which can be themselves.
            neighboring_nodes = set([ancestor[prnt] for prnt in parents])
            num_nghbr_nodes = len(neighboring_nodes)

            if num_nghbr_nodes == 0:
                raise ValueError("Should not reach here")
            elif num_nghbr_nodes == 1:
                # Add this cell to the existing node
                nd = neighboring_nodes.pop()
                self.parent[cell] = nd
                self.cells_in_node[nd].append(cell)
            elif num_nghbr_nodes == 2:
                # This cell is at the critical point; create new node.
                self.cells_in_node[cell] = [cell]
                self.parent[cell] = cell
                self.child[cell] = list(neighboring_nodes)
                ancestor[cell] = cell
                self.descendant[cell] = [cell]
                for nd in self.child[cell]:
                    # This node becomes a parent of its immediate children
                    self.parent[nd] = cell
                    # inherit all descendants of children
                    self.descendant[cell] += self.descendant[nd]
                    for nd in self.descendant[nd]:
                        # This node becomes a new ancestor of all its
                        # descendants
                        ancestor[nd] = cell
                num_remaining_nodes -= 1
                msg = ("Added a new node at the critical point. "
                       f"number of remaining nodes = {num_remaining_nodes}")
                print(msg)
                if num_remaining_nodes == 0:
                    print("We have reached the trunk. Stop climbing up")
                    break
            else:
                raise ValueError("Should not reach here")

    def prune(self, ncells_min=27):
        """Prune the buds by applying minimum number of cell criterion"""
        for leaf in self.cells_in_leaf:
            num_cells = len(self.cells_in_leaf[leaf])
            if num_cells < ncells_min:
                # this leaf is a bud.
                my_parent = self.parent[leaf]
                my_grandparent = self.parent[my_parent]
                sibling = self.child[my_parent]
                sibling.remove(leaf)
                sibling = sibling[0]
                if sibling in self.cells_in_leaf and len(self.cells_in_leaf[sibling]) < ncells_min:
                    print("WARNING: sibling is also a bud")

                if (my_parent == my_grandparent):
                    # This is a bud at the trunk. Cut it and define new trunk
                    orphans = (self.cells_in_node[my_parent]
                               + self.cells_in_node[leaf]
                               + self.cells_in_node[sibling])
                    for cell in orphans:
                        self.parent[cell] = -1
                    # sibling becomes the trunk node
                    self.cells_in_node[sibling] = [sibling,]
                    self.parent[sibling] = sibling
                    # Remove orphaned node
                    for nd in [my_parent, leaf]:
                        self.cells_in_node.pop(nd)
                        self.child.pop(nd)
                        self.descendant.pop(nd)
                else:
                    # Reset parent
                    orphans = (self.cells_in_node[my_parent]
                               + self.cells_in_node[leaf])
                    for cell in orphans:
                        self.parent[cell] = sibling
                        self.cells_in_node[sibling].append(cell)
                    self.parent[sibling] = my_grandparent
                    # Reset child
                    self.child[my_grandparent].remove(my_parent)
                    self.child[my_grandparent].append(sibling)
                    # Reset descendant
                    self.descendant[my_grandparent].remove(my_parent)
                    self.descendant[my_grandparent].remove(leaf)
                    # Remove orphaned node
                    for nd in [my_parent, leaf]:
                        self.cells_in_node.pop(nd)
                        self.child.pop(nd)
                        self.descendant.pop(nd)

    def find_leaf(self):
        self.cells_in_leaf = {}
        for nd in self.cells_in_node:
            if self.num_children(nd) == 0:
                self.cells_in_leaf[nd] = self.cells_in_node[nd]

    def num_children(self, nd):
        return len(self.child[nd])

    def check_sanity(self):
        for nd in self.cells_in_node:
            if not (self.num_children(nd) == 2 or self.num_children(nd) == 0):
                raise ValueError("number of children is not 2")
        print("Sane.")
