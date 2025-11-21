import numpy as np
import re
from ete3 import Tree

import warnings
warnings.simplefilter('always', UserWarning)


# Tree initialization: set the node name to the sequence order
def init(tree, branch=None, name='all', scale=0.1, display=False, return_map=False):
    if return_map: idx2node = {}
    i, j = 0, len(tree)
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            if name != 'interior':
                node.name, i = i, i + 1
            else:
                node.name = int(node.name)
        else:
            node.name, j = j, j + 1
        if not node.is_root():
            if isinstance(branch, str) and branch == 'random':
                node.dist = np.random.exponential(scale)
            elif branch is not None:
                node.dist = branch[node.name]
        else:
            node.dist = 0.0

        if return_map: idx2node[node.name] = node
        if display:
            print(node.name, node.dist)

    if return_map: return idx2node


def create(ntips, branch='random', scale=0.1):
    tree = Tree()
    tree.populate(ntips)
    tree.unroot()
    init(tree, branch=branch, scale=scale)

    return tree


def namenum(tree, taxon, nodetosplitMap=None):
    taxon2idx = {}
    j = len(taxon)
    if nodetosplitMap:
        idx2split = [''] * (2 * j - 3)
    for i, name in enumerate(taxon):
        taxon2idx[name] = i
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            # assert type(node.name) is str, "The taxon name should be strings"
            if not isinstance(node.name, str):
                warnings.warn("The taxon names are not strings, please check if they are already integers!")
            else:
                node.name = taxon2idx[node.name]
                if nodetosplitMap:
                    idx2split[node.name] = nodetosplitMap[node]
        else:
            node.name, j = j, j + 1
            if nodetosplitMap and not node.is_root():
                idx2split[node.name] = nodetosplitMap[node]

    if nodetosplitMap:
        return idx2split


def nametaxon(tree, taxon):
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            # assert type(node.name) is int, "The taxon name should be integers"
            if not isinstance(node.name, int):
                warnings.warn("The taxon names are not integers, please check if they are already strings!")
            else:
                node.name = taxon[node.name]


def is_internal(node):
    return not node.is_leaf() and not node.is_root()


# obtain a dictionary of nodes
def idx2nodeMAP(tree):
    idx2node = {}
    for node in tree.traverse("postorder"):
        idx2node[node.name] = node
    return idx2node


def NNI(node, include=False):
    if node.is_root() or node.is_leaf():
        print("Can not perform NNI on root or leaf branches!")
    else:
        if include:
            neighboor = np.random.randint(3)
            xchild = node.get_sisters()[0]
            parent = node.up
            if neighboor == 1:
                ychild = node.children[0]
            if neighboor == 2:
                ychild = node.children[1]
            if neighboor:
                xchild.detach()
                ychild.detach()
                parent.add_child(ychild)
                node.add_child(xchild)
                return 1
        else:
            neighboor = np.random.randint(2)
            xchild = node.get_sisters()[0]
            parent = node.up
            ychild = node.children[neighboor]
            xchild.detach()
            ychild.detach()
            parent.add_child(ychild)
            node.add_child(xchild)
            return 1

    return 0


def copy(tree, branch=None):
    copy_tree = tree.copy('newick')
    for node in copy_tree.traverse('postorder'):
        node.name = int(node.name)
        if not node.is_root() and branch:
            node.dist = branch[node.name]

    return copy_tree

def read_rooted_fossil_tree(fossil_tree_path):
    pattern = re.compile(r"""
        B          # Literal 'B'
        \(         # An opening parenthesis
        ([^,]+)    # Capture group 1: first float or value (anything that's not a comma)
        ,([^)]+)   # Capture group 2: second float or value (until closing parenthesis)
        \)         # Closing parenthesis
    """, re.VERBOSE)


    with open(fossil_tree_path, 'r') as f:
        fossil_tree = f.read()
    # print(fossil_tree)

    tree = Tree(fossil_tree, format=1, quoted_node_names=True)
    for node in tree.traverse('postorder'):
        if node.name.startswith('B('):
            match = re.match(pattern, node.name)
            lower_bound, upper_bound = match.groups()
            node.add_feature("lb", float(lower_bound))
            node.add_feature("ub", float(upper_bound))

        else:
            node.add_feature("lb", None)
            node.add_feature("ub", None)
        # if node.lb:
        #     print(node.name)
        #     print(node.lb)
        #     print(node.ub)
    # taxa = tree.get_leaf_names()
    # print(taxa)
    # namenum(tree, taxa)
    # print(tree)
    return tree




def fossil_bounds_to_branch_lengths(tree, root_age=None, midpoint=True, eps=1e-9):

    for node in tree.traverse("postorder"):
        if hasattr(node, "lb") and node.lb is not None:        # fossil-calibrated node
            if midpoint:
                node.height = 0.5 * (node.lb + node.ub)
            else:
                node.height = node.lb
        elif node.is_leaf():                                   # extant tip
            node.height = 0.0
        else:
            node.height = None                                 # will be filled in pass 2
    changed = True
    while changed:
        changed = False
        for node in tree.traverse("postorder"):
            if node.height is None:
                # only set if *all* children already have heights
                if all(hasattr(c, "height") and c.height is not None for c in node.children):
                    node.height = max(c.height for c in node.children) + eps
                    changed = True

    for node in tree.traverse():
        if node.height is None:
            raise RuntimeError(f"Height still missing for node {node.name}")

    if root_age is not None:
        factor = root_age / tree.get_tree_root().height
        for node in tree.traverse():
            node.height *= factor

    for node in tree.traverse("postorder"):
        if node.is_root():
            node.dist = 0.0                       # root branch length is undefined; keep 0
        else:
            node.dist = node.up.height - node.height
            if node.dist < 0:
                raise ValueError(f"Negative branch length at {node.name} ({node.dist})")

    return tree



if __name__ == '__main__':
    tree = read_rooted_fossil_tree("./data/cali_old.nwk")
    # fossil_bounds_to_branch_lengths(tree, root_age=3)
    print(tree)
    N_internal_node = 0
    taxa = tree.get_leaf_names()
    for node in tree.traverse("preorder"):
        if not node.is_leaf() and not node.is_root():
            node.dist = 0.2
            N_internal_node += 1
        if node.is_root():
            node.dist = 3.0
        if node.is_leaf():
            node.dist = 3 - 0.2*N_internal_node
    namenum(tree, taxa)
    print(tree)
    for node in tree.traverse("postorder"):
        print(node.name,node.dist)