"""
Pruning policy: graph → cutting-point pixel coordinates.

Implements the policy described in the VinPRO final report (Section 5.2):

  1. Find partition nodes — tree junctions where in_degree > 1, i.e. nodes
     from which more than one branch grows toward the tips.
  2. Discard redundant partition nodes: if a partition node P's direct parent
     (successor toward root) is also a partition node, cutting at the parent
     already removes P's subtree, so P is redundant.
  3. Place each cut at the midpoint between the partition node and its parent.
  4. Preserve at least one cane branch (one-year-old shoot carrying next
     season's fruit) by never cutting its connection to the trunk.

Graph conventions from WP2 (vinet/inference/association.py):
  - Nodes : ((x, y), (branch_type_name, node_type_name))
  - Edges : child → parent   (direction toward root)
  - in_degree(n)  = number of branches splitting off toward tips from n
  - out_degree(n) = 1 for all non-root nodes, 0 for root
  - Coordinates are in the 256×256 heatmap space; the caller is responsible
    for scaling to sensor resolution before depth lookup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Branch type that must be preserved (one cane kept for next season).
CANE_BRANCH_TYPE = "cane"


def _find_partition_nodes(tree: nx.DiGraph) -> list:
    """Return all nodes where more than one branch splits toward the tips."""
    return [n for n in tree.nodes if tree.in_degree(n) > 1]


def _parent_of(node, tree: nx.DiGraph):
    """Return the single parent of *node* (the node it points to), or None."""
    successors = list(tree.successors(node))
    return successors[0] if successors else None


def _is_redundant(node, partition_set: set, tree: nx.DiGraph) -> bool:
    """
    A partition node P is redundant when its direct parent is also a partition
    node.  In that case cutting at the parent already removes P's subtree.
    """
    parent = _parent_of(node, tree)
    return parent is not None and parent in partition_set


def _cane_nodes(tree: nx.DiGraph) -> set:
    """Return the set of nodes belonging to the cane branch type."""
    return {n for n in tree.nodes if n[1][0] == CANE_BRANCH_TYPE}


def _subtree_contains_cane(node, tree: nx.DiGraph, cane_nodes: set) -> bool:
    """
    Return True if the subtree rooted at *node* (growing toward the tips,
    i.e. following in-edges) contains any cane node.
    Traversal follows edges in reverse (tip direction).
    """
    visited = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if current in cane_nodes:
            return True
        # predecessors = botanical children (nodes pointing TO current)
        stack.extend(tree.predecessors(current))
    return False


def _midpoint(coord_a: tuple, coord_b: tuple) -> tuple[float, float]:
    """Return the pixel midpoint between two (x, y) coordinates."""
    return ((coord_a[0] + coord_b[0]) / 2.0, (coord_a[1] + coord_b[1]) / 2.0)


def select_cutting_points(
    tree: nx.DiGraph,
) -> list[dict]:
    """
    Run the pruning policy on a WP2 vine tree and return cutting points.

    Args:
        tree: Directed graph produced by ``grapevine_structure_estimation``.
              Nodes are ``((x, y), (branch_type, node_type))``; edges go
              child → parent (toward root).

    Returns:
        List of dicts, one per cut::

            {
                "pixel":     (float, float),   # (x, y) in 256×256 heatmap space
                "node":      tree node tuple,  # the partition node
                "parent":    tree node tuple,  # its parent in the tree
                "branch_dir": np.ndarray | None,  # unit direction from node to parent
            }

        Empty list when no pruning cuts are needed.
    """
    if tree.number_of_nodes() == 0:
        logger.warning("Empty vine tree — no cuts generated")
        return []

    # ── Step 1: find all partition nodes ──────────────────────────────────────
    partition_nodes = _find_partition_nodes(tree)
    if not partition_nodes:
        logger.info("No partition nodes detected; vine may be too simple to prune")
        return []

    partition_set = set(partition_nodes)
    logger.debug("Partition nodes found: %d", len(partition_nodes))

    # ── Step 2: discard redundant partition nodes ──────────────────────────────
    active = [p for p in partition_nodes if not _is_redundant(p, partition_set, tree)]
    logger.debug("Active partition nodes after redundancy removal: %d", len(active))

    # ── Step 3: identify cane subtrees to preserve ────────────────────────────
    cane_nodes = _cane_nodes(tree)
    cane_preserved = False  # track whether we have already preserved one cane

    cuts: list[dict] = []
    for partition_node in active:
        parent = _parent_of(partition_node, tree)
        if parent is None:
            # Partition node is the root — cannot cut above root
            continue

        # ── Step 4: preserve one cane branch ──────────────────────────────────
        # If the subtree below this partition node contains a cane AND we have
        # not yet preserved a cane, skip this cut to keep the cane intact.
        if not cane_preserved and _subtree_contains_cane(partition_node, tree, cane_nodes):
            cane_preserved = True
            logger.info(
                "Cane preserved at partition node %s — skipping this cut",
                partition_node[0],
            )
            continue

        # ── Compute midpoint cut position ──────────────────────────────────────
        node_xy = np.array(partition_node[0], dtype=float)
        parent_xy = np.array(parent[0], dtype=float)

        mid = _midpoint(partition_node[0], parent[0])

        # Branch direction unit vector (from partition node toward parent/root).
        delta = parent_xy - node_xy
        dist = np.linalg.norm(delta)
        branch_dir = (delta / dist) if dist > 1e-9 else None

        cuts.append(
            {
                "pixel": mid,
                "node": partition_node,
                "parent": parent,
                "branch_dir": branch_dir,
            }
        )

    logger.info("Pruning policy produced %d cutting points", len(cuts))
    return cuts
