"""
Tree structure association via resistivity graph and shortest paths (Section 2.3.4).

Pipeline:
    1. Build a directed graph where edges connect each node to compatible parents
    2. Compute edge resistivity based on alignment with predicted vector fields (Eq. 4-6)
    3. Run Dijkstra's shortest path from each node to the root crown
    4. The union of shortest paths defines the tree structure

This guarantees a valid tree output and handles unknown numbers of branches/nodes.
"""

import numpy as np
import networkx as nx
from scipy.spatial import distance

from ..config import BRANCH_TYPES, POSSIBLE_PARENTS, DEFAULT_ASSOCIATION_RADIUS


def calculate_resistivity(
    child: tuple,
    parent: tuple,
    vector_field: np.ndarray,
    n_samples: int = 100,
) -> float:
    """
    Calculate resistivity between two nodes based on vector field alignment.

    Implements Eq. 4-6 from the paper:
        A = integral of V_t(p(u)) · v_cp/||v_cp|| du
        R_e = (1 - A) * ||v_cp||

    Args:
        child: (x, y) coordinates of the child node.
        parent: (x, y) coordinates of the parent node.
        vector_field: Predicted vector field, shape (2, H, W).
        n_samples: Number of integration samples along the line segment.

    Returns:
        Resistivity value (lower = better alignment = more likely connection).
    """
    v_cp = np.array(child) - np.array(parent)
    v_cp_len = np.linalg.norm(v_cp)
    if v_cp_len < 1e-15:
        return 0.0
    v_cp_norm = v_cp / v_cp_len

    # Sample points along the child-parent line (Eq. 5)
    samples = np.linspace(0, 1, n_samples)
    alignments = []
    for u in samples:
        point = (1 - u) * np.array(child) + u * np.array(parent)
        y, x = int(round(point[1])), int(round(point[0]))
        if 0 <= y < vector_field.shape[1] and 0 <= x < vector_field.shape[2]:
            v_field = vector_field[:, y, x]
            alignments.append(np.dot(v_cp_norm, v_field))

    if not alignments:
        return v_cp_len  # Maximum resistivity if no valid samples

    A = np.mean(alignments)  # Alignment score (Eq. 4)
    resistivity = (1 - A) * v_cp_len  # Eq. 6
    return resistivity


def construct_resistivity_graph(
    total_nodes_extracted: dict,
    branch_types: dict = None,
    vector_fields: np.ndarray = None,
    possible_parents: dict = None,
    radius: float = DEFAULT_ASSOCIATION_RADIUS,
) -> nx.DiGraph:
    """
    Build a directed resistivity graph from detected nodes and vector fields.

    For each node, edges are created to compatible parent nodes within
    the given radius. If no compatible parent is within radius, the closest
    compatible node is used as a fallback (Section 2.3.4).

    Args:
        total_nodes_extracted: Dict mapping (branch_type, node_type) → list of (x, y).
        branch_types: Branch type → index mapping.
        vector_fields: Predicted vector fields, shape (N_branch_types, 2, H, W).
        possible_parents: Child type → list of valid parent types.
        radius: Maximum search radius for parent candidates (default: 0.2 * W_h).

    Returns:
        Directed graph with resistivity-weighted edges.
    """
    if branch_types is None:
        branch_types = BRANCH_TYPES
    if possible_parents is None:
        possible_parents = POSSIBLE_PARENTS

    G = nx.DiGraph()

    # Flatten all detected nodes into a set of (coords, (branch_type, node_type))
    all_nodes = set()
    for node_type, nodes in total_nodes_extracted.items():
        for node in nodes:
            all_nodes.add((node, node_type))

    # Add nodes to graph
    for node, node_type in all_nodes:
        G.add_node((node, node_type))

    # Add edges with resistivity weights
    for child, child_type in all_nodes:
        parents_found = False
        for parent, parent_type in all_nodes:
            if child == parent:
                continue
            if parent_type[0] not in possible_parents.get(child_type[0], []):
                continue

            dist = distance.euclidean(child, parent)
            if dist <= radius:
                r = calculate_resistivity(
                    child, parent,
                    vector_fields[branch_types[parent_type[0]]],
                )
                G.add_edge((child, child_type), (parent, parent_type), weight=r)
                parents_found = True

        # Fallback: use closest compatible node if none found within radius
        if not parents_found:
            compatible = [
                (p, pt)
                for p, pt in all_nodes
                if pt[0] in possible_parents.get(child_type[0], []) and p != child
            ]
            if compatible:
                closest = min(compatible, key=lambda x: distance.euclidean(child, x[0]))
                r = calculate_resistivity(
                    child, closest[0],
                    vector_fields[branch_types[closest[1][0]]],
                )
                G.add_edge((child, child_type), closest, weight=r)

    return G


def grapevine_structure_estimation(
    graph: nx.DiGraph,
    root_node: tuple,
) -> nx.DiGraph:
    """
    Estimate the tree structure using shortest paths from all nodes to the root.

    Runs Dijkstra's algorithm from each node to the root crown.
    The union of all shortest paths forms the estimated tree (Section 2.3.4).

    Args:
        graph: Resistivity graph with weighted edges.
        root_node: The root crown node as (coords, (branch_type, node_type)).

    Returns:
        Tree structure as a directed graph with resistivity weights on edges.
    """
    tree = nx.DiGraph()

    for node in graph.nodes:
        if node == root_node:
            continue
        try:
            path = nx.shortest_path(graph, source=node, target=root_node, weight="weight")
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                weight = graph[child][parent]["weight"]
                tree.add_edge(child, parent, weight=weight)
        except nx.NetworkXNoPath:
            pass  # Skip disconnected nodes

    return tree
