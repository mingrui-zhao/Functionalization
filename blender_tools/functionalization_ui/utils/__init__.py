"""Utils package for the graph-driven functionalization workflow.

Modules:
    graph_properties  — PropertyGroups + UILists (scene.graph_func_props)
    predicted_graph   — pred/input JSON loading, records, scene import
    install_helpers   — hinge/rail install wrappers over hinge/rail
    handle_placement  — additive + subtractive handle engine
    interior_graph    — pure-numpy compartment detection
    interior_scene    — compartment boxes / panel generation in-scene
    top_processor     — top-panel detection + synthesis engine
"""

from . import graph_properties
from . import predicted_graph
from . import install_helpers
from . import handle_placement
from . import interior_graph
from . import interior_scene
from . import top_processor

__all__ = [
    "graph_properties",
    "predicted_graph",
    "install_helpers",
    "handle_placement",
    "interior_graph",
    "interior_scene",
    "top_processor",
]
