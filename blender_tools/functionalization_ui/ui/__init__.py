"""UI package — panel stack + viewport overlay."""

from . import graph_panel
from . import overlay

__all__ = ["graph_panel", "overlay"]


def register():
    graph_panel.register()


def unregister():
    overlay.disable()
    graph_panel.unregister()
