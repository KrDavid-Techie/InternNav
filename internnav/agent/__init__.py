"""Agent public API.

Keep optional agent implementations lazy so lightweight deployment entrypoints
do not import every simulator/model dependency just to load one agent.
"""

from importlib import import_module

__all__ = [
    "Agent",
    "CmaAgent",
    "DialogAgent",
    "InternVLAN1Agent",
    "RdpAgent",
    "Seq2SeqAgent",
]

_LAZY_AGENTS = {
    "Agent": ("internnav.agent.base", "Agent"),
    "CmaAgent": ("internnav.agent.cma_agent", "CmaAgent"),
    "DialogAgent": ("internnav.agent.dialog_agent", "DialogAgent"),
    "InternVLAN1Agent": ("internnav.agent.internvla_n1_agent", "InternVLAN1Agent"),
    "RdpAgent": ("internnav.agent.rdp_agent", "RdpAgent"),
    "Seq2SeqAgent": ("internnav.agent.seq2seq_agent", "Seq2SeqAgent"),
}


def __getattr__(name):
    if name not in _LAZY_AGENTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_AGENTS[name]
    attribute = getattr(import_module(module_name), attribute_name)
    globals()[name] = attribute
    return attribute
