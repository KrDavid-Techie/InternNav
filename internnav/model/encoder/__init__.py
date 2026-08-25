"""Encoder exports without importing every optional backend eagerly."""

from importlib import import_module


__all__ = [
    'PositionalEncoding',
    'DistanceNetwork',
    'ImageEncoder',
    'InstructionEncoder',
    'InstructionLongCLIPEncoder',
    'LanguageEncoder',
    'VisionLanguageEncoder',
]

_EXPORTS = {
    'PositionalEncoding': ('.bert_backbone', 'PositionalEncoding'),
    'DistanceNetwork': ('.distance_encoder', 'DistanceNetwork'),
    'ImageEncoder': ('.image_clip_encoder', 'ImageEncoder'),
    'InstructionEncoder': ('.instruction_encoder', 'InstructionEncoder'),
    'InstructionLongCLIPEncoder': ('.instruction_longCLIP_encoder', 'InstructionLongCLIPEncoder'),
    'LanguageEncoder': ('.instruction_roberta_encoder', 'LanguageEncoder'),
    'VisionLanguageEncoder': ('.vision_language_encoder', 'VisionLanguageEncoder'),
}


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from exc

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
