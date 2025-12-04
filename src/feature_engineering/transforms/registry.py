from __future__ import annotations

# registry.py
TRANSFORM_REGISTRY = {}


def register(name):
    def wrapper(fn):
        TRANSFORM_REGISTRY[name] = fn
        return fn

    return wrapper


def get_all_transform():
    return TRANSFORM_REGISTRY
