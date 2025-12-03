# numeric_transforms.py
import numpy as np
import pandas as pd
from src.features_engineering.transformation_fe.registry import register

@register("standard")
def standard(x):
    return (x - x.mean()) / x.std()

@register("robust")
def robust(x):
    return (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25))

@register("impute_mean")
def impute_mean(x):
    return x.fillna(x.mean())

@register("impute_median")
def impute_median(x):
    if pd.api.types.is_numeric_dtype(x):
        return x.fillna(x.median())
    else:
        # fallback pour catégoriel
        return x.fillna(x.mode()[0] if not x.mode().empty else None)

@register("log1p")
def log1p(x):
    return np.log1p(x)

@register("sqrt")
def sqrt(x):
    return np.sqrt(x)

@register("square")
def square(x):
    return x**2

@register("cube")
def cube(x):
    return x**3

@register("clip")
def clip_fn(x, q01, q99):
    return x.clip(q01, q99)

@register("sum")
def sum_fn(x, y):
    return x + y

@register("diff")
def diff(x, y):
    return x - y

@register("prod")
def prod(x, y):
    return x * y

@register("ratio")
def ratio(x, y):
    return x / (y + 1e-6)

@register("min")
def min_fn(x, y):
    return pd.concat([x, y], axis=1).min(axis=1)

@register("max")
def max_fn(x, y):
    return pd.concat([x, y], axis=1).max(axis=1)


@register("bin_quantile")
def bin_quantile(col, q):
    """
    Découpe une variable numérique en quantiles.
    
    - col : pd.Series numérique
    - q   : nombre de quantiles (default 5)
    
    Retourne une série d'entiers entre 0 et q-1.
    """
    if not isinstance(col, pd.Series):
        raise TypeError(f"bin_quantile requires a pandas Series, got {type(col)}")

    if not isinstance(q, int) or q <= 1:
        raise ValueError("q must be an integer >= 2")

    # Quantile binning
    binned = pd.qcut(col, q, labels=False, duplicates="drop")

    return binned