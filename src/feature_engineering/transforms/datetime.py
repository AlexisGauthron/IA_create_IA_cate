# datetime_transforms.py
from src.features_engineering.transformation_fe.registry import register

@register("year")
def year(date):
    return date.dt.year

@register("month")
def month(date):
    return date.dt.month

@register("day")
def day(date):
    return date.dt.day

@register("dayofweek")
def dayofweek(date):
    return date.dt.dayofweek
