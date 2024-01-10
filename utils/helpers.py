import functools
from inspect import isfunction
from einops import rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# def compute_per_image(func, x, y):
#     assert x.shape == y.shape
#     if x.dim() == 4:
#         assert x.size(1) == 3 or x.size(1) == 1, f"Image Channel Error"
#         pass
#     elif x.dim() == 5:
#         x, y = map(lambda t: rearrange(t, 'b, c, t, h, w -> (b t), c, h, w'), (x, y))
#     return func(x, y)


def compute_per_image(func):
    @functools.wraps(func)
    def wrapper(x, y):
        assert x.shape == y.shape, f"{x.shape}, {y.shape}"
        if x.dim() == 4:
            assert x.size(1) == 3 or x.size(1) == 1, f"Image Channel Error"
        elif x.dim() == 5:
            x, y = map(lambda t: rearrange(t, 'b c t h w -> (b t) c h w'), (x, y))
        return func(x, y)

    return wrapper
