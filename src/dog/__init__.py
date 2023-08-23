TORCH_AVAILABLE = False
JAX_AVAILABLE = False

try:
    import torch
    from .torch import DoG, LDoG, PolynomialDecayAverager
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import jax
    from .jax import DoG as DoGJAX, LDoG as LDoGJAX, polynomial_decay_averaging, get_av_model
    JAX_AVAILABLE = True
except ImportError:
    pass

if not TORCH_AVAILABLE and not JAX_AVAILABLE:
    raise ImportError("Either PyTorch or JAX should be installed")
