import jax
import jax.numpy as jnp

def weighted_histogram(counts, indices, weights):
    """
    Returns: counts with every in-range weight scatter-added into its bin
    """
    bins = counts.shape[0]
    valid = (indices >= 0) & (indices < bins)
    safe = jnp.where(valid, indices, bins)
    return counts.at[safe].add(weights.astype(counts.dtype), mode="drop")