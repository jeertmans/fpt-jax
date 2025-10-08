# Fermat path-tracing with JAX

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)

`fpt-jax` is a standalone library for differentiable path-tracing using the Fermat principle, implemented with JAX.

## Installation

We do not upload releases to PyPI, but you can install the library directly from the GitHub repository:

```bash
pip install git+https://github.com/jeertmans/fpt-jax.git
```

If you want to install from a specific branch, tag, or commit, append `@<branch|tag|commit>` to the URL.

## Usage

This library implements a single function, `trace_rays`, which traces rays undergoing specular reflections and diffractions on planar objects defined by origins and basis vectors:

```
> from fpt_jax import trace_rays; help(trace_rays)
```

**trace_rays**<br>
&nbsp;&nbsp;&nbsp;`(tx: jax.Array, rx: jax.Array,`<br>
&nbsp;&nbsp;&nbsp;&nbsp;`object_origins: jax.Array, object_vectors: jax.Array, *,`<br>
&nbsp;&nbsp;&nbsp;&nbsp;`num_iters: int, unroll: int | bool = 1,`<br>
&nbsp;&nbsp;&nbsp;&nbsp;`num_iters_linesearch: int = 1, unroll_linesearch: int | bool = 1) -> jax.Array:`<br>

Compute the points of interaction of rays with objects using Fermat's principle.

Each ray is obtained by minimizing the total travel distance from transmitter to receiver,
using a quasi-Newton optimization algorithm (BFGS). At each iteration, a line search is performed
to find the optimal step size along the descent direction.

This function accepts batched inputs, where the leading dimensions must be broadcast-compatible.

**Args:**<br>
&nbsp;&nbsp;&nbsp;&nbsp;`tx`: Transmitter positions of shape `(..., 3)`.<br>
&nbsp;&nbsp;&nbsp;&nbsp;`rx`: Receiver positions of shape `(..., 3)`.<br>
&nbsp;&nbsp;&nbsp;&nbsp;`object_origins`: Origins of the objects of shape `(..., num_interactions, 3)`.<br>
&nbsp;&nbsp;&nbsp;&nbsp;`object_vectors`: Vectors defining the objects of shape `(..., num_interactions, num_dims, 3)`.<br>
&nbsp;&nbsp;&nbsp;&nbsp;`num_iters`: Number of iterations for the optimization algorithm.<br>
&nbsp;&nbsp;&nbsp;&nbsp;`unroll`: If an integer, the number of optimization iterations to unroll in the JAX [`scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html).<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If `True`, unroll all iterations. If `False`, do not unroll.<br>
&nbsp;&nbsp;&nbsp;&nbsp;`num_iters_linesearch`: Number of iterations for the line search fixed-point iteration.<br>
&nbsp;&nbsp;&nbsp;&nbsp;`unroll_linesearch`: If an integer, the number of fixed-point iterations to unroll in the JAX [`scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html).<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If `True`, unroll all iterations. If `False`, do not unroll.<br>
&nbsp;&nbsp;&nbsp;&nbsp;`implicit_diff`: Whether to use implicit differentiation for computing the gradient.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If `True`, assumes that the solution has converged and applies the implicit function theorem to differentiate the optimization problem with respect to the input parameters: `tx`, `rx`, `object_origins`, and `object_vectors`.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If `False`, the gradient is computed by backpropagating through all iterations of the optimization algorithm.<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Using implicit differentiation is more memory- and computationally efficient, as it does not require storing intermediate values from all iterations, but it may be less accurate if the optimization has not fully converged. Moreover, implicit differentiation is not compatible with forward-mode autodiff in JAX.<br>

**Returns:**<br>
&nbsp;&nbsp;&nbsp;&nbsp;The points of interaction of shape `(..., num_interactions, 3)`.<br>
&nbsp;&nbsp;&nbsp;&nbsp;To include the transmitter and receiver positions, concatenate `tx` and `rx` to the result.<br>

---

This algorithm is also available within [DiffeRT](https://github.com/jeertmans/DiffeRT), our differentiable ray tracing library for radio propagation.

## Getting help

For any question about the method or its implementation, make sure to first read the related [paper](https://arxiv.org/abs/2306.14822).

If you want to report a bug in this library or the underlying algorithm, please open an issue on this [GitHub repository](https://github.com/jeertmans/fpt-jax/issues). If you want to request a new feature, please consider opening an issue on [DiffeRT's GitHub repository](https://github.com/jeertmans/DiffeRT) instead.

## Citing

If you use this library in your research, please cite our paper:

```bibtex
@misc{eertmans2025fpt,
  title        = {Fast, Differentiable, GPU-Accelerated Ray Tracing for Multiple Diffraction and Reflection Paths},
  author       = {J\'{e}rome Eertmans and Sophie Lequeu and Beno\^{\i}t Legat and Laurent Jacques and Claude Oestges},
  year         = 2025,
  url          = {TODO},
  eprint       = {TODO},
  archiveprefix = {TODO},
  primaryclass = {TODO}
}
```
