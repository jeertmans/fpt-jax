# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Image method acceleration in `trace_rays` (`use_image_method=True`, enabled by default), which solves planar reflections and transmissions in closed form and restricts numerical optimization to diffraction edges.
- Adaptive solver support via `max_num_iters`, which uses a `while_loop` with a Cauchy termination criterion on both parameter and path length updates.
- Fast path for pure reflection paths, short-circuiting optimization when no diffraction is present.
- Projected curvature preconditioning for the initial inverse Hessian on diffraction edges.
- Initial diffraction edge coordinate centered at midpoint t = 0.5.
- Benchmark suite and dataset generation scripts comparing runtime and accuracy against ECOS and Sionna RT.
- Unit tests for the adaptive solver and argument validation.

### Changed

- `num_iters` in `trace_rays` now defaults to `None`. Either `num_iters` or `max_num_iters` must be specified.
- When `max_num_iters` is set, `rtol` and `atol` are required, and `unroll` must be 1 or False.

### Fixed

- Fixed edge classification in `_planar_info` so diffraction edges on 2D triangle mesh elements are not treated as planar reflections.
- Added step clipping in [-3.0, 3.0] to prevent unphysical parameter excursions across mirror planes.
- Added scale-aware Powell damping to avoid BFGS inverse Hessian ill-conditioning when s^T y is near zero.
- Added backtracking fallback in the fixed-point line search when estimated curvature is negative.
- Added diagonal regularization for planar components in the backward pass during implicit differentiation.

### Performance

- Pure reflection paths run up to 34x faster on CPU and 12x faster on GPU by bypassing the optimizer.
- Caching coordinates across iterations and line search steps reduces image method calls by up to two-thirds.
- Mixed reflection and diffraction paths reach millimeter accuracy in 3 to 5 iterations.

## [0.1.1] - 2026-03-04

### Changed

- Refactored test suite to use TOML files for test case definitions (#2).
- Added `bump-my-version` development dependency.
- Fixed `implicit_diff` argument description in docstring.

## [0.1.0] - 2026-02-28

### Added

- Initial public release of `fpt-jax` on PyPI.
- Core Fermat path tracing solver using quasi-Newton optimization (BFGS) with JAX.
- Support for reverse-mode and implicit differentiation via custom VJP.
- Vectorized batch ray tracing with broadcasting.
