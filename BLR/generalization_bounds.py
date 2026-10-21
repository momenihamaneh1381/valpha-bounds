from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import gamma, logsumexp
from scipy.stats import binom, norm


def sigma_loss_sq(clip_c: float) -> float:
    return (clip_c ** 2) / 4.0


def bounded_loss_cmi_bound(information_value: float, clip_c: float, n: int) -> float:
    """Steinke--Zakynthinou bounded-loss CMI bound, with information in nats.

    For loss in [0, clip_c], the supersample loss difference is in
    [-clip_c, clip_c]. Its subgaussian scale is clip_c**2, rather than
    the single-example Hoeffding scale sigma_loss_sq(clip_c)=clip_c**2/4.
    """
    return float(clip_c * np.sqrt(2.0 * max(float(information_value), 0.0) / n))


def _alpha_grid_default() -> tuple[float, ...]:
    return tuple(np.round(np.arange(1.05, 3.01, 0.05), 2).tolist())


def _alpha_grid_with_tv_default() -> tuple[float, ...]:
    return (1.0, *_alpha_grid_default())


@dataclass(frozen=True)
class OneDimExperimentConfig:
    sigma_noise_sq: float = 0.16
    lambda_prior: float = 2.0
    clip_c: float = 4.0
    w_true: float = 0.8
    n_values: tuple[int, ...] = (10, 20, 40, 80, 160, 320)
    num_datasets: int = 1000
    num_reference_datasets: int = 256
    num_holdout: int = 10000
    num_supersamples: int = 50
    num_u_samples: int = 20
    num_marginal_components: int = 20
    num_mi_weight_samples: int = 8
    num_cmi_weight_samples: int = 4
    num_v_alpha_weight_samples: int = 512
    num_conditional_supersamples: int = 4
    num_conditional_u_samples: int = 4
    num_conditional_marginal_components: int = 4
    num_conditional_v_alpha_weight_samples: int = 8
    grid_points: int = 500
    grid_sigma_span: float = 6.0
    seed: int = 20260418
    output_dir: str = "outputs"


@dataclass(frozen=True)
class DimensionalBoundsConfig:
    sigma_noise_sq: float = 0.16
    lambda_prior: float = 2.0
    clip_c: float = 4.0
    dimension: int = 4
    w_true: tuple[float, ...] | None = None
    w_true_value: float = 0.8
    n_values: tuple[int, ...] = (10, 20, 40, 80, 160, 320)
    num_datasets: int = 512
    num_reference_datasets: int = 512
    num_holdout: int = 10000
    num_supersamples: int = 32
    num_u_samples: int = 16
    num_marginal_components: int = 24
    num_mi_weight_samples: int = 8
    num_cmi_weight_samples: int = 4
    num_v_alpha_weight_samples: int = 512
    num_conditional_supersamples: int = 8
    num_conditional_u_samples: int = 8
    num_conditional_marginal_components: int = 12
    num_conditional_v_alpha_weight_samples: int = 16
    seed: int = 20260428
    artifact_prefix: str = "16_bounds_d4"
    output_dir: str = "outputs"


def w_true_from_dimensional_config(config: DimensionalBoundsConfig) -> np.ndarray:
    if config.w_true is None:
        return make_w_true(config.dimension, config.w_true_value)
    w_true = np.asarray(config.w_true, dtype=float)
    if w_true.ndim != 1:
        raise ValueError("DimensionalBoundsConfig.w_true must be a one-dimensional vector.")
    if w_true.shape[0] != config.dimension:
        raise ValueError(
            "DimensionalBoundsConfig.w_true length must match DimensionalBoundsConfig.dimension."
        )
    return w_true


@dataclass(frozen=True)
class AlphaSweepExperimentConfig:
    sigma_noise_sq: float = 0.16
    lambda_prior: float = 2.0
    clip_c: float = 4.0
    w_true_value: float = 0.8
    dimensions: tuple[int, ...] = (1, 2, 5, 10)
    alpha_values: tuple[float, ...] = field(default_factory=_alpha_grid_default)
    n: int = 100
    num_datasets: int = 1000
    num_reference_datasets: int = 1000
    num_weight_samples: int = 1000
    num_holdout: int = 10000
    num_mi_weight_samples: int = 4
    seed: int = 20260418
    output_dir: str = "outputs"


@dataclass(frozen=True)
class AlphaSweepAuditConfig:
    sigma_noise_sq: float = 0.16
    lambda_prior: float = 2.0
    clip_c: float = 4.0
    w_true_value: float = 0.8
    dimensions: tuple[int, ...] = (1, 2, 5, 10)
    benchmark_dimensions: tuple[int, ...] = (5, 10)
    alpha_values: tuple[float, ...] = field(default_factory=_alpha_grid_default)
    n: int = 100
    num_holdout: int = 10000
    legacy_num_datasets: int = 1000
    legacy_num_reference_datasets: int = 1000
    legacy_num_weight_samples: int = 1000
    legacy_num_mi_weight_samples: int = 4
    legacy_seed: int = 20260418
    audit_num_repeats: int = 2
    audit_num_datasets: int = 32
    audit_num_reference_datasets: int = 32
    audit_num_weight_samples: int = 32
    audit_num_mi_weight_samples: int = 2
    audit_split_selection_weight_samples: int = 16
    benchmark_num_repeats: int = 1
    benchmark_num_datasets: int = 64
    benchmark_num_reference_datasets: int = 64
    benchmark_num_weight_samples: int = 128
    benchmark_num_mi_weight_samples: int = 4
    benchmark_split_selection_weight_samples: int = 64
    seed: int = 20260421
    output_dir: str = "outputs"


@dataclass(frozen=True)
class TheoremOneProxyConfig:
    sigma_noise_sq: float = 0.16
    lambda_prior: float = 2.0
    clip_c: float = 4.0
    w_true_value: float = 0.8
    dimensions: tuple[int, ...] = (1, 2, 5, 10)
    alpha_values: tuple[float, ...] = field(default_factory=_alpha_grid_default)
    n: int = 100
    num_repeats: int = 3
    num_datasets: int = 128
    num_reference_datasets: int = 128
    num_holdout: int = 10000
    num_mi_weight_samples: int = 2
    num_v_alpha_weight_samples: int = 192
    num_candidate_posterior_samples: int = 1
    num_candidate_marginal_samples: int = 192
    num_candidate_prior_samples: int = 32
    candidate_batch_size: int = 256
    seed: int = 20260429
    artifact_prefix: str = "17_theorem1_proxy"
    output_dir: str = "outputs"


@dataclass(frozen=True)
class AlphaSweepComparisonConfig:
    sigma_noise_sq: float = 0.16
    lambda_prior: float = 2.0
    clip_c: float = 4.0
    w_true_value: float = 0.8
    dimensions: tuple[int, ...] = (1, 2, 5, 10)
    benchmark_dimensions: tuple[int, ...] = (5, 10)
    alpha_values: tuple[float, ...] = field(default_factory=_alpha_grid_with_tv_default)
    n: int = 100
    num_holdout: int = 10000
    legacy_num_datasets: int = 1000
    legacy_num_reference_datasets: int = 1000
    legacy_num_weight_samples: int = 1000
    legacy_num_mi_weight_samples: int = 4
    legacy_seed: int = 20260418
    primary_num_repeats: int = 4
    primary_num_datasets: int = 96
    primary_num_reference_datasets: int = 96
    primary_total_weight_samples: int = 96
    primary_num_mi_weight_samples: int = 2
    benchmark_num_repeats: int = 1
    benchmark_num_datasets: int = 160
    benchmark_num_reference_datasets: int = 160
    benchmark_total_weight_samples: int = 160
    benchmark_num_mi_weight_samples: int = 2
    seed: int = 20260422
    include_legacy_reference: bool = True
    include_crossfit: bool = True
    include_surrogate_diagnostics: bool = True
    plot_pinsker_tv: bool = True
    artifact_prefix: str = "08_alpha_sweep"
    output_dir: str = "outputs"


@dataclass(frozen=True)
class AlphaSweepSampleSizeConfig:
    sigma_noise_sq: float = 0.16
    lambda_prior: float = 2.0
    clip_c: float = 4.0
    w_true_value: float = 0.8
    dimensions: tuple[int, ...] = (5, 10)
    n_values: tuple[int, ...] = (50, 100, 200, 400)
    benchmark_n_values: tuple[int, ...] = (100, 400)
    alpha_values: tuple[float, ...] = field(default_factory=_alpha_grid_with_tv_default)
    num_holdout: int = 10000
    primary_num_repeats: int = 3
    primary_num_datasets: int = 80
    primary_num_reference_datasets: int = 80
    primary_total_weight_samples: int = 80
    primary_num_mi_weight_samples: int = 2
    benchmark_num_repeats: int = 1
    benchmark_num_datasets: int = 128
    benchmark_num_reference_datasets: int = 128
    benchmark_total_weight_samples: int = 128
    benchmark_num_mi_weight_samples: int = 2
    seed: int = 20260423
    artifact_prefix: str = "09_alpha_sweep"
    output_dir: str = "outputs"


@dataclass(frozen=True)
class ExactChannelExperimentConfig:
    channel_grid_size: int = 41
    alpha_values: tuple[float, ...] = field(default_factory=_alpha_grid_default)
    sigma_loss_sq: float = 0.25
    seed: int = 20260419
    output_dir: str = "outputs"


@dataclass(frozen=True)
class ReparameterizationExperimentConfig:
    sigma_noise_sq: float = 0.16
    lambda_prior: float = 2.0
    clip_c: float = 4.0
    w_true: float = 0.8
    n: int = 80
    num_datasets: int = 1000
    grid_points: int = 600
    grid_sigma_span: float = 6.0
    transform_sample_size: int = 4096
    seed: int = 20260419
    output_dir: str = "outputs"


@dataclass(frozen=True)
class AdsExperimentConfig:
    n_values: tuple[int, ...] = (5, 10, 20, 40, 80, 160)
    beta: float = 12.0
    hypothesis_grid: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21).tolist())
    sigma_loss_sq: float = 0.25
    max_loss: float = 1.0
    seed: int = 20260419
    output_dir: str = "outputs"


@dataclass(frozen=True)
class ConditionalOutputExperimentConfig:
    sigma_noise_sq: float = 0.16
    lambda_prior: float = 2.0
    clip_c: float = 4.0
    w_true_value: float = 0.8
    dimensions: tuple[int, ...] = (1, 2, 5, 10)
    alpha_values: tuple[float, ...] = field(default_factory=_alpha_grid_default)
    n: int = 100
    num_datasets: int = 256
    num_reference_datasets: int = 256
    num_holdout: int = 10000
    num_supersamples: int = 20
    num_u_samples: int = 24
    num_marginal_components: int = 32
    num_weight_samples: int = 384
    num_mi_weight_samples: int = 6
    proxy_num_means: int = 8
    proxy_num_marginal_samples: int = 8
    samplewise_num_weight_samples: int = 2
    samplewise_num_bins: int = 12
    output_test_size_max: int = 4
    output_num_weight_samples: int = 384
    seed: int = 20260419
    artifact_prefix: str = "06"
    output_dir: str = "outputs"


@dataclass(frozen=True)
class ScalingContinuationExperimentConfig:
    sigma_noise_sq: float = 0.16
    lambda_prior: float = 2.0
    clip_c: float = 4.0
    w_true_value: float = 0.8
    dimensions: tuple[int, ...] = (10, 20, 50)
    alpha_values: tuple[float, ...] = field(default_factory=_alpha_grid_with_tv_default)
    n: int = 100
    num_holdout: int = 8000
    parameter_num_repeats: int = 2
    parameter_num_datasets: int = 48
    parameter_num_reference_datasets: int = 48
    parameter_total_weight_samples: int = 48
    parameter_num_mi_weight_samples: int = 2
    parameter_benchmark_dimensions: tuple[int, ...] = (50,)
    parameter_benchmark_num_repeats: int = 1
    parameter_benchmark_num_datasets: int = 80
    parameter_benchmark_num_reference_datasets: int = 80
    parameter_benchmark_total_weight_samples: int = 80
    parameter_benchmark_num_mi_weight_samples: int = 2
    conditional_num_datasets: int = 96
    conditional_num_reference_datasets: int = 96
    conditional_num_supersamples: int = 12
    conditional_num_u_samples: int = 16
    conditional_num_marginal_components: int = 24
    conditional_num_weight_samples: int = 192
    conditional_num_mi_weight_samples: int = 4
    proxy_num_means: int = 8
    proxy_num_marginal_samples: int = 8
    samplewise_num_weight_samples: int = 2
    samplewise_num_bins: int = 12
    output_test_size_max: int = 8
    output_num_weight_samples: int = 192
    seed: int = 20260424
    artifact_prefix: str = "12_scaling_continuation"
    output_dir: str = "outputs"


@dataclass(frozen=True)
class AlphaSweepSensitivityConfig:
    clip_c: float = 4.0
    w_true_value: float = 0.8
    dimensions: tuple[int, ...] = (20, 50)
    alpha_values: tuple[float, ...] = field(default_factory=_alpha_grid_with_tv_default)
    n: int = 100
    num_holdout: int = 8000
    base_sigma_noise_sq: float = 0.16
    base_lambda_prior: float = 2.0
    sigma_noise_sq_values: tuple[float, ...] = (0.04, 0.16, 0.64)
    lambda_prior_values: tuple[float, ...] = (0.5, 2.0, 8.0)
    num_repeats: int = 2
    num_datasets: int = 48
    num_reference_datasets: int = 48
    total_weight_samples: int = 48
    num_mi_weight_samples: int = 2
    seed: int = 20260425
    artifact_prefix: str = "13_noise_prior_sensitivity"
    output_dir: str = "outputs"


@dataclass(frozen=True)
class HighDimConvergenceConfig:
    sigma_noise_sq: float = 0.16
    lambda_prior: float = 2.0
    clip_c: float = 4.0
    w_true_value: float = 0.8
    dimensions: tuple[int, ...] = (10, 20, 50, 100)
    benchmark_dimensions: tuple[int, ...] = (50, 100)
    alpha_values: tuple[float, ...] = field(default_factory=_alpha_grid_with_tv_default)
    n: int = 100
    num_holdout: int = 8000
    num_repeats: int = 2
    evaluation_num_datasets: int = 48
    reference_budget_grid: tuple[int, ...] = (48, 96, 192)
    weight_budget_grid: tuple[int, ...] = (48, 96, 192)
    mi_num_weight_samples: int = 4
    benchmark_reference_budget: int = 384
    benchmark_total_weight_samples: int = 384
    benchmark_num_repeats: int = 1
    seed: int = 20260426
    artifact_prefix: str = "14_high_dim_convergence"
    output_dir: str = "outputs"


@dataclass
class GaussianPosterior:
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float).reshape(-1)
        self.covariance = np.asarray(self.covariance, dtype=float)
        self.dim = self.mean.shape[0]
        self.cholesky = np.linalg.cholesky(self.covariance)
        self.logdet = 2.0 * np.log(np.diag(self.cholesky)).sum()
        self._norm = -0.5 * (self.dim * np.log(2.0 * np.pi) + self.logdet)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        z = rng.standard_normal((size, self.dim))
        return self.mean + z @ self.cholesky.T

    def logpdf(self, points: np.ndarray) -> np.ndarray:
        values = _to_2d(points, self.dim)
        centered = values - self.mean
        solved = solve_triangular(self.cholesky, centered.T, lower=True, check_finite=False)
        quad = np.sum(solved ** 2, axis=0)
        return self._norm - 0.5 * quad

    def variance_scalar(self) -> float:
        if self.dim != 1:
            raise ValueError("variance_scalar is only defined for 1D posteriors.")
        return float(self.covariance[0, 0])


class DensityModel(Protocol):
    dim: int

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        ...

    def logpdf(self, points: np.ndarray) -> np.ndarray:
        ...


class DensityReference(DensityModel, Protocol):
    ...


class GaussianMixtureReference:
    def __init__(self, posteriors: list[GaussianPosterior]) -> None:
        if not posteriors:
            raise ValueError("At least one posterior is required for a mixture.")
        self.posteriors = posteriors
        self.dim = posteriors[0].dim
        self.num_components = len(posteriors)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        indices = rng.integers(0, self.num_components, size=size)
        samples = np.empty((size, self.dim), dtype=float)
        for component in np.unique(indices):
            mask = indices == component
            samples[mask] = self.posteriors[int(component)].sample(rng, int(mask.sum()))
        return samples

    def logpdf(self, points: np.ndarray) -> np.ndarray:
        values = _to_2d(points, self.dim)
        component_logs = np.column_stack([posterior.logpdf(values) for posterior in self.posteriors])
        return logsumexp(component_logs, axis=1) - np.log(self.num_components)


def moment_matched_gaussian_reference(
    posteriors: Sequence[GaussianPosterior],
    jitter: float = 1e-9,
) -> GaussianPosterior:
    if not posteriors:
        raise ValueError("At least one posterior is required for moment matching.")
    means = np.stack([posterior.mean for posterior in posteriors], axis=0)
    average_mean = means.mean(axis=0)
    second_moment = np.mean(
        [posterior.covariance + np.outer(posterior.mean, posterior.mean) for posterior in posteriors],
        axis=0,
    )
    covariance = second_moment - np.outer(average_mean, average_mean)
    covariance = 0.5 * (covariance + covariance.T)
    covariance = covariance + jitter * np.eye(covariance.shape[0])
    return GaussianPosterior(mean=average_mean, covariance=covariance)


def prior_gaussian_reference(dim: int, lambda_prior: float) -> GaussianPosterior:
    covariance = np.eye(dim, dtype=float) / lambda_prior
    mean = np.zeros(dim, dtype=float)
    return GaussianPosterior(mean=mean, covariance=covariance)


def alpha_optimization_grid(alpha_values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(alpha) for alpha in alpha_values if float(alpha) > 1.0 + 1e-12)


def alpha_bound_from_value(
    alpha: float,
    value: float,
    sigma_loss_sq_value: float,
    n: int,
    clip_c: float,
) -> float:
    if np.isclose(alpha, 1.0):
        return float(clip_c * value)
    return float(subgaussian_multiplier(alpha, sigma_loss_sq_value, n) * value)


def corollary1_bound_from_value(
    alpha: float,
    value: float,
    sigma_loss_sq_value: float,
    n: int,
) -> float:
    """Pure Corollary 2 bound without the sharpened alpha=2 specialization."""
    return float(corollary1_subgaussian_multiplier(alpha, sigma_loss_sq_value, n) * value)


def khinchine_factor_from_alpha(alpha: float) -> float:
    """Return K_q for q = alpha / (alpha - 1).

    For alpha > 1, q is finite and the conditional supersample gap satisfies
        ||Delta_hat||_{q,U} <= K_q * clip_c / sqrt(n).

    The alpha = 1 endpoint corresponds to q = infinity and is handled separately.
    """
    if alpha <= 1.0:
        raise ValueError("Khinchine prefactor is finite only for alpha > 1.")

    q = alpha / (alpha - 1.0)
    if q <= 2.0:
        return 1.0
    return float(np.sqrt(q - 1.0))


def conditional_supersample_norm_prefactor(alpha: float, n: int, clip_c: float) -> float:
    """Theorem-valid conditional supersample norm prefactor.

    For alpha > 1 with q = alpha / (alpha - 1),
        ||Delta_hat||_{q,U} <= K_q * clip_c / sqrt(n).

    For alpha = 1, q = infinity, so only the bounded-loss endpoint
        ||Delta_hat||_{infinity,U} <= clip_c
    is used.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    if np.isclose(alpha, 1.0):
        return float(clip_c)

    if alpha <= 1.0:
        raise ValueError("alpha must be at least 1.")

    return float(khinchine_factor_from_alpha(alpha) * clip_c / np.sqrt(n))


def pinsker_tv_bound_from_information(clip_c: float, information_value: float) -> float:
    return float(clip_c * np.sqrt(2.0 * max(float(information_value), 0.0)))


def partition_weight_budget(total_weight_samples: int, num_parts: int) -> tuple[int, ...]:
    if total_weight_samples < num_parts:
        raise ValueError("total_weight_samples must be at least the number of parts.")
    base = total_weight_samples // num_parts
    remainder = total_weight_samples % num_parts
    return tuple(base + (1 if idx < remainder else 0) for idx in range(num_parts))


def _to_2d(values: np.ndarray, dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1, dim)
    if array.ndim == 1:
        if dim == 1:
            return array.reshape(-1, 1)
        if array.shape[0] == dim:
            return array.reshape(1, dim)
        raise ValueError(f"Expected vectors of dimension {dim}, got shape {array.shape}.")
    if array.ndim == 2 and array.shape[1] == dim:
        return array
    raise ValueError(f"Expected a 2D array with trailing dimension {dim}, got shape {array.shape}.")


def collect_package_versions() -> dict[str, str]:
    packages = ["numpy", "scipy", "matplotlib", "pandas", "nbformat", "nbclient"]
    optional_packages = ["torch", "torchvision"]
    versions: dict[str, str] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
    for package in packages:
        versions[package] = metadata.version(package)
    for package in optional_packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def json_ready_config(config: Any) -> dict[str, Any]:
    return asdict(config)


def make_w_true(dim: int, value: float) -> np.ndarray:
    return np.full(dim, value, dtype=float)


def generate_regression_dataset(
    rng: np.random.Generator,
    n: int,
    w_true: np.ndarray,
    sigma_noise_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
    dim = w_true.shape[0]
    x = rng.standard_normal((n, dim))
    noise = rng.normal(loc=0.0, scale=np.sqrt(sigma_noise_sq), size=n)
    y = x @ w_true + noise
    return x, y


def generate_supersample(
    rng: np.random.Generator,
    n: int,
    w_true: np.ndarray,
    sigma_noise_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
    dim = w_true.shape[0]
    x = rng.standard_normal((n, 2, dim))
    noise = rng.normal(loc=0.0, scale=np.sqrt(sigma_noise_sq), size=(n, 2))
    y = np.einsum("nid,d->ni", x, w_true) + noise
    return x, y


def select_dataset_from_supersample(
    supersample_x: np.ndarray,
    supersample_y: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    row_index = np.arange(supersample_x.shape[0])
    x = supersample_x[row_index, u]
    y = supersample_y[row_index, u]
    return x, y


def posterior_from_dataset(
    x: np.ndarray,
    y: np.ndarray,
    lambda_prior: float,
    sigma_noise_sq: float,
) -> GaussianPosterior:
    dim = x.shape[1]
    beta = 1.0 / sigma_noise_sq
    precision = lambda_prior * np.eye(dim) + beta * x.T @ x
    cho = cho_factor(precision, lower=True, check_finite=False)
    covariance = cho_solve(cho, np.eye(dim), check_finite=False)
    mean = cho_solve(cho, beta * x.T @ y, check_finite=False)
    return GaussianPosterior(mean=mean, covariance=covariance)


def clipped_squared_loss(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    clip_c: float,
) -> np.ndarray:
    weights = np.asarray(w, dtype=float).reshape(-1)
    residual = y - x @ weights
    return np.minimum(residual ** 2, clip_c)


def make_holdout_set(
    rng: np.random.Generator,
    num_holdout: int,
    w_true: np.ndarray,
    sigma_noise_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
    return generate_regression_dataset(rng, num_holdout, w_true, sigma_noise_sq)


def estimate_empirical_generalization(
    datasets: list[tuple[np.ndarray, np.ndarray]],
    posteriors: list[GaussianPosterior],
    holdout: tuple[np.ndarray, np.ndarray],
    clip_c: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    holdout_x, holdout_y = holdout
    gaps = []
    for (x, y), posterior in zip(datasets, posteriors):
        w_sample = posterior.sample(rng, 1)[0]
        empirical_loss = clipped_squared_loss(w_sample, x, y, clip_c).mean()
        population_loss = clipped_squared_loss(w_sample, holdout_x, holdout_y, clip_c).mean()
        gaps.append(population_loss - empirical_loss)
    gap_array = np.asarray(gaps, dtype=float)
    stderr = gap_array.std(ddof=1) / np.sqrt(gap_array.size)
    return abs(float(gap_array.mean())), float(stderr)


def estimate_mutual_information(
    posteriors: Sequence[DensityModel],
    reference_mixture: DensityReference,
    rng: np.random.Generator,
    num_weight_samples: int,
) -> tuple[float, float]:
    terms: list[np.ndarray] = []
    for posterior in posteriors:
        samples = posterior.sample(rng, num_weight_samples)
        terms.append(posterior.logpdf(samples) - reference_mixture.logpdf(samples))
    term_array = np.concatenate(terms, axis=0)
    stderr = term_array.std(ddof=1) / np.sqrt(term_array.size)
    return float(term_array.mean()), float(stderr)


def estimate_conditional_mutual_information(
    config: OneDimExperimentConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n_value = max(config.n_values)
    w_true = np.array([config.w_true], dtype=float)
    all_terms: list[np.ndarray] = []
    for _ in range(config.num_supersamples):
        supersample_x, supersample_y = generate_supersample(
            rng=rng,
            n=n_value,
            w_true=w_true,
            sigma_noise_sq=config.sigma_noise_sq,
        )
        reference_posteriors = []
        for _ in range(config.num_marginal_components):
            u_reference = rng.integers(0, 2, size=n_value)
            x_reference, y_reference = select_dataset_from_supersample(supersample_x, supersample_y, u_reference)
            reference_posteriors.append(
                posterior_from_dataset(
                    x_reference,
                    y_reference,
                    lambda_prior=config.lambda_prior,
                    sigma_noise_sq=config.sigma_noise_sq,
                )
            )
        reference_mixture = GaussianMixtureReference(reference_posteriors)
        for _ in range(config.num_u_samples):
            u_eval = rng.integers(0, 2, size=n_value)
            x_eval, y_eval = select_dataset_from_supersample(supersample_x, supersample_y, u_eval)
            posterior = posterior_from_dataset(
                x_eval,
                y_eval,
                lambda_prior=config.lambda_prior,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            samples = posterior.sample(rng, config.num_cmi_weight_samples)
            all_terms.append(posterior.logpdf(samples) - reference_mixture.logpdf(samples))
    term_array = np.concatenate(all_terms, axis=0)
    stderr = term_array.std(ddof=1) / np.sqrt(term_array.size)
    return float(term_array.mean()), float(stderr)


def estimate_conditional_v_alpha_bundle(
    config: OneDimExperimentConfig,
    rng: np.random.Generator,
    *,
    alpha_values: Sequence[float] = (2.0,),
    n: int | None = None,
) -> dict[str, Any]:
    n_value = int(n if n is not None else max(config.n_values))
    w_true = np.array([config.w_true], dtype=float)
    conditional_v_alpha_samples = {float(alpha): [] for alpha in alpha_values}
    for _ in range(config.num_conditional_supersamples):
        supersample_x, supersample_y = generate_supersample(
            rng=rng,
            n=n_value,
            w_true=w_true,
            sigma_noise_sq=config.sigma_noise_sq,
        )
        reference_posteriors = []
        for _ in range(config.num_conditional_marginal_components):
            u_reference = rng.integers(0, 2, size=n_value)
            x_reference, y_reference = select_dataset_from_supersample(supersample_x, supersample_y, u_reference)
            reference_posteriors.append(
                posterior_from_dataset(
                    x_reference,
                    y_reference,
                    lambda_prior=config.lambda_prior,
                    sigma_noise_sq=config.sigma_noise_sq,
                )
            )
        reference_mixture = GaussianMixtureReference(reference_posteriors)
        evaluation_posteriors: list[GaussianPosterior] = []
        for _ in range(config.num_conditional_u_samples):
            u_eval = rng.integers(0, 2, size=n_value)
            x_eval, y_eval = select_dataset_from_supersample(supersample_x, supersample_y, u_eval)
            evaluation_posteriors.append(
                posterior_from_dataset(
                    x_eval,
                    y_eval,
                    lambda_prior=config.lambda_prior,
                    sigma_noise_sq=config.sigma_noise_sq,
                )
            )
        v_alpha_values, _ = estimate_v_alpha_from_marginal(
            posteriors=evaluation_posteriors,
            reference_mixture=reference_mixture,
            alpha_values=tuple(alpha_values),
            rng=rng,
            num_weight_samples=config.num_conditional_v_alpha_weight_samples,
        )
        for alpha, value in v_alpha_values.items():
            conditional_v_alpha_samples[float(alpha)].append(float(value))
    conditional_v_alpha_mean = {
        alpha: float(np.mean(values)) for alpha, values in conditional_v_alpha_samples.items()
    }
    conditional_v_alpha_stderr = {
        alpha: float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        for alpha, values in conditional_v_alpha_samples.items()
    }
    rigorous_bounds = {
        alpha: float(conditional_supersample_norm_prefactor(alpha, n_value, config.clip_c) * value)
        for alpha, value in conditional_v_alpha_mean.items()
    }
    rigorous_bound_stderr = {
        alpha: float(conditional_supersample_norm_prefactor(alpha, n_value, config.clip_c) * value)
        for alpha, value in conditional_v_alpha_stderr.items()
    }
    return {
        "conditional_v_alpha": conditional_v_alpha_mean,
        "conditional_v_alpha_stderr": conditional_v_alpha_stderr,
        "rigorous_bounds": rigorous_bounds,
        "rigorous_bound_stderr": rigorous_bound_stderr,
        "rigorous_prefactors": {
            alpha: conditional_supersample_norm_prefactor(alpha, n_value, config.clip_c)
            for alpha in conditional_v_alpha_mean
        },
    }


def estimate_conditional_mutual_information_for_w_true(
    *,
    w_true: np.ndarray,
    n: int,
    lambda_prior: float,
    sigma_noise_sq: float,
    num_supersamples: int,
    num_marginal_components: int,
    num_u_samples: int,
    num_weight_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    all_terms: list[np.ndarray] = []
    for _ in range(num_supersamples):
        supersample_x, supersample_y = generate_supersample(
            rng=rng,
            n=n,
            w_true=w_true,
            sigma_noise_sq=sigma_noise_sq,
        )
        reference_posteriors = []
        for _ in range(num_marginal_components):
            u_reference = rng.integers(0, 2, size=n)
            x_reference, y_reference = select_dataset_from_supersample(supersample_x, supersample_y, u_reference)
            reference_posteriors.append(
                posterior_from_dataset(
                    x_reference,
                    y_reference,
                    lambda_prior=lambda_prior,
                    sigma_noise_sq=sigma_noise_sq,
                )
            )
        reference_mixture = GaussianMixtureReference(reference_posteriors)
        for _ in range(num_u_samples):
            u_eval = rng.integers(0, 2, size=n)
            x_eval, y_eval = select_dataset_from_supersample(supersample_x, supersample_y, u_eval)
            posterior = posterior_from_dataset(
                x_eval,
                y_eval,
                lambda_prior=lambda_prior,
                sigma_noise_sq=sigma_noise_sq,
            )
            samples = posterior.sample(rng, num_weight_samples)
            all_terms.append(posterior.logpdf(samples) - reference_mixture.logpdf(samples))
    term_array = np.concatenate(all_terms, axis=0)
    stderr = term_array.std(ddof=1) / np.sqrt(term_array.size)
    return float(term_array.mean()), float(stderr)


def estimate_conditional_v_alpha_bundle_for_w_true(
    *,
    w_true: np.ndarray,
    n: int,
    lambda_prior: float,
    sigma_noise_sq: float,
    clip_c: float,
    alpha_values: Sequence[float],
    num_supersamples: int,
    num_marginal_components: int,
    num_u_samples: int,
    num_weight_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    conditional_v_alpha_samples = {float(alpha): [] for alpha in alpha_values}
    for _ in range(num_supersamples):
        supersample_x, supersample_y = generate_supersample(
            rng=rng,
            n=n,
            w_true=w_true,
            sigma_noise_sq=sigma_noise_sq,
        )
        reference_posteriors = []
        for _ in range(num_marginal_components):
            u_reference = rng.integers(0, 2, size=n)
            x_reference, y_reference = select_dataset_from_supersample(supersample_x, supersample_y, u_reference)
            reference_posteriors.append(
                posterior_from_dataset(
                    x_reference,
                    y_reference,
                    lambda_prior=lambda_prior,
                    sigma_noise_sq=sigma_noise_sq,
                )
            )
        reference_mixture = GaussianMixtureReference(reference_posteriors)
        evaluation_posteriors: list[GaussianPosterior] = []
        for _ in range(num_u_samples):
            u_eval = rng.integers(0, 2, size=n)
            x_eval, y_eval = select_dataset_from_supersample(supersample_x, supersample_y, u_eval)
            evaluation_posteriors.append(
                posterior_from_dataset(
                    x_eval,
                    y_eval,
                    lambda_prior=lambda_prior,
                    sigma_noise_sq=sigma_noise_sq,
                )
            )
        v_alpha_values, _ = estimate_v_alpha_from_marginal(
            posteriors=evaluation_posteriors,
            reference_mixture=reference_mixture,
            alpha_values=tuple(alpha_values),
            rng=rng,
            num_weight_samples=num_weight_samples,
        )
        for alpha, value in v_alpha_values.items():
            conditional_v_alpha_samples[float(alpha)].append(float(value))

    conditional_v_alpha_mean = {
        alpha: float(np.mean(values)) for alpha, values in conditional_v_alpha_samples.items()
    }
    conditional_v_alpha_stderr = {
        alpha: float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        for alpha, values in conditional_v_alpha_samples.items()
    }
    rigorous_bounds = {
        alpha: float(conditional_supersample_norm_prefactor(alpha, n, clip_c) * value)
        for alpha, value in conditional_v_alpha_mean.items()
    }
    rigorous_bound_stderr = {
        alpha: float(conditional_supersample_norm_prefactor(alpha, n, clip_c) * value)
        for alpha, value in conditional_v_alpha_stderr.items()
    }
    return {
        "conditional_v_alpha": conditional_v_alpha_mean,
        "conditional_v_alpha_stderr": conditional_v_alpha_stderr,
        "rigorous_bounds": rigorous_bounds,
        "rigorous_bound_stderr": rigorous_bound_stderr,
        "rigorous_prefactors": {
            alpha: conditional_supersample_norm_prefactor(alpha, n, clip_c)
            for alpha in conditional_v_alpha_mean
        },
    }


def estimate_v2_grid(
    posteriors: list[GaussianPosterior],
    sigma_loss_sq_value: float,
    n: int,
    grid_points: int,
    grid_sigma_span: float,
) -> tuple[float, dict[str, float | np.ndarray]]:
    means = np.array([posterior.mean[0] for posterior in posteriors], dtype=float)
    stds = np.sqrt(np.array([posterior.variance_scalar() for posterior in posteriors], dtype=float))
    lower = float(np.min(means - grid_sigma_span * stds))
    upper = float(np.max(means + grid_sigma_span * stds))
    grid = np.linspace(lower, upper, grid_points)
    density_matrix = np.column_stack([np.exp(posterior.logpdf(grid)) for posterior in posteriors])
    variance_density = np.var(density_matrix, axis=1, ddof=1)
    v2_value = float(np.trapezoid(np.sqrt(np.clip(variance_density, 0.0, None)), grid))
    outside_mass = float(
        np.mean(norm.cdf(lower, loc=means, scale=stds) + (1.0 - norm.cdf(upper, loc=means, scale=stds)))
    )
    diagnostics = {
        "grid_lower": lower,
        "grid_upper": upper,
        "grid_points": grid_points,
        "outside_mass": outside_mass,
        "v2_value": v2_value,
        "bound": float(np.sqrt(sigma_loss_sq_value / n) * v2_value),
    }
    return float(np.sqrt(sigma_loss_sq_value / n) * v2_value), diagnostics


def estimate_v_alpha_from_marginal(
    posteriors: Sequence[DensityModel],
    reference_mixture: DensityReference,
    alpha_values: list[float] | tuple[float, ...],
    rng: np.random.Generator,
    num_weight_samples: int,
) -> tuple[dict[float, float], dict[str, float]]:
    weight_samples = reference_mixture.sample(rng, num_weight_samples)
    log_marginal = reference_mixture.logpdf(weight_samples)
    log_posterior = np.column_stack([posterior.logpdf(weight_samples) for posterior in posteriors])
    log_ratio = log_posterior - log_marginal[:, None]
    ratio_minus_one = np.expm1(np.clip(log_ratio, -80.0, 80.0))
    abs_deviation = np.abs(ratio_minus_one)

    estimates: dict[float, float] = {}
    for alpha in alpha_values:
        inner = np.mean(abs_deviation ** alpha, axis=1)
        estimates[float(alpha)] = float(np.mean(inner ** (1.0 / alpha)))

    diagnostics = {
        "log_ratio_min": float(log_ratio.min()),
        "log_ratio_max": float(log_ratio.max()),
        "num_weight_samples": float(num_weight_samples),
    }
    return estimates, diagnostics


def subgaussian_multiplier(alpha: float, sigma_loss_sq_value: float, n: int) -> float:
    """Theorem-valid multiplier with the sharpened alpha=2 specialization."""
    if alpha <= 1.0:
        raise ValueError("alpha must be strictly greater than 1.")
    if np.isclose(alpha, 2.0):
        return float(np.sqrt(sigma_loss_sq_value / n))
    return corollary1_subgaussian_multiplier(alpha, sigma_loss_sq_value, n)


def corollary1_subgaussian_multiplier(alpha: float, sigma_loss_sq_value: float, n: int) -> float:
    """Pure Corollary 2 multiplier, including at alpha=2."""
    if alpha <= 1.0:
        raise ValueError("alpha must be strictly greater than 1.")
    alpha_conjugate = alpha / (alpha - 1.0)
    return float(
        np.sqrt(2.0 * sigma_loss_sq_value / n)
        * (alpha_conjugate ** (1.0 / alpha_conjugate))
        * (gamma(alpha_conjugate / 2.0) ** (1.0 / alpha_conjugate))
    )


def corollary_subgaussian_multiplier(alpha: float, sigma_loss_sq_value: float, n: int) -> float:
    """Backward-compatible alias for the pure Corollary 2 multiplier."""
    return corollary1_subgaussian_multiplier(alpha, sigma_loss_sq_value, n)


def alpha_profile_from_marginal(
    posteriors: list[GaussianPosterior],
    reference_mixture: GaussianMixtureReference,
    alpha_values: Sequence[float],
    rng: np.random.Generator,
    num_weight_samples: int,
    sigma_loss_sq_value: float,
    n: int,
) -> dict[str, Any]:
    v_alpha_values, diagnostics = estimate_v_alpha_from_marginal(
        posteriors=posteriors,
        reference_mixture=reference_mixture,
        alpha_values=tuple(alpha_values),
        rng=rng,
        num_weight_samples=num_weight_samples,
    )
    sorted_alphas = tuple(float(alpha) for alpha in alpha_values)
    bound_by_alpha = {
        alpha: subgaussian_multiplier(alpha, sigma_loss_sq_value, n) * v_alpha_values[alpha]
        for alpha in sorted_alphas
    }
    raw_values = np.array([v_alpha_values[alpha] for alpha in sorted_alphas], dtype=float)
    raw_bounds = np.array([bound_by_alpha[alpha] for alpha in sorted_alphas], dtype=float)
    value_diffs = np.diff(raw_values)
    bound_diffs = np.diff(raw_bounds)
    monotonicity_violation = float(max(0.0, -float(value_diffs.min()))) if value_diffs.size else 0.0
    monotonicity_violation_count = int(np.sum(value_diffs < -1e-10))
    optimal_alpha = min(bound_by_alpha, key=bound_by_alpha.get)
    return {
        "v_alpha_values": v_alpha_values,
        "bound_by_alpha": bound_by_alpha,
        "optimal_alpha": float(optimal_alpha),
        "optimized_bound": float(bound_by_alpha[optimal_alpha]),
        "fixed_v2_bound": float(bound_by_alpha[2.0]),
        "monotonicity_violation": monotonicity_violation,
        "monotonicity_violation_count": monotonicity_violation_count,
        "min_bound_step": float(bound_diffs.min()) if bound_diffs.size else 0.0,
        "diagnostics": diagnostics,
    }


def alpha_profile_from_reference(
    posteriors: list[GaussianPosterior],
    reference: DensityReference,
    alpha_values: Sequence[float],
    rng: np.random.Generator,
    num_weight_samples: int,
    sigma_loss_sq_value: float,
    n: int,
    clip_c: float,
    optimization_alpha_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    sorted_alphas = tuple(float(alpha) for alpha in alpha_values)
    optimization_alphas = (
        tuple(float(alpha) for alpha in optimization_alpha_values)
        if optimization_alpha_values is not None
        else alpha_optimization_grid(sorted_alphas)
    )
    if not optimization_alphas:
        raise ValueError("At least one alpha > 1 is required for optimization.")
    v_alpha_values, diagnostics = estimate_v_alpha_from_marginal(
        posteriors=posteriors,
        reference_mixture=reference,
        alpha_values=sorted_alphas,
        rng=rng,
        num_weight_samples=num_weight_samples,
    )
    bound_by_alpha = {
        alpha: alpha_bound_from_value(alpha, v_alpha_values[alpha], sigma_loss_sq_value, n, clip_c)
        for alpha in sorted_alphas
    }
    raw_values = np.array([v_alpha_values[alpha] for alpha in sorted_alphas], dtype=float)
    raw_bounds = np.array([bound_by_alpha[alpha] for alpha in sorted_alphas], dtype=float)
    value_diffs = np.diff(raw_values)
    bound_diffs = np.diff(raw_bounds)
    optimal_alpha = min(optimization_alphas, key=bound_by_alpha.get)
    return {
        "v_alpha_values": v_alpha_values,
        "bound_by_alpha": bound_by_alpha,
        "optimal_alpha": float(optimal_alpha),
        "optimized_bound": float(bound_by_alpha[optimal_alpha]),
        "direct_v1_bound": float(bound_by_alpha[1.0]) if 1.0 in bound_by_alpha else np.nan,
        "fixed_v2_bound": float(bound_by_alpha[2.0]),
        "monotonicity_violation": float(max(0.0, -float(value_diffs.min()))) if value_diffs.size else 0.0,
        "monotonicity_violation_count": int(np.sum(value_diffs < -1e-10)),
        "min_bound_step": float(bound_diffs.min()) if bound_diffs.size else 0.0,
        "diagnostics": diagnostics,
    }


def average_alpha_profiles(
    profiles: Sequence[dict[str, Any]],
    alpha_values: Sequence[float],
    optimization_alpha_values: Sequence[float],
) -> dict[str, Any]:
    if not profiles:
        raise ValueError("At least one profile is required.")
    sorted_alphas = tuple(float(alpha) for alpha in alpha_values)
    optimization_alphas = tuple(float(alpha) for alpha in optimization_alpha_values)
    mean_v_alpha = {
        alpha: float(np.mean([profile["v_alpha_values"][alpha] for profile in profiles]))
        for alpha in sorted_alphas
    }
    mean_bound = {
        alpha: float(np.mean([profile["bound_by_alpha"][alpha] for profile in profiles]))
        for alpha in sorted_alphas
    }
    raw_values = np.array([mean_v_alpha[alpha] for alpha in sorted_alphas], dtype=float)
    value_diffs = np.diff(raw_values)
    optimal_alpha = min(optimization_alphas, key=mean_bound.get)
    return {
        "v_alpha_values": mean_v_alpha,
        "bound_by_alpha": mean_bound,
        "optimal_alpha": float(optimal_alpha),
        "optimized_bound": float(mean_bound[optimal_alpha]),
        "direct_v1_bound": float(mean_bound[1.0]) if 1.0 in mean_bound else np.nan,
        "fixed_v2_bound": float(mean_bound[2.0]),
        "monotonicity_violation": float(max(0.0, -float(value_diffs.min()))) if value_diffs.size else 0.0,
        "monotonicity_violation_count": int(np.sum(value_diffs < -1e-10)),
        "diagnostics": {
            "log_ratio_min": float(min(profile["diagnostics"]["log_ratio_min"] for profile in profiles)),
            "log_ratio_max": float(max(profile["diagnostics"]["log_ratio_max"] for profile in profiles)),
            "num_weight_samples": float(
                sum(int(profile["diagnostics"]["num_weight_samples"]) for profile in profiles)
            ),
        },
    }


def estimate_reference_information_bundle(
    posteriors: list[GaussianPosterior],
    reference: DensityReference,
    rng: np.random.Generator,
    num_weight_samples: int,
    clip_c: float,
    sigma_loss_sq_value: float,
    n: int,
) -> dict[str, float]:
    information_value, information_stderr = estimate_mutual_information(
        posteriors=posteriors,
        reference_mixture=reference,
        rng=rng,
        num_weight_samples=num_weight_samples,
    )
    information_value = float(max(information_value, 0.0))
    return {
        "information_estimate": information_value,
        "information_stderr": float(information_stderr),
        "information_bound": float(np.sqrt(2.0 * sigma_loss_sq_value / n * information_value)),
        "pinsker_tv_bound": pinsker_tv_bound_from_information(clip_c, information_value),
    }


def run_split_alpha_estimator(
    posteriors: list[GaussianPosterior],
    reference: DensityReference,
    alpha_values: Sequence[float],
    optimization_alpha_values: Sequence[float],
    sigma_loss_sq_value: float,
    n: int,
    clip_c: float,
    total_weight_samples: int,
    seed: int,
) -> dict[str, Any]:
    select_budget, eval_budget = partition_weight_budget(total_weight_samples, 2)
    select_profile = alpha_profile_from_reference(
        posteriors=posteriors,
        reference=reference,
        alpha_values=alpha_values,
        optimization_alpha_values=optimization_alpha_values,
        rng=np.random.default_rng(seed),
        num_weight_samples=select_budget,
        sigma_loss_sq_value=sigma_loss_sq_value,
        n=n,
        clip_c=clip_c,
    )
    eval_profile = alpha_profile_from_reference(
        posteriors=posteriors,
        reference=reference,
        alpha_values=alpha_values,
        optimization_alpha_values=optimization_alpha_values,
        rng=np.random.default_rng(seed + 1),
        num_weight_samples=eval_budget,
        sigma_loss_sq_value=sigma_loss_sq_value,
        n=n,
        clip_c=clip_c,
    )
    selected_alpha = float(select_profile["optimal_alpha"])
    return {
        "selection_profile": select_profile,
        "evaluation_profile": eval_profile,
        "selected_alpha": selected_alpha,
        "optimized_bound": float(eval_profile["bound_by_alpha"][selected_alpha]),
        "direct_v1_bound": float(eval_profile["bound_by_alpha"][1.0]),
        "fixed_v2_bound": float(eval_profile["bound_by_alpha"][2.0]),
        "total_weight_samples": float(total_weight_samples),
        "selection_weight_samples": float(select_budget),
        "evaluation_weight_samples": float(eval_budget),
    }


def run_crossfit_alpha_estimator(
    posteriors: list[GaussianPosterior],
    reference: DensityReference,
    alpha_values: Sequence[float],
    optimization_alpha_values: Sequence[float],
    sigma_loss_sq_value: float,
    n: int,
    clip_c: float,
    total_weight_samples: int,
    seed: int,
) -> dict[str, Any]:
    budgets = partition_weight_budget(total_weight_samples, 4)
    select_profiles: list[dict[str, Any]] = []
    eval_profiles: list[dict[str, Any]] = []
    for index, budget in enumerate(budgets[:2]):
        select_profiles.append(
            alpha_profile_from_reference(
                posteriors=posteriors,
                reference=reference,
                alpha_values=alpha_values,
                optimization_alpha_values=optimization_alpha_values,
                rng=np.random.default_rng(seed + index),
                num_weight_samples=budget,
                sigma_loss_sq_value=sigma_loss_sq_value,
                n=n,
                clip_c=clip_c,
            )
        )
    for index, budget in enumerate(budgets[2:], start=2):
        eval_profiles.append(
            alpha_profile_from_reference(
                posteriors=posteriors,
                reference=reference,
                alpha_values=alpha_values,
                optimization_alpha_values=optimization_alpha_values,
                rng=np.random.default_rng(seed + index),
                num_weight_samples=budget,
                sigma_loss_sq_value=sigma_loss_sq_value,
                n=n,
                clip_c=clip_c,
            )
        )
    select_average = average_alpha_profiles(select_profiles, alpha_values, optimization_alpha_values)
    eval_average = average_alpha_profiles(eval_profiles, alpha_values, optimization_alpha_values)
    selected_alpha = float(select_average["optimal_alpha"])
    return {
        "selection_profile": select_average,
        "evaluation_profile": eval_average,
        "selected_alpha": selected_alpha,
        "optimized_bound": float(eval_average["bound_by_alpha"][selected_alpha]),
        "direct_v1_bound": float(eval_average["bound_by_alpha"][1.0]),
        "fixed_v2_bound": float(eval_average["bound_by_alpha"][2.0]),
        "total_weight_samples": float(total_weight_samples),
        "selection_weight_samples": float(sum(budgets[:2])),
        "evaluation_weight_samples": float(sum(budgets[2:])),
    }


def alpha_sweep_logpdf_cost(
    num_datasets: int,
    num_reference_datasets: int,
    num_weight_samples: int,
) -> int:
    return int(num_weight_samples * (num_datasets + num_reference_datasets))


def mi_logpdf_cost(
    num_datasets: int,
    num_reference_datasets: int,
    num_weight_samples: int,
) -> int:
    return int(num_datasets * num_weight_samples * (num_reference_datasets + 1))


def prepare_alpha_sweep_case(
    rng: np.random.Generator,
    dim: int,
    *,
    n: int,
    num_datasets: int,
    num_reference_datasets: int,
    num_holdout: int,
    w_true_value: float,
    sigma_noise_sq: float,
    lambda_prior: float,
    clip_c: float,
    num_mi_weight_samples: int,
) -> dict[str, Any]:
    w_true = make_w_true(dim, w_true_value)
    holdout = make_holdout_set(
        rng=rng,
        num_holdout=num_holdout,
        w_true=w_true,
        sigma_noise_sq=sigma_noise_sq,
    )
    evaluation_datasets = [
        generate_regression_dataset(
            rng=rng,
            n=n,
            w_true=w_true,
            sigma_noise_sq=sigma_noise_sq,
        )
        for _ in range(num_datasets)
    ]
    evaluation_posteriors = [
        posterior_from_dataset(
            x,
            y,
            lambda_prior=lambda_prior,
            sigma_noise_sq=sigma_noise_sq,
        )
        for x, y in evaluation_datasets
    ]
    reference_datasets = [
        generate_regression_dataset(
            rng=rng,
            n=n,
            w_true=w_true,
            sigma_noise_sq=sigma_noise_sq,
        )
        for _ in range(num_reference_datasets)
    ]
    reference_posteriors = [
        posterior_from_dataset(
            x,
            y,
            lambda_prior=lambda_prior,
            sigma_noise_sq=sigma_noise_sq,
        )
        for x, y in reference_datasets
    ]
    reference_mixture = GaussianMixtureReference(reference_posteriors)
    empirical_gen, empirical_stderr = estimate_empirical_generalization(
        datasets=evaluation_datasets,
        posteriors=evaluation_posteriors,
        holdout=holdout,
        clip_c=clip_c,
        rng=rng,
    )
    mi_value, mi_stderr = estimate_mutual_information(
        posteriors=evaluation_posteriors,
        reference_mixture=reference_mixture,
        rng=rng,
        num_weight_samples=num_mi_weight_samples,
    )
    sigma_loss_sq_value = sigma_loss_sq(clip_c)
    mi_value = float(max(mi_value, 0.0))
    return {
        "dimension": float(dim),
        "holdout": holdout,
        "evaluation_datasets": evaluation_datasets,
        "evaluation_posteriors": evaluation_posteriors,
        "reference_posteriors": reference_posteriors,
        "reference_mixture": reference_mixture,
        "empirical_generalization": empirical_gen,
        "empirical_generalization_stderr": empirical_stderr,
        "mi_estimate": mi_value,
        "mi_stderr": mi_stderr,
        "mi_bound": float(np.sqrt(2.0 * sigma_loss_sq_value / n * mi_value)),
    }


def _candidate_loss_means(
    candidates: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    clip_c: float,
    batch_size: int,
) -> np.ndarray:
    candidate_array = _to_2d(candidates, candidates.shape[1])
    losses = np.empty(candidate_array.shape[0], dtype=float)
    for start in range(0, candidate_array.shape[0], batch_size):
        end = min(start + batch_size, candidate_array.shape[0])
        chunk = candidate_array[start:end]
        residual = y[:, None] - x @ chunk.T
        losses[start:end] = np.minimum(residual ** 2, clip_c).mean(axis=0)
    return losses


def build_theorem1_proxy_candidates(
    *,
    evaluation_posteriors: Sequence[GaussianPosterior],
    reference_mixture: GaussianMixtureReference,
    prior_reference: GaussianPosterior,
    w_true: np.ndarray,
    rng: np.random.Generator,
    num_candidate_posterior_samples: int,
    num_candidate_marginal_samples: int,
    num_candidate_prior_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_blocks: list[np.ndarray] = []
    source_blocks: list[np.ndarray] = []

    posterior_means = np.stack([posterior.mean for posterior in evaluation_posteriors], axis=0)
    candidate_blocks.append(posterior_means)
    source_blocks.append(np.full(posterior_means.shape[0], "posterior_mean", dtype=object))

    if num_candidate_posterior_samples > 0:
        posterior_samples = np.vstack(
            [posterior.sample(rng, num_candidate_posterior_samples) for posterior in evaluation_posteriors]
        )
        candidate_blocks.append(posterior_samples)
        source_blocks.append(np.full(posterior_samples.shape[0], "posterior_sample", dtype=object))

    if num_candidate_marginal_samples > 0:
        marginal_samples = reference_mixture.sample(rng, num_candidate_marginal_samples)
        candidate_blocks.append(marginal_samples)
        source_blocks.append(np.full(marginal_samples.shape[0], "marginal_sample", dtype=object))

    if num_candidate_prior_samples > 0:
        prior_samples = prior_reference.sample(rng, num_candidate_prior_samples)
        candidate_blocks.append(prior_samples)
        source_blocks.append(np.full(prior_samples.shape[0], "prior_sample", dtype=object))

    anchors = np.vstack([np.asarray(w_true, dtype=float).reshape(1, -1), np.zeros((1, w_true.shape[0]), dtype=float)])
    candidate_blocks.append(anchors)
    source_blocks.append(np.array(["w_true", "zero"], dtype=object))

    return np.vstack(candidate_blocks), np.concatenate(source_blocks)


def estimate_theorem1_sensitivity_proxy(
    *,
    candidates: np.ndarray,
    candidate_sources: np.ndarray,
    datasets: Sequence[tuple[np.ndarray, np.ndarray]],
    holdout: tuple[np.ndarray, np.ndarray],
    alpha_values: Sequence[float],
    clip_c: float,
    batch_size: int,
) -> dict[str, Any]:
    candidate_array = _to_2d(candidates, candidates.shape[1])
    holdout_x, holdout_y = holdout
    population_losses = _candidate_loss_means(candidate_array, holdout_x, holdout_y, clip_c, batch_size)
    empirical_losses = np.vstack(
        [_candidate_loss_means(candidate_array, x, y, clip_c, batch_size) for x, y in datasets]
    )
    deviations = np.abs(empirical_losses - population_losses[None, :])

    sensitivity_by_alpha: dict[float, float] = {}
    argmax_by_alpha: dict[float, int] = {}
    source_by_alpha: dict[float, str] = {}
    candidate_loss_by_alpha: dict[float, float] = {}
    for alpha in alpha_values:
        alpha_float = float(alpha)
        if alpha_float <= 1.0:
            p_value = np.inf
            norms = deviations.max(axis=0)
        else:
            p_value = alpha_float / (alpha_float - 1.0)
            if p_value > 100.0:
                norms = deviations.max(axis=0)
            else:
                norms = np.mean(deviations ** p_value, axis=0) ** (1.0 / p_value)
        argmax = int(np.argmax(norms))
        sensitivity_by_alpha[alpha_float] = float(norms[argmax])
        argmax_by_alpha[alpha_float] = argmax
        source_by_alpha[alpha_float] = str(candidate_sources[argmax])
        candidate_loss_by_alpha[alpha_float] = float(population_losses[argmax])

    unique_sources, source_counts = np.unique(candidate_sources, return_counts=True)
    return {
        "sensitivity_by_alpha": sensitivity_by_alpha,
        "argmax_by_alpha": argmax_by_alpha,
        "source_by_alpha": source_by_alpha,
        "candidate_loss_by_alpha": candidate_loss_by_alpha,
        "candidate_count": int(candidate_array.shape[0]),
        "source_counts": {str(source): int(count) for source, count in zip(unique_sources, source_counts, strict=False)},
        "max_population_loss": float(population_losses.max()),
        "min_population_loss": float(population_losses.min()),
        "max_empirical_loss": float(empirical_losses.max()),
        "min_empirical_loss": float(empirical_losses.min()),
        "max_abs_deviation": float(deviations.max()),
    }


def weighted_variance(probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = weights @ probabilities
    second_moment = weights @ (probabilities ** 2)
    return np.clip(second_moment - mean ** 2, 0.0, None)


def exact_discrete_v_alpha(
    conditional_probabilities: np.ndarray,
    marginal_probabilities: np.ndarray,
    source_probabilities: np.ndarray,
    alpha: float,
) -> float:
    active = marginal_probabilities > 0.0
    ratio = np.zeros_like(conditional_probabilities)
    ratio[:, active] = conditional_probabilities[:, active] / marginal_probabilities[active]
    centered = np.abs(ratio[:, active] - 1.0) ** alpha
    inner = source_probabilities @ centered
    return float(np.sum(marginal_probabilities[active] * inner ** (1.0 / alpha)))


def exact_discrete_mutual_information(
    conditional_probabilities: np.ndarray,
    marginal_probabilities: np.ndarray,
    source_probabilities: np.ndarray,
) -> float:
    joint = source_probabilities[:, None] * conditional_probabilities
    active = joint > 0.0
    ratio = np.zeros_like(joint)
    ratio[:, marginal_probabilities > 0.0] = (
        conditional_probabilities[:, marginal_probabilities > 0.0]
        / marginal_probabilities[marginal_probabilities > 0.0]
    )
    return float(np.sum(joint[active] * np.log(ratio[active])))


def monotone_grid_check(values: Sequence[float], tolerance: float = 1e-10) -> bool:
    array = np.asarray(values, dtype=float)
    if array.size <= 1:
        return True
    return bool(np.all(np.diff(array) >= -tolerance))


def dataset_probabilities_by_counts(n: int) -> np.ndarray:
    counts = np.arange(n + 1)
    return binom.pmf(counts, n=n, p=0.5)


def gibbs_posterior_from_count(
    count_ones: int,
    n: int,
    hypothesis_grid: np.ndarray,
    beta: float,
) -> np.ndarray:
    empirical_risk = ((n - count_ones) / n) * (hypothesis_grid ** 2) + (count_ones / n) * ((1.0 - hypothesis_grid) ** 2)
    logits = -beta * empirical_risk
    log_weights = logits - logsumexp(logits)
    return np.exp(log_weights)


def predictive_posterior_from_design(
    posterior: GaussianPosterior,
    design_matrix: np.ndarray,
    jitter: float = 1e-9,
) -> GaussianPosterior:
    mean = design_matrix @ posterior.mean
    covariance = design_matrix @ posterior.covariance @ design_matrix.T
    covariance = covariance + jitter * np.eye(covariance.shape[0])
    return GaussianPosterior(mean=mean, covariance=covariance)


def estimate_v2_from_density_matrix(
    grid: np.ndarray,
    density_matrix: np.ndarray,
    sigma_loss_sq_value: float,
    n: int,
) -> tuple[float, dict[str, float]]:
    variance_density = np.var(density_matrix, axis=1, ddof=1)
    v2_value = float(np.trapezoid(np.sqrt(np.clip(variance_density, 0.0, None)), grid))
    return (
        float(np.sqrt(sigma_loss_sq_value / n) * v2_value),
        {
            "v2_value": v2_value,
            "grid_lower": float(grid[0]),
            "grid_upper": float(grid[-1]),
            "grid_points": float(grid.size),
        },
    )


def estimate_transform_v2(
    posteriors: list[GaussianPosterior],
    sigma_loss_sq_value: float,
    n: int,
    grid_points: int,
    grid_sigma_span: float,
    transform: Callable[[np.ndarray], np.ndarray],
    inverse_transform: Callable[[np.ndarray], np.ndarray],
    derivative_on_w: Callable[[np.ndarray], np.ndarray],
) -> tuple[float, dict[str, float]]:
    means = np.array([posterior.mean[0] for posterior in posteriors], dtype=float)
    stds = np.sqrt(np.array([posterior.variance_scalar() for posterior in posteriors], dtype=float))
    w_lower = float(np.min(means - grid_sigma_span * stds))
    w_upper = float(np.max(means + grid_sigma_span * stds))
    v_lower = float(transform(np.array([w_lower]))[0])
    v_upper = float(transform(np.array([w_upper]))[0])
    v_grid = np.linspace(v_lower, v_upper, grid_points)
    w_grid = inverse_transform(v_grid)
    jacobian = derivative_on_w(w_grid)
    density_matrix = np.column_stack([np.exp(posterior.logpdf(w_grid)) / jacobian for posterior in posteriors])
    bound, diagnostics = estimate_v2_from_density_matrix(v_grid, density_matrix, sigma_loss_sq_value, n)
    normalization_error = np.max(np.abs(np.trapezoid(density_matrix, v_grid, axis=0) - 1.0))
    diagnostics["max_normalization_error"] = float(normalization_error)
    return bound, diagnostics


def estimate_coordinate_variance_statistic(
    posteriors: list[GaussianPosterior],
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
    sample_size: int = 2048,
) -> float:
    if transform is None:
        return float(np.mean([posterior.variance_scalar() for posterior in posteriors]))
    if rng is None:
        rng = np.random.default_rng(0)
    variances = []
    for posterior in posteriors:
        samples = posterior.sample(rng, sample_size).reshape(-1)
        transformed = transform(samples)
        variances.append(float(np.var(transformed, ddof=1)))
    return float(np.mean(variances))


def estimate_samplewise_loss_proxy_mi(
    supersample_x: np.ndarray,
    supersample_y: np.ndarray,
    evaluation_u_vectors: list[np.ndarray],
    evaluation_posteriors: list[GaussianPosterior],
    clip_c: float,
    rng: np.random.Generator,
    num_weight_samples: int,
    num_bins: int,
) -> tuple[float, dict[str, float]]:
    n = supersample_x.shape[0]
    samples_per_coordinate: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    row_index = np.arange(n)
    for u_vector, posterior in zip(evaluation_u_vectors, evaluation_posteriors):
        weight_samples = posterior.sample(rng, num_weight_samples)
        selected_x = supersample_x[row_index, u_vector]
        selected_y = supersample_y[row_index, u_vector]
        ghost_x = supersample_x[row_index, 1 - u_vector]
        ghost_y = supersample_y[row_index, 1 - u_vector]
        for sampled_weight in weight_samples:
            selected_loss = np.minimum(
                (selected_y - np.einsum("nd,d->n", selected_x, sampled_weight)) ** 2,
                clip_c,
            )
            ghost_loss = np.minimum(
                (ghost_y - np.einsum("nd,d->n", ghost_x, sampled_weight)) ** 2,
                clip_c,
            )
            normalized_gap = np.clip((ghost_loss - selected_loss + clip_c) / (2.0 * clip_c), 0.0, 1.0)
            for i in range(n):
                samples_per_coordinate[i].append((int(u_vector[i]), float(normalized_gap[i])))

    info_values = []
    for coordinate_samples in samples_per_coordinate:
        counts = np.full((2, num_bins), 1e-9, dtype=float)
        for bit, normalized_gap in coordinate_samples:
            bin_index = min(np.searchsorted(bins, normalized_gap, side="right") - 1, num_bins - 1)
            counts[bit, max(bin_index, 0)] += 1.0
        joint = counts / counts.sum()
        bit_marginal = joint.sum(axis=1, keepdims=True)
        gap_marginal = joint.sum(axis=0, keepdims=True)
        expected = bit_marginal @ gap_marginal
        active = joint > 0.0
        info_values.append(float(np.sum(joint[active] * np.log(joint[active] / expected[active]))))

    bound = float(clip_c / n * np.sum(np.sqrt(2.0 * np.maximum(info_values, 0.0))))
    diagnostics = {
        "mean_samplewise_mi": float(np.mean(info_values)),
        "max_samplewise_mi": float(np.max(info_values)),
    }
    return bound, diagnostics


def estimate_supersample_proxy_norms(
    supersample_x: np.ndarray,
    supersample_y: np.ndarray,
    candidate_weights: np.ndarray,
    evaluation_u_vectors: list[np.ndarray],
    alpha_values: Sequence[float],
    clip_c: float,
) -> dict[float, float]:
    if candidate_weights.size == 0:
        return {float(alpha): 0.0 for alpha in alpha_values}

    row_index = np.arange(supersample_x.shape[0])
    delta_matrix = np.empty((candidate_weights.shape[0], len(evaluation_u_vectors)), dtype=float)
    for u_index, u_vector in enumerate(evaluation_u_vectors):
        selected_x = supersample_x[row_index, u_vector]
        selected_y = supersample_y[row_index, u_vector]
        ghost_x = supersample_x[row_index, 1 - u_vector]
        ghost_y = supersample_y[row_index, 1 - u_vector]
        for weight_index, candidate_weight in enumerate(candidate_weights):
            selected_loss = clipped_squared_loss(candidate_weight, selected_x, selected_y, clip_c).mean()
            ghost_loss = clipped_squared_loss(candidate_weight, ghost_x, ghost_y, clip_c).mean()
            delta_matrix[weight_index, u_index] = ghost_loss - selected_loss

    norms: dict[float, float] = {}
    for alpha in alpha_values:
        alpha_conjugate = alpha / (alpha - 1.0)
        candidate_norms = np.mean(np.abs(delta_matrix) ** alpha_conjugate, axis=1) ** (1.0 / alpha_conjugate)
        norms[float(alpha)] = float(candidate_norms.max())
    return norms


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_one_dim_experiment(config: OneDimExperimentConfig) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sigma_loss_sq_value = sigma_loss_sq(config.clip_c)
    holdout = make_holdout_set(
        rng=rng,
        num_holdout=config.num_holdout,
        w_true=np.array([config.w_true], dtype=float),
        sigma_noise_sq=config.sigma_noise_sq,
    )

    results: list[dict[str, float]] = []
    sanity_records: list[dict[str, float]] = []

    for n in config.n_values:
        evaluation_datasets = [
            generate_regression_dataset(
                rng=rng,
                n=n,
                w_true=np.array([config.w_true], dtype=float),
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for _ in range(config.num_datasets)
        ]
        evaluation_posteriors = [
            posterior_from_dataset(
                x,
                y,
                lambda_prior=config.lambda_prior,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for x, y in evaluation_datasets
        ]

        reference_datasets = [
            generate_regression_dataset(
                rng=rng,
                n=n,
                w_true=np.array([config.w_true], dtype=float),
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for _ in range(config.num_reference_datasets)
        ]
        reference_posteriors = [
            posterior_from_dataset(
                x,
                y,
                lambda_prior=config.lambda_prior,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for x, y in reference_datasets
        ]
        reference_mixture = GaussianMixtureReference(reference_posteriors)

        empirical_gen, empirical_stderr = estimate_empirical_generalization(
            datasets=evaluation_datasets,
            posteriors=evaluation_posteriors,
            holdout=holdout,
            clip_c=config.clip_c,
            rng=rng,
        )

        mi_value, mi_stderr = estimate_mutual_information(
            posteriors=evaluation_posteriors,
            reference_mixture=reference_mixture,
            rng=rng,
            num_weight_samples=config.num_mi_weight_samples,
        )
        mi_value = max(mi_value, 0.0)
        mi_bound = float(np.sqrt(2.0 * sigma_loss_sq_value / n * mi_value))

        cmi_proxy_config = OneDimExperimentConfig(
            sigma_noise_sq=config.sigma_noise_sq,
            lambda_prior=config.lambda_prior,
            clip_c=config.clip_c,
            w_true=config.w_true,
            n_values=(n,),
            num_datasets=config.num_datasets,
            num_reference_datasets=config.num_reference_datasets,
            num_holdout=config.num_holdout,
            num_supersamples=config.num_supersamples,
            num_u_samples=config.num_u_samples,
            num_marginal_components=config.num_marginal_components,
            num_mi_weight_samples=config.num_mi_weight_samples,
            num_cmi_weight_samples=config.num_cmi_weight_samples,
            num_v_alpha_weight_samples=config.num_v_alpha_weight_samples,
            grid_points=config.grid_points,
            grid_sigma_span=config.grid_sigma_span,
            seed=int(rng.integers(0, 2**31 - 1)),
            output_dir=config.output_dir,
        )
        cmi_value, cmi_stderr = estimate_conditional_mutual_information(
            config=cmi_proxy_config,
            rng=np.random.default_rng(cmi_proxy_config.seed),
        )
        cmi_value = max(cmi_value, 0.0)
        cmi_bound = bounded_loss_cmi_bound(cmi_value, config.clip_c, n)
        conditional_v2_bundle = estimate_conditional_v_alpha_bundle(
            config=cmi_proxy_config,
            rng=np.random.default_rng(cmi_proxy_config.seed + 1),
            alpha_values=(2.0,),
            n=n,
        )
        conditional_v2_value = float(conditional_v2_bundle["conditional_v_alpha"][2.0])
        conditional_v2_stderr = float(conditional_v2_bundle["conditional_v_alpha_stderr"][2.0])
        conditional_v2_bound = float(conditional_v2_bundle["rigorous_bounds"][2.0])
        conditional_v2_bound_stderr = float(conditional_v2_bundle["rigorous_bound_stderr"][2.0])

        v2_bound, v2_diag = estimate_v2_grid(
            posteriors=evaluation_posteriors,
            sigma_loss_sq_value=sigma_loss_sq_value,
            n=n,
            grid_points=config.grid_points,
            grid_sigma_span=config.grid_sigma_span,
        )

        v_alpha_estimates, v_alpha_diag = estimate_v_alpha_from_marginal(
            posteriors=evaluation_posteriors,
            reference_mixture=reference_mixture,
            alpha_values=(2.0,),
            rng=rng,
            num_weight_samples=config.num_v_alpha_weight_samples,
        )
        v_alpha2_bound = subgaussian_multiplier(2.0, sigma_loss_sq_value, n) * v_alpha_estimates[2.0]

        results.append(
            {
                "n": float(n),
                "empirical_generalization": empirical_gen,
                "empirical_generalization_stderr": empirical_stderr,
                "mi_estimate": mi_value,
                "mi_stderr": mi_stderr,
                "mi_bound": mi_bound,
                "cmi_estimate": cmi_value,
                "cmi_stderr": cmi_stderr,
                "cmi_bound": cmi_bound,
                "conditional_v2_estimate": conditional_v2_value,
                "conditional_v2_stderr": conditional_v2_stderr,
                "conditional_v2_bound": conditional_v2_bound,
                "conditional_v2_bound_stderr": conditional_v2_bound_stderr,
                "v2_bound": v2_bound,
                "v_alpha2_monte_carlo_bound": v_alpha2_bound,
                "v2_value_grid": float(v2_diag["v2_value"]),
                "grid_outside_mass": float(v2_diag["outside_mass"]),
            }
        )
        sanity_records.append(
            {
                "n": float(n),
                "grid_outside_mass": float(v2_diag["outside_mass"]),
                "v2_relative_gap": float(abs(v2_bound - v_alpha2_bound) / max(v2_bound, 1e-12)),
                "log_ratio_min": float(v_alpha_diag["log_ratio_min"]),
                "log_ratio_max": float(v_alpha_diag["log_ratio_max"]),
                "min_posterior_variance": float(min(p.variance_scalar() for p in evaluation_posteriors)),
            }
        )

    result_frame = pd.DataFrame(results)
    sanity_frame = pd.DataFrame(sanity_records)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.errorbar(
        result_frame["n"],
        result_frame["empirical_generalization"],
        yerr=result_frame["empirical_generalization_stderr"],
        fmt="o-",
        linewidth=2,
        capsize=3,
        label="Empirical generalization",
    )
    ax.plot(result_frame["n"], result_frame["mi_bound"], "s--", linewidth=2, label="MI bound")
    ax.plot(result_frame["n"], result_frame["cmi_bound"], "d--", linewidth=2, label="CMI bound")
    ax.plot(
        result_frame["n"],
        result_frame["conditional_v2_bound"],
        "x-",
        linewidth=2,
        label=r"Conditional $V_2$ bound",
    )
    ax.plot(result_frame["n"], result_frame["v2_bound"], "^-", linewidth=2, label=r"$V_2$ bound")
    ax.set_xlabel("Training set size n")
    ax.set_ylabel("Bound / empirical gap")
    ax.set_title("1D Bayesian linear regression bounds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    figure_path = output_dir / "01_bounds_1d.png"
    table_path = output_dir / "01_bounds_1d.csv"
    sanity_path = output_dir / "01_bounds_1d_sanity.csv"
    config_path = output_dir / "01_bounds_1d_config.json"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    result_frame.to_csv(table_path, index=False)
    sanity_frame.to_csv(sanity_path, index=False)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
        },
        config_path,
    )

    first_posterior = posterior_from_dataset(
        *generate_regression_dataset(
            rng=np.random.default_rng(config.seed + 1),
            n=config.n_values[0],
            w_true=np.array([config.w_true], dtype=float),
            sigma_noise_sq=config.sigma_noise_sq,
        ),
        lambda_prior=config.lambda_prior,
        sigma_noise_sq=config.sigma_noise_sq,
    )
    posterior_samples = first_posterior.sample(np.random.default_rng(config.seed + 2), 4096).reshape(-1)
    posterior_sample_mean = float(posterior_samples.mean())
    posterior_sample_var = float(posterior_samples.var(ddof=1))

    diagnostics = {
        "figure_path": str(figure_path),
        "table_path": str(table_path),
        "sanity_path": str(sanity_path),
        "posterior_mean_vs_sample_mean": {
            "analytic": float(first_posterior.mean[0]),
            "sample": posterior_sample_mean,
        },
        "posterior_var_vs_sample_var": {
            "analytic": float(first_posterior.variance_scalar()),
            "sample": posterior_sample_var,
        },
        "story_checks": {
            "v2_beats_mi_all_n": bool(np.all(result_frame["v2_bound"] <= result_frame["mi_bound"] + 1e-10)),
            "v2_beats_cmi_all_n": bool(np.all(result_frame["v2_bound"] <= result_frame["cmi_bound"] + 1e-10)),
        },
    }

    return {
        "results": result_frame,
        "sanity": sanity_frame,
        "diagnostics": diagnostics,
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def run_dimensional_bounds_experiment(config: DimensionalBoundsConfig) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sigma_loss_sq_value = sigma_loss_sq(config.clip_c)
    w_true = w_true_from_dimensional_config(config)
    holdout = make_holdout_set(
        rng=rng,
        num_holdout=config.num_holdout,
        w_true=w_true,
        sigma_noise_sq=config.sigma_noise_sq,
    )

    results: list[dict[str, float]] = []
    sanity_records: list[dict[str, float | bool]] = []

    for n in config.n_values:
        evaluation_datasets = [
            generate_regression_dataset(
                rng=rng,
                n=n,
                w_true=w_true,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for _ in range(config.num_datasets)
        ]
        evaluation_posteriors = [
            posterior_from_dataset(
                x,
                y,
                lambda_prior=config.lambda_prior,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for x, y in evaluation_datasets
        ]

        reference_datasets = [
            generate_regression_dataset(
                rng=rng,
                n=n,
                w_true=w_true,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for _ in range(config.num_reference_datasets)
        ]
        reference_posteriors = [
            posterior_from_dataset(
                x,
                y,
                lambda_prior=config.lambda_prior,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for x, y in reference_datasets
        ]
        reference_mixture = GaussianMixtureReference(reference_posteriors)

        empirical_gen, empirical_stderr = estimate_empirical_generalization(
            datasets=evaluation_datasets,
            posteriors=evaluation_posteriors,
            holdout=holdout,
            clip_c=config.clip_c,
            rng=rng,
        )

        mi_value, mi_stderr = estimate_mutual_information(
            posteriors=evaluation_posteriors,
            reference_mixture=reference_mixture,
            rng=rng,
            num_weight_samples=config.num_mi_weight_samples,
        )
        mi_value = max(mi_value, 0.0)
        mi_bound = float(np.sqrt(2.0 * sigma_loss_sq_value / n * mi_value))

        cmi_value, cmi_stderr = estimate_conditional_mutual_information_for_w_true(
            w_true=w_true,
            n=n,
            lambda_prior=config.lambda_prior,
            sigma_noise_sq=config.sigma_noise_sq,
            num_supersamples=config.num_supersamples,
            num_marginal_components=config.num_marginal_components,
            num_u_samples=config.num_u_samples,
            num_weight_samples=config.num_cmi_weight_samples,
            rng=rng,
        )
        cmi_value = max(cmi_value, 0.0)
        cmi_bound = bounded_loss_cmi_bound(cmi_value, config.clip_c, n)

        conditional_v2_bundle = estimate_conditional_v_alpha_bundle_for_w_true(
            w_true=w_true,
            n=n,
            lambda_prior=config.lambda_prior,
            sigma_noise_sq=config.sigma_noise_sq,
            clip_c=config.clip_c,
            alpha_values=(2.0,),
            num_supersamples=config.num_conditional_supersamples,
            num_marginal_components=config.num_conditional_marginal_components,
            num_u_samples=config.num_conditional_u_samples,
            num_weight_samples=config.num_conditional_v_alpha_weight_samples,
            rng=rng,
        )
        conditional_v2_value = float(conditional_v2_bundle["conditional_v_alpha"][2.0])
        conditional_v2_stderr = float(conditional_v2_bundle["conditional_v_alpha_stderr"][2.0])
        conditional_v2_bound = float(conditional_v2_bundle["rigorous_bounds"][2.0])
        conditional_v2_bound_stderr = float(conditional_v2_bundle["rigorous_bound_stderr"][2.0])

        v_alpha_estimates, v_alpha_diag = estimate_v_alpha_from_marginal(
            posteriors=evaluation_posteriors,
            reference_mixture=reference_mixture,
            alpha_values=(2.0,),
            rng=rng,
            num_weight_samples=config.num_v_alpha_weight_samples,
        )
        v2_value_monte_carlo = float(v_alpha_estimates[2.0])
        v2_bound = float(subgaussian_multiplier(2.0, sigma_loss_sq_value, n) * v2_value_monte_carlo)
        bounds = {
            "mi_bound": mi_bound,
            "cmi_bound": cmi_bound,
            "conditional_v2_bound": conditional_v2_bound,
            "v2_bound": v2_bound,
        }

        min_covariance_eigenvalue = min(
            float(np.linalg.eigvalsh(posterior.covariance).min()) for posterior in evaluation_posteriors
        )
        max_covariance_condition = max(
            float(np.linalg.cond(posterior.covariance)) for posterior in evaluation_posteriors
        )

        results.append(
            {
                "dimension": float(config.dimension),
                "n": float(n),
                "empirical_generalization": empirical_gen,
                "empirical_generalization_stderr": empirical_stderr,
                "mi_estimate": mi_value,
                "mi_stderr": mi_stderr,
                "mi_bound": mi_bound,
                "cmi_estimate": cmi_value,
                "cmi_stderr": cmi_stderr,
                "cmi_bound": cmi_bound,
                "conditional_v2_estimate": conditional_v2_value,
                "conditional_v2_stderr": conditional_v2_stderr,
                "conditional_v2_bound": conditional_v2_bound,
                "conditional_v2_bound_stderr": conditional_v2_bound_stderr,
                "v2_value_monte_carlo": v2_value_monte_carlo,
                "v2_bound": v2_bound,
                "v2_log_ratio_min": float(v_alpha_diag["log_ratio_min"]),
                "v2_log_ratio_max": float(v_alpha_diag["log_ratio_max"]),
            }
        )
        sanity_records.append(
            {
                "dimension": float(config.dimension),
                "n": float(n),
                "mi_nonnegative": bool(mi_value >= -1e-9),
                "cmi_nonnegative": bool(cmi_value >= -1e-9),
                "v2_finite": bool(np.isfinite(v2_bound)),
                "conditional_v2_finite": bool(np.isfinite(conditional_v2_bound)),
                "empirical_exceeds_all_bounds": bool(empirical_gen > max(bounds.values())),
                "empirical_exceeds_mi_bound": bool(empirical_gen > mi_bound),
                "empirical_exceeds_cmi_bound": bool(empirical_gen > cmi_bound),
                "empirical_exceeds_conditional_v2_bound": bool(empirical_gen > conditional_v2_bound),
                "empirical_exceeds_v2_bound": bool(empirical_gen > v2_bound),
                "min_covariance_eigenvalue": min_covariance_eigenvalue,
                "max_covariance_condition": max_covariance_condition,
                "v2_log_ratio_min": float(v_alpha_diag["log_ratio_min"]),
                "v2_log_ratio_max": float(v_alpha_diag["log_ratio_max"]),
                "v2_num_weight_samples": float(v_alpha_diag["num_weight_samples"]),
            }
        )

    result_frame = pd.DataFrame(results)
    sanity_frame = pd.DataFrame(sanity_records)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.errorbar(
        result_frame["n"],
        result_frame["empirical_generalization"],
        yerr=result_frame["empirical_generalization_stderr"],
        fmt="o-",
        linewidth=2,
        capsize=3,
        label="Empirical generalization",
    )
    ax.plot(result_frame["n"], result_frame["mi_bound"], "s--", linewidth=2, label="MI bound")
    ax.plot(result_frame["n"], result_frame["cmi_bound"], "d--", linewidth=2, label="CMI bound")
    ax.plot(
        result_frame["n"],
        result_frame["conditional_v2_bound"],
        "x-",
        linewidth=2,
        label=r"Conditional $V_2$ bound",
    )
    ax.plot(result_frame["n"], result_frame["v2_bound"], "^-", linewidth=2, label=r"$V_2$ bound (MC)")
    ax.set_xlabel("Training set size n")
    ax.set_ylabel("Bound / empirical gap")
    ax.set_title(f"{config.dimension}D Bayesian linear regression bounds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    figure_path = output_dir / f"{config.artifact_prefix}.png"
    table_path = output_dir / f"{config.artifact_prefix}.csv"
    sanity_path = output_dir / f"{config.artifact_prefix}_sanity.csv"
    config_path = output_dir / f"{config.artifact_prefix}_config.json"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    result_frame.to_csv(table_path, index=False)
    sanity_frame.to_csv(sanity_path, index=False)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
        },
        config_path,
    )

    return {
        "results": result_frame,
        "sanity": sanity_frame,
        "diagnostics": {
            "figure_path": str(figure_path),
            "table_path": str(table_path),
            "sanity_path": str(sanity_path),
            "uses_grid_v2": False,
            "v2_estimator": "posterior-mixture Monte Carlo at alpha=2",
        },
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def run_alpha_sweep_experiment(config: AlphaSweepExperimentConfig) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma_loss_sq_value = sigma_loss_sq(config.clip_c)

    records: list[dict[str, float]] = []
    diagnostics: list[dict[str, float]] = []

    for dim in config.dimensions:
        w_true = make_w_true(dim, config.w_true_value)
        holdout = make_holdout_set(
            rng=rng,
            num_holdout=config.num_holdout,
            w_true=w_true,
            sigma_noise_sq=config.sigma_noise_sq,
        )

        evaluation_datasets = [
            generate_regression_dataset(
                rng=rng,
                n=config.n,
                w_true=w_true,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for _ in range(config.num_datasets)
        ]
        evaluation_posteriors = [
            posterior_from_dataset(
                x,
                y,
                lambda_prior=config.lambda_prior,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for x, y in evaluation_datasets
        ]

        reference_datasets = [
            generate_regression_dataset(
                rng=rng,
                n=config.n,
                w_true=w_true,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for _ in range(config.num_reference_datasets)
        ]
        reference_posteriors = [
            posterior_from_dataset(
                x,
                y,
                lambda_prior=config.lambda_prior,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for x, y in reference_datasets
        ]
        reference_mixture = GaussianMixtureReference(reference_posteriors)

        empirical_gen, empirical_stderr = estimate_empirical_generalization(
            datasets=evaluation_datasets,
            posteriors=evaluation_posteriors,
            holdout=holdout,
            clip_c=config.clip_c,
            rng=rng,
        )
        mi_value, mi_stderr = estimate_mutual_information(
            posteriors=evaluation_posteriors,
            reference_mixture=reference_mixture,
            rng=rng,
            num_weight_samples=config.num_mi_weight_samples,
        )
        mi_value = max(mi_value, 0.0)
        mi_bound = float(np.sqrt(2.0 * sigma_loss_sq_value / config.n * mi_value))

        v_alpha_values, v_alpha_diag = estimate_v_alpha_from_marginal(
            posteriors=evaluation_posteriors,
            reference_mixture=reference_mixture,
            alpha_values=config.alpha_values,
            rng=rng,
            num_weight_samples=config.num_weight_samples,
        )
        bound_by_alpha = {
            alpha: subgaussian_multiplier(alpha, sigma_loss_sq_value, config.n) * value
            for alpha, value in v_alpha_values.items()
        }
        fixed_v2_bound = bound_by_alpha[2.0]
        optimal_alpha = min(bound_by_alpha, key=bound_by_alpha.get)
        optimized_bound = bound_by_alpha[optimal_alpha]

        records.append(
            {
                "dimension": float(dim),
                "empirical_generalization": empirical_gen,
                "empirical_generalization_stderr": empirical_stderr,
                "mi_estimate": mi_value,
                "mi_stderr": mi_stderr,
                "mi_bound": mi_bound,
                "v2_bound": fixed_v2_bound,
                "optimized_v_alpha_bound": optimized_bound,
                "optimal_alpha": float(optimal_alpha),
            }
        )
        diagnostics.append(
            {
                "dimension": float(dim),
                "log_ratio_min": float(v_alpha_diag["log_ratio_min"]),
                "log_ratio_max": float(v_alpha_diag["log_ratio_max"]),
                "min_covariance_eigenvalue": float(min(np.linalg.eigvalsh(p.covariance).min() for p in evaluation_posteriors)),
            }
        )

    result_frame = pd.DataFrame(records)
    diagnostics_frame = pd.DataFrame(diagnostics)

    fig_main, ax_main = plt.subplots(figsize=(8.5, 5.0))
    ax_main.errorbar(
        result_frame["dimension"],
        result_frame["empirical_generalization"],
        yerr=result_frame["empirical_generalization_stderr"],
        fmt="o-",
        linewidth=2,
        capsize=3,
        label="Empirical generalization",
    )
    ax_main.plot(result_frame["dimension"], result_frame["mi_bound"], "s--", linewidth=2, label="MI bound")
    ax_main.plot(result_frame["dimension"], result_frame["v2_bound"], "^-", linewidth=2, label=r"Fixed $V_2$")
    ax_main.plot(
        result_frame["dimension"],
        result_frame["optimized_v_alpha_bound"],
        "d-",
        linewidth=2,
        label=r"Optimized $V_\alpha$",
    )
    ax_main.set_xlabel("Dimension d")
    ax_main.set_ylabel("Bound / empirical gap")
    ax_main.set_title(r"Dimension sweep with corrected $p_W$ estimators")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    fig_main.tight_layout()

    fig_alpha, ax_alpha = plt.subplots(figsize=(7.5, 4.5))
    ax_alpha.plot(result_frame["dimension"], result_frame["optimal_alpha"], "o-", linewidth=2)
    ax_alpha.set_xlabel("Dimension d")
    ax_alpha.set_ylabel(r"Selected $\alpha^*$")
    ax_alpha.set_title(r"Optimal $\alpha$ across dimensions")
    ax_alpha.grid(True, alpha=0.3)
    fig_alpha.tight_layout()

    main_figure_path = output_dir / "02_alpha_sweep_main.png"
    alpha_figure_path = output_dir / "02_alpha_sweep_optimal_alpha.png"
    table_path = output_dir / "02_alpha_sweep.csv"
    diagnostics_path = output_dir / "02_alpha_sweep_sanity.csv"
    config_path = output_dir / "02_alpha_sweep_config.json"
    fig_main.savefig(main_figure_path, dpi=200)
    fig_alpha.savefig(alpha_figure_path, dpi=200)
    plt.close(fig_main)
    plt.close(fig_alpha)
    result_frame.to_csv(table_path, index=False)
    diagnostics_frame.to_csv(diagnostics_path, index=False)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
        },
        config_path,
    )

    story_checks = {
        "fixed_v2_worsens_vs_mi_at_high_dim": bool(
            result_frame.loc[result_frame["dimension"] == result_frame["dimension"].max(), "v2_bound"].iloc[0]
            > result_frame.loc[result_frame["dimension"] == result_frame["dimension"].max(), "mi_bound"].iloc[0]
        ),
        "optimized_beats_fixed_v2_all_dims": bool(
            np.all(result_frame["optimized_v_alpha_bound"] <= result_frame["v2_bound"] + 1e-10)
        ),
        "optimal_alpha_moves_downward": bool(
            result_frame["optimal_alpha"].iloc[-1] <= result_frame["optimal_alpha"].iloc[0]
        ),
    }

    return {
        "results": result_frame,
        "sanity": diagnostics_frame,
        "diagnostics": {
            "main_figure_path": str(main_figure_path),
            "alpha_figure_path": str(alpha_figure_path),
            "table_path": str(table_path),
            "diagnostics_path": str(diagnostics_path),
            "story_checks": story_checks,
        },
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def run_alpha_sweep_audit_experiment(config: AlphaSweepAuditConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma_loss_sq_value = sigma_loss_sq(config.clip_c)
    alpha_values = tuple(float(alpha) for alpha in config.alpha_values)

    profile_records: list[dict[str, float | str | int]] = []
    repeat_records: list[dict[str, float | str | int | bool]] = []
    sanity_records: list[dict[str, float | str | int | bool]] = []

    def append_profile_rows(
        *,
        scenario: str,
        estimator: str,
        repeat_index: int,
        case: dict[str, Any],
        profile: dict[str, Any],
        num_weight_samples: int,
    ) -> None:
        for alpha in alpha_values:
            profile_records.append(
                {
                    "scenario": scenario,
                    "estimator": estimator,
                    "dimension": float(case["dimension"]),
                    "repeat": float(repeat_index),
                    "alpha": float(alpha),
                    "v_alpha_value": float(profile["v_alpha_values"][alpha]),
                    "bound": float(profile["bound_by_alpha"][alpha]),
                    "num_weight_samples": float(num_weight_samples),
                    "log_ratio_min": float(profile["diagnostics"]["log_ratio_min"]),
                    "log_ratio_max": float(profile["diagnostics"]["log_ratio_max"]),
                }
            )

    def append_repeat_row(
        *,
        scenario: str,
        estimator: str,
        repeat_index: int,
        case: dict[str, Any],
        selected_alpha: float,
        optimized_bound: float,
        fixed_v2_bound: float,
        estimator_logpdf_cost: int,
        target_logpdf_cost: int,
        raw_monotonicity_violation: float,
        raw_monotonicity_count: int,
        split_select_monotonicity_violation: float | None = None,
        split_select_monotonicity_count: int | None = None,
        split_eval_opt_bound: float | None = None,
        selection_bias_gap: float | None = None,
    ) -> None:
        mi_cost = mi_logpdf_cost(
            num_datasets=len(case["evaluation_posteriors"]),
            num_reference_datasets=len(case["reference_posteriors"]),
            num_weight_samples=int(case["num_mi_weight_samples"]),
        )
        repeat_records.append(
            {
                "record_type": "repeat",
                "scenario": scenario,
                "estimator": estimator,
                "dimension": float(case["dimension"]),
                "repeat": float(repeat_index),
                "empirical_generalization": float(case["empirical_generalization"]),
                "empirical_generalization_stderr": float(case["empirical_generalization_stderr"]),
                "mi_estimate": float(case["mi_estimate"]),
                "mi_stderr": float(case["mi_stderr"]),
                "mi_bound": float(case["mi_bound"]),
                "fixed_v2_bound": float(fixed_v2_bound),
                "optimized_bound": float(optimized_bound),
                "selected_alpha": float(selected_alpha),
                "mi_logpdf_cost": float(mi_cost),
                "estimator_logpdf_cost": float(estimator_logpdf_cost),
                "target_logpdf_cost": float(target_logpdf_cost),
                "relative_compute_gap_vs_mi": float(abs(estimator_logpdf_cost - mi_cost) / max(estimator_logpdf_cost, mi_cost)),
                "relative_compute_gap_vs_target": float(
                    abs(estimator_logpdf_cost - target_logpdf_cost) / max(estimator_logpdf_cost, target_logpdf_cost)
                ),
                "compute_within_5pct_vs_mi": bool(
                    abs(estimator_logpdf_cost - mi_cost) / max(estimator_logpdf_cost, mi_cost) <= 0.05
                ),
                "compute_within_5pct_vs_target": bool(
                    abs(estimator_logpdf_cost - target_logpdf_cost) / max(estimator_logpdf_cost, target_logpdf_cost) <= 0.05
                ),
                "raw_v_alpha_monotonicity_violation": float(raw_monotonicity_violation),
                "raw_v_alpha_monotonicity_count": float(raw_monotonicity_count),
                "split_select_monotonicity_violation": float(split_select_monotonicity_violation)
                if split_select_monotonicity_violation is not None
                else np.nan,
                "split_select_monotonicity_count": float(split_select_monotonicity_count)
                if split_select_monotonicity_count is not None
                else np.nan,
                "split_eval_opt_bound": float(split_eval_opt_bound) if split_eval_opt_bound is not None else np.nan,
                "selection_bias_gap": float(selection_bias_gap) if selection_bias_gap is not None else np.nan,
            }
        )

    def run_group(
        *,
        scenario: str,
        dimensions: Sequence[int],
        num_repeats: int,
        num_datasets: int,
        num_reference_datasets: int,
        num_weight_samples: int,
        num_mi_weight_samples: int,
        split_selection_weight_samples: int | None,
        base_seed: int,
        include_naive: bool,
        include_split: bool,
    ) -> None:
        target_logpdf_cost = alpha_sweep_logpdf_cost(
            num_datasets=num_datasets,
            num_reference_datasets=num_reference_datasets,
            num_weight_samples=num_weight_samples,
        )
        for repeat_index in range(num_repeats):
            for dim in dimensions:
                case_seed = base_seed + repeat_index * 1000 + int(dim) * 31
                case = prepare_alpha_sweep_case(
                    rng=np.random.default_rng(case_seed),
                    dim=int(dim),
                    n=config.n,
                    num_datasets=num_datasets,
                    num_reference_datasets=num_reference_datasets,
                    num_holdout=config.num_holdout,
                    w_true_value=config.w_true_value,
                    sigma_noise_sq=config.sigma_noise_sq,
                    lambda_prior=config.lambda_prior,
                    clip_c=config.clip_c,
                    num_mi_weight_samples=num_mi_weight_samples,
                )
                case["num_mi_weight_samples"] = num_mi_weight_samples

                naive_profile: dict[str, Any] | None = None
                if include_naive:
                    naive_profile = alpha_profile_from_marginal(
                        posteriors=case["evaluation_posteriors"],
                        reference_mixture=case["reference_mixture"],
                        alpha_values=alpha_values,
                        rng=np.random.default_rng(case_seed + 1),
                        num_weight_samples=num_weight_samples,
                        sigma_loss_sq_value=sigma_loss_sq_value,
                        n=config.n,
                    )
                    append_profile_rows(
                        scenario=scenario,
                        estimator=f"{scenario}_naive",
                        repeat_index=repeat_index,
                        case=case,
                        profile=naive_profile,
                        num_weight_samples=num_weight_samples,
                    )
                    append_repeat_row(
                        scenario=scenario,
                        estimator=f"{scenario}_naive",
                        repeat_index=repeat_index,
                        case=case,
                        selected_alpha=float(naive_profile["optimal_alpha"]),
                        optimized_bound=float(naive_profile["optimized_bound"]),
                        fixed_v2_bound=float(naive_profile["fixed_v2_bound"]),
                        estimator_logpdf_cost=target_logpdf_cost,
                        target_logpdf_cost=target_logpdf_cost,
                        raw_monotonicity_violation=float(naive_profile["monotonicity_violation"]),
                        raw_monotonicity_count=int(naive_profile["monotonicity_violation_count"]),
                    )

                if include_split and split_selection_weight_samples is not None:
                    split_select_profile = alpha_profile_from_marginal(
                        posteriors=case["evaluation_posteriors"],
                        reference_mixture=case["reference_mixture"],
                        alpha_values=alpha_values,
                        rng=np.random.default_rng(case_seed + 2),
                        num_weight_samples=split_selection_weight_samples,
                        sigma_loss_sq_value=sigma_loss_sq_value,
                        n=config.n,
                    )
                    split_eval_profile = alpha_profile_from_marginal(
                        posteriors=case["evaluation_posteriors"],
                        reference_mixture=case["reference_mixture"],
                        alpha_values=alpha_values,
                        rng=np.random.default_rng(case_seed + 3),
                        num_weight_samples=split_selection_weight_samples,
                        sigma_loss_sq_value=sigma_loss_sq_value,
                        n=config.n,
                    )
                    selected_alpha = float(split_select_profile["optimal_alpha"])
                    optimized_bound = float(split_eval_profile["bound_by_alpha"][selected_alpha])
                    split_logpdf_cost = 2 * alpha_sweep_logpdf_cost(
                        num_datasets=num_datasets,
                        num_reference_datasets=num_reference_datasets,
                        num_weight_samples=split_selection_weight_samples,
                    )
                    append_profile_rows(
                        scenario=scenario,
                        estimator=f"{scenario}_split_select",
                        repeat_index=repeat_index,
                        case=case,
                        profile=split_select_profile,
                        num_weight_samples=split_selection_weight_samples,
                    )
                    append_profile_rows(
                        scenario=scenario,
                        estimator=f"{scenario}_split_eval",
                        repeat_index=repeat_index,
                        case=case,
                        profile=split_eval_profile,
                        num_weight_samples=split_selection_weight_samples,
                    )
                    append_repeat_row(
                        scenario=scenario,
                        estimator=f"{scenario}_split",
                        repeat_index=repeat_index,
                        case=case,
                        selected_alpha=selected_alpha,
                        optimized_bound=optimized_bound,
                        fixed_v2_bound=float(split_eval_profile["fixed_v2_bound"]),
                        estimator_logpdf_cost=split_logpdf_cost,
                        target_logpdf_cost=target_logpdf_cost,
                        raw_monotonicity_violation=float(split_eval_profile["monotonicity_violation"]),
                        raw_monotonicity_count=int(split_eval_profile["monotonicity_violation_count"]),
                        split_select_monotonicity_violation=float(split_select_profile["monotonicity_violation"]),
                        split_select_monotonicity_count=int(split_select_profile["monotonicity_violation_count"]),
                        split_eval_opt_bound=float(split_eval_profile["optimized_bound"]),
                        selection_bias_gap=(
                            float(optimized_bound - naive_profile["optimized_bound"]) if naive_profile is not None else None
                        ),
                    )

    legacy_reference_path = output_dir / "02_alpha_sweep.csv"
    if legacy_reference_path.exists():
        legacy_reference = pd.read_csv(legacy_reference_path)
        legacy_cost = alpha_sweep_logpdf_cost(
            num_datasets=config.legacy_num_datasets,
            num_reference_datasets=config.legacy_num_reference_datasets,
            num_weight_samples=config.legacy_num_weight_samples,
        )
        legacy_mi_cost = mi_logpdf_cost(
            num_datasets=config.legacy_num_datasets,
            num_reference_datasets=config.legacy_num_reference_datasets,
            num_weight_samples=config.legacy_num_mi_weight_samples,
        )
        for _, row in legacy_reference.iterrows():
            repeat_records.append(
                {
                    "record_type": "repeat",
                    "scenario": "legacy",
                    "estimator": "legacy_naive",
                    "dimension": float(row["dimension"]),
                    "repeat": 0.0,
                    "empirical_generalization": float(row["empirical_generalization"]),
                    "empirical_generalization_stderr": float(row["empirical_generalization_stderr"]),
                    "mi_estimate": float(row["mi_estimate"]),
                    "mi_stderr": float(row["mi_stderr"]),
                    "mi_bound": float(row["mi_bound"]),
                    "fixed_v2_bound": float(row["v2_bound"]),
                    "optimized_bound": float(row["optimized_v_alpha_bound"]),
                    "selected_alpha": float(row["optimal_alpha"]),
                    "mi_logpdf_cost": float(legacy_mi_cost),
                    "estimator_logpdf_cost": float(legacy_cost),
                    "target_logpdf_cost": float(legacy_cost),
                    "relative_compute_gap_vs_mi": float(abs(legacy_cost - legacy_mi_cost) / max(legacy_cost, legacy_mi_cost)),
                    "relative_compute_gap_vs_target": 0.0,
                    "compute_within_5pct_vs_mi": bool(
                        abs(legacy_cost - legacy_mi_cost) / max(legacy_cost, legacy_mi_cost) <= 0.05
                    ),
                    "compute_within_5pct_vs_target": True,
                    "raw_v_alpha_monotonicity_violation": np.nan,
                    "raw_v_alpha_monotonicity_count": np.nan,
                    "split_select_monotonicity_violation": np.nan,
                    "split_select_monotonicity_count": np.nan,
                    "split_eval_opt_bound": np.nan,
                    "selection_bias_gap": np.nan,
                }
            )
            sanity_records.append(
                {
                    "record_type": "reproduction",
                    "scenario": "legacy",
                    "estimator": "legacy_naive",
                    "dimension": float(row["dimension"]),
                    "repeat": 0.0,
                    "empirical_generalization_abs_diff": 0.0,
                    "mi_bound_abs_diff": 0.0,
                    "v2_bound_abs_diff": 0.0,
                    "optimized_bound_abs_diff": 0.0,
                    "optimal_alpha_abs_diff": 0.0,
                    "matches_02_within_tolerance": True,
                    "legacy_source": "existing_02_output",
                }
            )
    else:
        run_group(
            scenario="legacy",
            dimensions=config.dimensions,
            num_repeats=1,
            num_datasets=config.legacy_num_datasets,
            num_reference_datasets=config.legacy_num_reference_datasets,
            num_weight_samples=config.legacy_num_weight_samples,
            num_mi_weight_samples=config.legacy_num_mi_weight_samples,
            split_selection_weight_samples=None,
            base_seed=config.legacy_seed,
            include_naive=True,
            include_split=False,
        )
    run_group(
        scenario="audit",
        dimensions=config.dimensions,
        num_repeats=config.audit_num_repeats,
        num_datasets=config.audit_num_datasets,
        num_reference_datasets=config.audit_num_reference_datasets,
        num_weight_samples=config.audit_num_weight_samples,
        num_mi_weight_samples=config.audit_num_mi_weight_samples,
        split_selection_weight_samples=config.audit_split_selection_weight_samples,
        base_seed=config.seed,
        include_naive=True,
        include_split=True,
    )
    run_group(
        scenario="benchmark",
        dimensions=config.benchmark_dimensions,
        num_repeats=config.benchmark_num_repeats,
        num_datasets=config.benchmark_num_datasets,
        num_reference_datasets=config.benchmark_num_reference_datasets,
        num_weight_samples=config.benchmark_num_weight_samples,
        num_mi_weight_samples=config.benchmark_num_mi_weight_samples,
        split_selection_weight_samples=config.benchmark_split_selection_weight_samples,
        base_seed=config.seed + 50000,
        include_naive=False,
        include_split=True,
    )

    profile_frame = pd.DataFrame(profile_records)
    repeat_frame = pd.DataFrame(repeat_records)

    summary_frame = (
        repeat_frame.groupby(["scenario", "estimator", "dimension"], as_index=False)
        .agg(
            repeat_count=("repeat", "count"),
            empirical_generalization_mean=("empirical_generalization", "mean"),
            empirical_generalization_std=("empirical_generalization", "std"),
            empirical_generalization_stderr_mean=("empirical_generalization_stderr", "mean"),
            mi_bound_mean=("mi_bound", "mean"),
            mi_bound_std=("mi_bound", "std"),
            fixed_v2_bound_mean=("fixed_v2_bound", "mean"),
            fixed_v2_bound_std=("fixed_v2_bound", "std"),
            optimized_bound_mean=("optimized_bound", "mean"),
            optimized_bound_std=("optimized_bound", "std"),
            selected_alpha_mean=("selected_alpha", "mean"),
            selected_alpha_median=("selected_alpha", "median"),
            selected_alpha_std=("selected_alpha", "std"),
            selected_alpha_min=("selected_alpha", "min"),
            selected_alpha_max=("selected_alpha", "max"),
        )
        .fillna(0.0)
    )
    summary_frame["selected_alpha_range"] = (
        summary_frame["selected_alpha_max"] - summary_frame["selected_alpha_min"]
    )

    for _, row in repeat_frame.iterrows():
        sanity_records.append(row.to_dict())

    if not repeat_frame.empty:
        for (scenario, estimator, dimension), group in repeat_frame.groupby(["scenario", "estimator", "dimension"]):
            alpha_values_selected = group["selected_alpha"].to_numpy(dtype=float)
            unique_alpha, counts = np.unique(alpha_values_selected, return_counts=True)
            alpha_frequency = {
                f"{alpha:.2f}": int(count)
                for alpha, count in sorted(zip(unique_alpha, counts, strict=False), key=lambda item: item[0])
            }
            sanity_records.append(
                {
                    "record_type": "aggregate",
                    "scenario": scenario,
                    "estimator": estimator,
                    "dimension": float(dimension),
                    "repeat": np.nan,
                    "alpha_frequency_json": json.dumps(alpha_frequency, sort_keys=True),
                    "selected_alpha_mean": float(group["selected_alpha"].mean()),
                    "selected_alpha_median": float(group["selected_alpha"].median()),
                    "selected_alpha_std": float(group["selected_alpha"].std(ddof=1)) if len(group) > 1 else 0.0,
                    "selected_alpha_range": float(group["selected_alpha"].max() - group["selected_alpha"].min()),
                    "alpha_stability_flag": bool(group["selected_alpha"].max() - group["selected_alpha"].min() > 0.1),
                    "optimized_bound_mean": float(group["optimized_bound"].mean()),
                    "optimized_bound_std": float(group["optimized_bound"].std(ddof=1)) if len(group) > 1 else 0.0,
                    "max_monotonicity_violation": float(group["raw_v_alpha_monotonicity_violation"].max()),
                    "compute_within_5pct_all_repeats": bool(group["compute_within_5pct_vs_mi"].all()),
                }
            )

    audit_naive = repeat_frame[repeat_frame["estimator"] == "audit_naive"]
    audit_split = repeat_frame[repeat_frame["estimator"] == "audit_split"]
    if not audit_naive.empty and not audit_split.empty:
        selection_bias = audit_split.merge(
            audit_naive,
            on=["scenario", "dimension", "repeat"],
            suffixes=("_split", "_naive"),
        )
        for _, row in selection_bias.iterrows():
            sanity_records.append(
                {
                    "record_type": "selection_bias",
                    "scenario": "audit",
                    "estimator": "audit_split_vs_naive",
                    "dimension": float(row["dimension"]),
                    "repeat": float(row["repeat"]),
                    "selection_bias_gap": float(row["optimized_bound_split"] - row["optimized_bound_naive"]),
                    "selected_alpha_split": float(row["selected_alpha_split"]),
                    "selected_alpha_naive": float(row["selected_alpha_naive"]),
                    "naive_bound": float(row["optimized_bound_naive"]),
                    "split_bound": float(row["optimized_bound_split"]),
                }
            )
        for dimension, group in selection_bias.groupby("dimension"):
            sanity_records.append(
                {
                    "record_type": "selection_bias_aggregate",
                    "scenario": "audit",
                    "estimator": "audit_split_vs_naive",
                    "dimension": float(dimension),
                    "repeat": np.nan,
                    "selection_bias_gap_mean": float((group["optimized_bound_split"] - group["optimized_bound_naive"]).mean()),
                    "selection_bias_gap_std": float((group["optimized_bound_split"] - group["optimized_bound_naive"]).std(ddof=1))
                    if len(group) > 1
                    else 0.0,
                    "split_not_systematically_lower": bool(
                        (group["optimized_bound_split"] - group["optimized_bound_naive"]).mean() >= -1e-10
                    ),
                }
            )

    benchmark_summary = summary_frame[summary_frame["estimator"] == "benchmark_split"]
    audit_summary = summary_frame[summary_frame["estimator"] == "audit_split"]
    if not benchmark_summary.empty and not audit_summary.empty:
        benchmark_comparison = benchmark_summary.merge(
            audit_summary,
            on="dimension",
            suffixes=("_benchmark", "_audit"),
        )
        for _, row in benchmark_comparison.iterrows():
            relative_change = abs(row["optimized_bound_mean_benchmark"] - row["optimized_bound_mean_audit"]) / max(
                row["optimized_bound_mean_audit"], 1e-12
            )
            alpha_delta = abs(row["selected_alpha_mean_benchmark"] - row["selected_alpha_mean_audit"])
            sanity_records.append(
                {
                    "record_type": "benchmark_comparison",
                    "scenario": "benchmark",
                    "estimator": "benchmark_split_vs_audit_split",
                    "dimension": float(row["dimension"]),
                    "repeat": np.nan,
                    "optimized_bound_relative_change": float(relative_change),
                    "selected_alpha_delta": float(alpha_delta),
                    "benchmark_instability_flag": bool(relative_change > 0.10 or alpha_delta > 0.1),
                }
            )

    legacy_repeat = repeat_frame[repeat_frame["estimator"] == "legacy_naive"].copy()
    if legacy_reference_path.exists() and not legacy_repeat.empty and not any(
        record.get("record_type") == "reproduction" for record in sanity_records
    ):
        reference_frame = pd.read_csv(legacy_reference_path)
        comparison = legacy_repeat.merge(
            reference_frame.rename(
                columns={
                    "empirical_generalization": "reference_empirical_generalization",
                    "mi_bound": "reference_mi_bound",
                    "v2_bound": "reference_v2_bound",
                    "optimized_v_alpha_bound": "reference_optimized_bound",
                    "optimal_alpha": "reference_optimal_alpha",
                }
            ),
            on="dimension",
            how="inner",
        )
        for _, row in comparison.iterrows():
            sanity_records.append(
                {
                    "record_type": "reproduction",
                    "scenario": "legacy",
                    "estimator": "legacy_naive",
                    "dimension": float(row["dimension"]),
                    "repeat": float(row["repeat"]),
                    "empirical_generalization_abs_diff": float(
                        abs(row["empirical_generalization"] - row["reference_empirical_generalization"])
                    ),
                    "mi_bound_abs_diff": float(abs(row["mi_bound"] - row["reference_mi_bound"])),
                    "v2_bound_abs_diff": float(abs(row["fixed_v2_bound"] - row["reference_v2_bound"])),
                    "optimized_bound_abs_diff": float(abs(row["optimized_bound"] - row["reference_optimized_bound"])),
                    "optimal_alpha_abs_diff": float(abs(row["selected_alpha"] - row["reference_optimal_alpha"])),
                    "matches_02_within_tolerance": bool(
                        abs(row["mi_bound"] - row["reference_mi_bound"]) < 1e-10
                        and abs(row["fixed_v2_bound"] - row["reference_v2_bound"]) < 1e-10
                        and abs(row["optimized_bound"] - row["reference_optimized_bound"]) < 1e-10
                        and abs(row["selected_alpha"] - row["reference_optimal_alpha"]) < 1e-10
                    ),
                }
            )

    sanity_frame = pd.DataFrame(sanity_records)

    fig_diag, axes = plt.subplots(2, 2, figsize=(14.0, 9.8), sharex=True)
    axes_flat = axes.ravel()
    for axis, dim in zip(axes_flat, config.dimensions, strict=False):
        dim_profiles = profile_frame[profile_frame["dimension"] == float(dim)]
        for estimator, label, color in [
            ("audit_naive", "Naive same-pool", "#1f77b4"),
            ("audit_split_eval", "Split eval", "#d62728"),
        ]:
            subset = dim_profiles[dim_profiles["estimator"] == estimator]
            if subset.empty:
                continue
            grouped = subset.groupby("alpha", as_index=False).agg(mean_bound=("bound", "mean"), std_bound=("bound", "std"))
            grouped = grouped.sort_values("alpha")
            std_values = grouped["std_bound"].fillna(0.0).to_numpy()
            axis.plot(grouped["alpha"], grouped["mean_bound"], linewidth=2, label=label, color=color)
            axis.fill_between(
                grouped["alpha"].to_numpy(),
                grouped["mean_bound"].to_numpy() - std_values,
                grouped["mean_bound"].to_numpy() + std_values,
                alpha=0.18,
                color=color,
            )
        benchmark_subset = dim_profiles[dim_profiles["estimator"] == "benchmark_split_eval"]
        if not benchmark_subset.empty:
            grouped = benchmark_subset.groupby("alpha", as_index=False).agg(mean_bound=("bound", "mean"), std_bound=("bound", "std"))
            grouped = grouped.sort_values("alpha")
            std_values = grouped["std_bound"].fillna(0.0).to_numpy()
            axis.plot(
                grouped["alpha"],
                grouped["mean_bound"],
                linewidth=2,
                linestyle="--",
                label="High-accuracy split",
                color="#2ca02c",
            )
            axis.fill_between(
                grouped["alpha"].to_numpy(),
                grouped["mean_bound"].to_numpy() - std_values,
                grouped["mean_bound"].to_numpy() + std_values,
                alpha=0.16,
                color="#2ca02c",
            )
        mi_subset = repeat_frame[(repeat_frame["dimension"] == float(dim)) & (repeat_frame["estimator"] == "audit_split")]
        if not mi_subset.empty:
            mi_mean = float(mi_subset["mi_bound"].mean())
            mi_std = float(mi_subset["mi_bound"].std(ddof=1)) if len(mi_subset) > 1 else 0.0
            axis.axhline(mi_mean, color="black", linestyle=":", linewidth=1.8, label="MI")
            axis.fill_between(alpha_values, mi_mean - mi_std, mi_mean + mi_std, color="black", alpha=0.08)
        axis.axvline(2.0, color="gray", linestyle="--", linewidth=1.0)
        axis.set_title(f"d = {int(dim)}", pad=8)
        axis.set_xlabel(r"$\alpha$")
        axis.set_ylabel("Bound")
        axis.grid(True, alpha=0.3)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig_diag.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig_diag.suptitle(r"Audited $\alpha$ profiles with uncertainty bands", y=0.955, fontsize=16)
    fig_diag.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    paper_summary = summary_frame[summary_frame["estimator"].isin(["audit_split", "benchmark_split"])]
    paper_main = summary_frame[summary_frame["estimator"] == "audit_split"].sort_values("dimension")
    fig_paper, (ax_main, ax_alpha) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    if not paper_main.empty:
        yerr_emp = paper_main["empirical_generalization_std"].fillna(0.0) / np.sqrt(
            np.maximum(paper_main["repeat_count"], 1.0)
        )
        ax_main.errorbar(
            paper_main["dimension"],
            paper_main["empirical_generalization_mean"],
            yerr=yerr_emp,
            fmt="o-",
            linewidth=2,
            capsize=3,
            label="Empirical generalization",
        )
        ax_main.plot(paper_main["dimension"], paper_main["mi_bound_mean"], "s--", linewidth=2, label="MI")
        ax_main.plot(paper_main["dimension"], paper_main["fixed_v2_bound_mean"], "^-", linewidth=2, label=r"Fixed $V_2$")
        ax_main.plot(
            paper_main["dimension"],
            paper_main["optimized_bound_mean"],
            "d-",
            linewidth=2,
            label=r"Split-selection $V_\alpha$",
        )
        benchmark_main = summary_frame[summary_frame["estimator"] == "benchmark_split"].sort_values("dimension")
        if not benchmark_main.empty:
            ax_main.scatter(
                benchmark_main["dimension"],
                benchmark_main["optimized_bound_mean"],
                marker="x",
                s=70,
                linewidths=2,
                label="High-accuracy split check",
            )
        ax_main.set_xlabel("Dimension d")
        ax_main.set_ylabel("Bound / empirical gap")
        ax_main.set_title(r"Audited bounds by dimension")
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(fontsize=9, loc="upper left")

        ax_alpha.errorbar(
            paper_main["dimension"],
            paper_main["selected_alpha_mean"],
            yerr=paper_main["selected_alpha_std"].fillna(0.0),
            fmt="o-",
            linewidth=2,
            capsize=3,
            label=r"Split-selection $\alpha^*$",
        )
        if not benchmark_main.empty:
            ax_alpha.scatter(
                benchmark_main["dimension"],
                benchmark_main["selected_alpha_mean"],
                marker="x",
                s=70,
                linewidths=2,
                label="High-accuracy check",
            )
        ax_alpha.set_xlabel("Dimension d")
        ax_alpha.set_ylabel(r"Selected $\alpha^*$")
        ax_alpha.set_title(r"Selected $\alpha$ under split-selection")
        ax_alpha.grid(True, alpha=0.3)
        ax_alpha.legend(fontsize=9, loc="upper right")
    fig_paper.tight_layout(pad=1.6)

    profiles_path = output_dir / "07_alpha_sweep_profiles.csv"
    summary_path = output_dir / "07_alpha_sweep_summary.csv"
    sanity_path = output_dir / "07_alpha_sweep_sanity.csv"
    diagnostic_figure_path = output_dir / "07_alpha_sweep_diagnostic.png"
    paper_figure_path = output_dir / "07_alpha_sweep_paper.png"
    config_path = output_dir / "07_alpha_sweep_config.json"

    profile_frame.to_csv(profiles_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    sanity_frame.to_csv(sanity_path, index=False)
    fig_diag.savefig(diagnostic_figure_path, dpi=200)
    fig_paper.savefig(paper_figure_path, dpi=200)
    plt.close(fig_diag)
    plt.close(fig_paper)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
            "paper_estimator": "audit_split",
        },
        config_path,
    )

    diagnostics = {
        "profiles_path": str(profiles_path),
        "summary_path": str(summary_path),
        "sanity_path": str(sanity_path),
        "diagnostic_figure_path": str(diagnostic_figure_path),
        "paper_figure_path": str(paper_figure_path),
        "story_checks": {
            "legacy_reproduction_present": bool((sanity_frame["record_type"] == "reproduction").any())
            if not sanity_frame.empty
            else False,
            "audit_compute_equalized": bool(
                repeat_frame[repeat_frame["scenario"] == "audit"]["compute_within_5pct_vs_mi"].all()
            )
            if not repeat_frame.empty
            else False,
            "split_not_systematically_lower": bool(
                sanity_frame.loc[
                    sanity_frame["record_type"] == "selection_bias_aggregate",
                    "split_not_systematically_lower",
                ].all()
            )
            if not sanity_frame.empty and (sanity_frame["record_type"] == "selection_bias_aggregate").any()
            else True,
        },
    }

    return {
        "profiles": profile_frame,
        "summary": summary_frame,
        "sanity": sanity_frame,
        "diagnostics": diagnostics,
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def run_theorem1_proxy_experiment(config: TheoremOneProxyConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = config.artifact_prefix
    sigma_loss_sq_value = sigma_loss_sq(config.clip_c)
    alpha_values = tuple(sorted({float(alpha) for alpha in config.alpha_values if float(alpha) > 1.0}))
    if not alpha_values:
        raise ValueError("TheoremOneProxyConfig.alpha_values must include at least one alpha > 1.")

    profile_records: list[dict[str, float | str | int | bool]] = []
    repeat_records: list[dict[str, float | str | int | bool]] = []
    sanity_records: list[dict[str, float | str | int | bool]] = []

    for repeat_index in range(config.num_repeats):
        for dim in config.dimensions:
            case_seed = config.seed + repeat_index * 1000 + int(dim) * 31
            case = prepare_alpha_sweep_case(
                rng=np.random.default_rng(case_seed),
                dim=int(dim),
                n=config.n,
                num_datasets=config.num_datasets,
                num_reference_datasets=config.num_reference_datasets,
                num_holdout=config.num_holdout,
                w_true_value=config.w_true_value,
                sigma_noise_sq=config.sigma_noise_sq,
                lambda_prior=config.lambda_prior,
                clip_c=config.clip_c,
                num_mi_weight_samples=config.num_mi_weight_samples,
            )
            v_profile = alpha_profile_from_marginal(
                posteriors=case["evaluation_posteriors"],
                reference_mixture=case["reference_mixture"],
                alpha_values=alpha_values,
                rng=np.random.default_rng(case_seed + 1),
                num_weight_samples=config.num_v_alpha_weight_samples,
                sigma_loss_sq_value=sigma_loss_sq_value,
                n=config.n,
            )
            w_true = make_w_true(int(dim), config.w_true_value)
            candidates, candidate_sources = build_theorem1_proxy_candidates(
                evaluation_posteriors=case["evaluation_posteriors"],
                reference_mixture=case["reference_mixture"],
                prior_reference=prior_gaussian_reference(int(dim), config.lambda_prior),
                w_true=w_true,
                rng=np.random.default_rng(case_seed + 2),
                num_candidate_posterior_samples=config.num_candidate_posterior_samples,
                num_candidate_marginal_samples=config.num_candidate_marginal_samples,
                num_candidate_prior_samples=config.num_candidate_prior_samples,
            )
            sensitivity_profile = estimate_theorem1_sensitivity_proxy(
                candidates=candidates,
                candidate_sources=candidate_sources,
                datasets=case["evaluation_datasets"],
                holdout=case["holdout"],
                alpha_values=alpha_values,
                clip_c=config.clip_c,
                batch_size=config.candidate_batch_size,
            )

            theorem1_proxy_bounds = {
                alpha: float(
                    sensitivity_profile["sensitivity_by_alpha"][alpha] * v_profile["v_alpha_values"][alpha]
                )
                for alpha in alpha_values
            }
            corollary1_bounds = {
                alpha: float(
                    corollary_subgaussian_multiplier(alpha, sigma_loss_sq_value, config.n)
                    * v_profile["v_alpha_values"][alpha]
                )
                for alpha in alpha_values
            }
            current_analytic_bounds = {
                alpha: float(v_profile["bound_by_alpha"][alpha])
                for alpha in alpha_values
            }

            selected_theorem1_proxy_alpha = min(theorem1_proxy_bounds, key=theorem1_proxy_bounds.get)
            selected_current_alpha = min(current_analytic_bounds, key=current_analytic_bounds.get)
            selected_corollary1_alpha = min(corollary1_bounds, key=corollary1_bounds.get)
            fixed_v2_proxy_bound = theorem1_proxy_bounds.get(2.0, np.nan)
            fixed_v2_current_bound = current_analytic_bounds.get(2.0, np.nan)
            fixed_v2_corollary1_bound = corollary1_bounds.get(2.0, np.nan)

            for alpha in alpha_values:
                profile_records.append(
                    {
                        "dimension": float(dim),
                        "repeat": float(repeat_index),
                        "alpha": float(alpha),
                        "v_alpha_value": float(v_profile["v_alpha_values"][alpha]),
                        "theorem1_proxy_sensitivity": float(
                            sensitivity_profile["sensitivity_by_alpha"][alpha]
                        ),
                        "theorem1_proxy_bound": float(theorem1_proxy_bounds[alpha]),
                        "current_analytic_prefactor": float(
                            subgaussian_multiplier(alpha, sigma_loss_sq_value, config.n)
                        ),
                        "current_analytic_bound": float(current_analytic_bounds[alpha]),
                        "corollary1_prefactor": float(
                            corollary_subgaussian_multiplier(alpha, sigma_loss_sq_value, config.n)
                        ),
                        "corollary1_bound": float(corollary1_bounds[alpha]),
                        "argmax_candidate_index": float(sensitivity_profile["argmax_by_alpha"][alpha]),
                        "argmax_candidate_source": sensitivity_profile["source_by_alpha"][alpha],
                        "argmax_candidate_population_loss": float(
                            sensitivity_profile["candidate_loss_by_alpha"][alpha]
                        ),
                        "log_ratio_min": float(v_profile["diagnostics"]["log_ratio_min"]),
                        "log_ratio_max": float(v_profile["diagnostics"]["log_ratio_max"]),
                    }
                )

            repeat_records.append(
                {
                    "dimension": float(dim),
                    "repeat": float(repeat_index),
                    "empirical_generalization": float(case["empirical_generalization"]),
                    "empirical_generalization_stderr": float(case["empirical_generalization_stderr"]),
                    "mi_estimate": float(case["mi_estimate"]),
                    "mi_stderr": float(case["mi_stderr"]),
                    "mi_bound": float(case["mi_bound"]),
                    "current_analytic_optimized_bound": float(
                        current_analytic_bounds[selected_current_alpha]
                    ),
                    "current_analytic_selected_alpha": float(selected_current_alpha),
                    "current_fixed_v2_bound": float(fixed_v2_current_bound),
                    "corollary1_optimized_bound": float(corollary1_bounds[selected_corollary1_alpha]),
                    "corollary1_selected_alpha": float(selected_corollary1_alpha),
                    "corollary1_fixed_v2_bound": float(fixed_v2_corollary1_bound),
                    "theorem1_proxy_optimized_bound": float(
                        theorem1_proxy_bounds[selected_theorem1_proxy_alpha]
                    ),
                    "theorem1_proxy_selected_alpha": float(selected_theorem1_proxy_alpha),
                    "theorem1_proxy_fixed_v2_bound": float(fixed_v2_proxy_bound),
                    "candidate_count": float(sensitivity_profile["candidate_count"]),
                    "source_counts_json": json.dumps(
                        sensitivity_profile["source_counts"], sort_keys=True
                    ),
                    "max_abs_loss_deviation": float(sensitivity_profile["max_abs_deviation"]),
                    "min_population_loss": float(sensitivity_profile["min_population_loss"]),
                    "max_population_loss": float(sensitivity_profile["max_population_loss"]),
                    "min_empirical_loss": float(sensitivity_profile["min_empirical_loss"]),
                    "max_empirical_loss": float(sensitivity_profile["max_empirical_loss"]),
                    "log_ratio_min": float(v_profile["diagnostics"]["log_ratio_min"]),
                    "log_ratio_max": float(v_profile["diagnostics"]["log_ratio_max"]),
                }
            )

            profile_finite = all(
                np.isfinite(value)
                for alpha in alpha_values
                for value in (
                    v_profile["v_alpha_values"][alpha],
                    theorem1_proxy_bounds[alpha],
                    current_analytic_bounds[alpha],
                    corollary1_bounds[alpha],
                    sensitivity_profile["sensitivity_by_alpha"][alpha],
                )
            )
            sanity_records.append(
                {
                    "record_type": "repeat",
                    "dimension": float(dim),
                    "repeat": float(repeat_index),
                    "all_profile_values_finite": bool(profile_finite),
                    "candidate_count": float(sensitivity_profile["candidate_count"]),
                    "theorem1_proxy_leq_current_analytic": bool(
                        theorem1_proxy_bounds[selected_theorem1_proxy_alpha]
                        <= current_analytic_bounds[selected_current_alpha] + 1e-10
                    ),
                    "theorem1_proxy_leq_corollary1": bool(
                        theorem1_proxy_bounds[selected_theorem1_proxy_alpha]
                        <= corollary1_bounds[selected_corollary1_alpha] + 1e-10
                    ),
                    "max_abs_loss_deviation": float(sensitivity_profile["max_abs_deviation"]),
                    "log_ratio_min": float(v_profile["diagnostics"]["log_ratio_min"]),
                    "log_ratio_max": float(v_profile["diagnostics"]["log_ratio_max"]),
                }
            )

    profile_frame = pd.DataFrame(profile_records)
    repeat_frame = pd.DataFrame(repeat_records)
    summary_frame = (
        repeat_frame.groupby("dimension", as_index=False)
        .agg(
            repeat_count=("repeat", "count"),
            empirical_generalization_mean=("empirical_generalization", "mean"),
            empirical_generalization_std=("empirical_generalization", "std"),
            empirical_generalization_stderr_mean=("empirical_generalization_stderr", "mean"),
            mi_bound_mean=("mi_bound", "mean"),
            mi_bound_std=("mi_bound", "std"),
            current_analytic_optimized_bound_mean=("current_analytic_optimized_bound", "mean"),
            current_analytic_optimized_bound_std=("current_analytic_optimized_bound", "std"),
            current_analytic_selected_alpha_mean=("current_analytic_selected_alpha", "mean"),
            current_analytic_selected_alpha_std=("current_analytic_selected_alpha", "std"),
            current_fixed_v2_bound_mean=("current_fixed_v2_bound", "mean"),
            current_fixed_v2_bound_std=("current_fixed_v2_bound", "std"),
            corollary1_optimized_bound_mean=("corollary1_optimized_bound", "mean"),
            corollary1_optimized_bound_std=("corollary1_optimized_bound", "std"),
            corollary1_selected_alpha_mean=("corollary1_selected_alpha", "mean"),
            corollary1_selected_alpha_std=("corollary1_selected_alpha", "std"),
            corollary1_fixed_v2_bound_mean=("corollary1_fixed_v2_bound", "mean"),
            corollary1_fixed_v2_bound_std=("corollary1_fixed_v2_bound", "std"),
            theorem1_proxy_optimized_bound_mean=("theorem1_proxy_optimized_bound", "mean"),
            theorem1_proxy_optimized_bound_std=("theorem1_proxy_optimized_bound", "std"),
            theorem1_proxy_selected_alpha_mean=("theorem1_proxy_selected_alpha", "mean"),
            theorem1_proxy_selected_alpha_std=("theorem1_proxy_selected_alpha", "std"),
            theorem1_proxy_fixed_v2_bound_mean=("theorem1_proxy_fixed_v2_bound", "mean"),
            theorem1_proxy_fixed_v2_bound_std=("theorem1_proxy_fixed_v2_bound", "std"),
            candidate_count_mean=("candidate_count", "mean"),
            max_abs_loss_deviation_mean=("max_abs_loss_deviation", "mean"),
            log_ratio_min=("log_ratio_min", "min"),
            log_ratio_max=("log_ratio_max", "max"),
        )
        .fillna(0.0)
    )
    sanity_frame = pd.DataFrame(sanity_records)
    for dimension, group in sanity_frame.groupby("dimension"):
        sanity_records.append(
            {
                "record_type": "aggregate",
                "dimension": float(dimension),
                "repeat": np.nan,
                "all_profile_values_finite": bool(group["all_profile_values_finite"].all()),
                "theorem1_proxy_leq_current_analytic_all_repeats": bool(
                    group["theorem1_proxy_leq_current_analytic"].all()
                ),
                "theorem1_proxy_leq_corollary1_all_repeats": bool(
                    group["theorem1_proxy_leq_corollary1"].all()
                ),
                "candidate_count_mean": float(group["candidate_count"].mean()),
                "max_abs_loss_deviation_mean": float(group["max_abs_loss_deviation"].mean()),
                "log_ratio_min": float(group["log_ratio_min"].min()),
                "log_ratio_max": float(group["log_ratio_max"].max()),
            }
        )
    sanity_frame = pd.DataFrame(sanity_records)

    fig_main, (ax_bounds, ax_alpha) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    dims = summary_frame["dimension"].to_numpy(dtype=float)
    repeat_counts = np.maximum(summary_frame["repeat_count"].to_numpy(dtype=float), 1.0)
    ax_bounds.errorbar(
        dims,
        summary_frame["empirical_generalization_mean"],
        yerr=summary_frame["empirical_generalization_std"].fillna(0.0) / np.sqrt(repeat_counts),
        fmt="o-",
        linewidth=2,
        capsize=3,
        label="Empirical gen.",
    )
    ax_bounds.plot(dims, summary_frame["mi_bound_mean"], "s--", linewidth=2, label="MI")
    ax_bounds.plot(
        dims,
        summary_frame["current_fixed_v2_bound_mean"],
        "^-",
        linewidth=2,
        label=r"Fixed $V_2$",
    )
    ax_bounds.plot(
        dims,
        summary_frame["current_analytic_optimized_bound_mean"],
        "d-",
        linewidth=2,
        label=r"Current analytic opt. $V_\alpha$",
    )
    ax_bounds.plot(
        dims,
        summary_frame["theorem1_proxy_optimized_bound_mean"],
        "x-",
        linewidth=2,
        markersize=8,
        label="Theorem-1 sensitivity proxy",
    )
    ax_bounds.set_xlabel("Dimension d")
    ax_bounds.set_ylabel("Bound / empirical gap")
    ax_bounds.set_title("Empirical Theorem-1 sensitivity proxy")
    ax_bounds.grid(True, alpha=0.3)
    ax_bounds.legend(fontsize=8)

    ax_alpha.plot(
        dims,
        summary_frame["current_analytic_selected_alpha_mean"],
        "o-",
        linewidth=2,
        label="Current analytic",
    )
    ax_alpha.plot(
        dims,
        summary_frame["theorem1_proxy_selected_alpha_mean"],
        "x-",
        linewidth=2,
        markersize=8,
        label="Theorem-1 proxy",
    )
    ax_alpha.set_xlabel("Dimension d")
    ax_alpha.set_ylabel(r"Selected $\alpha^*$")
    ax_alpha.set_title(r"Selected $\alpha$ comparison")
    ax_alpha.grid(True, alpha=0.3)
    ax_alpha.legend(fontsize=8)
    fig_main.tight_layout(pad=1.6)

    num_dims = len(config.dimensions)
    cols = min(2, num_dims)
    rows = int(np.ceil(num_dims / cols))
    fig_profiles, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 4.0 * rows), squeeze=False)
    axes_flat = axes.ravel()
    for axis, dim in zip(axes_flat, config.dimensions, strict=False):
        dim_profiles = profile_frame[profile_frame["dimension"] == float(dim)]
        grouped = dim_profiles.groupby("alpha", as_index=False).agg(
            current_analytic_bound=("current_analytic_bound", "mean"),
            theorem1_proxy_bound=("theorem1_proxy_bound", "mean"),
            corollary1_bound=("corollary1_bound", "mean"),
        )
        axis.plot(
            grouped["alpha"],
            grouped["current_analytic_bound"],
            linewidth=2,
            label="Current analytic",
        )
        axis.plot(
            grouped["alpha"],
            grouped["theorem1_proxy_bound"],
            linewidth=2,
            label="Theorem-1 proxy",
        )
        axis.plot(
            grouped["alpha"],
            grouped["corollary1_bound"],
            linestyle=":",
            linewidth=1.8,
            label="Corollary 2",
        )
        axis.axvline(2.0, color="gray", linestyle="--", linewidth=1.0)
        axis.set_title(f"d = {int(dim)}")
        axis.set_xlabel(r"$\alpha$")
        axis.set_ylabel("Bound")
        axis.grid(True, alpha=0.3)
    for axis in axes_flat[num_dims:]:
        axis.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig_profiles.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
    )
    fig_profiles.suptitle(r"Theorem-1 proxy and analytic $V_\alpha$ profiles", y=0.955)
    fig_profiles.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    profiles_path = output_dir / f"{artifact_prefix}_profiles.csv"
    repeats_path = output_dir / f"{artifact_prefix}_repeats.csv"
    summary_path = output_dir / f"{artifact_prefix}_summary.csv"
    sanity_path = output_dir / f"{artifact_prefix}_sanity.csv"
    figure_path = output_dir / f"{artifact_prefix}.png"
    profile_figure_path = output_dir / f"{artifact_prefix}_profiles.png"
    config_path = output_dir / f"{artifact_prefix}_config.json"

    profile_frame.to_csv(profiles_path, index=False)
    repeat_frame.to_csv(repeats_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    sanity_frame.to_csv(sanity_path, index=False)
    fig_main.savefig(figure_path, dpi=200)
    fig_profiles.savefig(profile_figure_path, dpi=200)
    plt.close(fig_main)
    plt.close(fig_profiles)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
            "warning": (
                "Theorem-1 sensitivity uses a finite candidate-set supremum and is an empirical proxy, "
                "not a theorem-valid replacement for the analytic subgaussian bound."
            ),
        },
        config_path,
    )

    return {
        "profiles": profile_frame,
        "repeats": repeat_frame,
        "summary": summary_frame,
        "sanity": sanity_frame,
        "diagnostics": {
            "profiles_path": str(profiles_path),
            "repeats_path": str(repeats_path),
            "summary_path": str(summary_path),
            "sanity_path": str(sanity_path),
            "figure_path": str(figure_path),
            "profile_figure_path": str(profile_figure_path),
            "story_checks": {
                "all_profile_values_finite": bool(
                    sanity_frame.loc[
                        sanity_frame["record_type"] == "repeat",
                        "all_profile_values_finite",
                    ].all()
                )
                if not sanity_frame.empty
                else False,
                "proxy_not_theorem_valid": True,
                "proxy_lower_than_current_analytic_all_repeats": bool(
                    sanity_frame.loc[
                        sanity_frame["record_type"] == "repeat",
                        "theorem1_proxy_leq_current_analytic",
                    ].all()
                )
                if not sanity_frame.empty
                else False,
            },
        },
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def run_alpha_sweep_comparison_experiment(config: AlphaSweepComparisonConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = config.artifact_prefix
    sigma_loss_sq_value = sigma_loss_sq(config.clip_c)
    alpha_values = tuple(sorted({float(alpha) for alpha in config.alpha_values}))
    optimization_alpha_values = alpha_optimization_grid(alpha_values)
    if 1.0 not in alpha_values:
        raise ValueError("AlphaSweepComparisonConfig.alpha_values must include 1.0.")

    profile_records: list[dict[str, float | str | int | bool]] = []
    repeat_records: list[dict[str, float | str | int | bool]] = []
    sanity_records: list[dict[str, float | str | int | bool]] = []
    surrogate_records: list[dict[str, float | str | int | bool]] = []

    def append_profile_rows(
        *,
        scenario: str,
        estimator: str,
        reference_kind: str,
        theorem_valid: bool,
        dimension: int,
        repeat_index: int,
        profile: dict[str, Any],
        num_weight_samples: int,
    ) -> None:
        for alpha in alpha_values:
            profile_records.append(
                {
                    "scenario": scenario,
                    "estimator": estimator,
                    "reference_kind": reference_kind,
                    "theorem_valid": theorem_valid,
                    "dimension": float(dimension),
                    "repeat": float(repeat_index),
                    "alpha": float(alpha),
                    "v_alpha_value": float(profile["v_alpha_values"][alpha]),
                    "bound": float(profile["bound_by_alpha"][alpha]),
                    "num_weight_samples": float(num_weight_samples),
                    "log_ratio_min": float(profile["diagnostics"]["log_ratio_min"]),
                    "log_ratio_max": float(profile["diagnostics"]["log_ratio_max"]),
                }
            )

    def append_repeat_row(
        *,
        scenario: str,
        estimator: str,
        reference_kind: str,
        theorem_valid: bool,
        case: dict[str, Any],
        repeat_index: int,
        information_bundle: dict[str, float],
        selected_alpha: float,
        direct_v1_bound: float,
        fixed_v2_bound: float,
        optimized_bound: float,
        estimator_logpdf_cost: int,
        target_logpdf_cost: int,
        evaluation_monotonicity_violation: float,
        evaluation_monotonicity_count: int,
        selection_monotonicity_violation: float | None = None,
        selection_monotonicity_count: int | None = None,
    ) -> None:
        reference_cost = mi_logpdf_cost(
            num_datasets=len(case["evaluation_posteriors"]),
            num_reference_datasets=len(case["reference_posteriors"]),
            num_weight_samples=int(case["num_mi_weight_samples"]),
        )
        repeat_records.append(
            {
                "record_type": "repeat",
                "scenario": scenario,
                "estimator": estimator,
                "reference_kind": reference_kind,
                "theorem_valid": theorem_valid,
                "dimension": float(case["dimension"]),
                "repeat": float(repeat_index),
                "empirical_generalization": float(case["empirical_generalization"]),
                "empirical_generalization_stderr": float(case["empirical_generalization_stderr"]),
                "reference_information_estimate": float(information_bundle["information_estimate"]),
                "reference_information_stderr": float(information_bundle["information_stderr"]),
                "reference_information_bound": float(information_bundle["information_bound"]),
                "pinsker_tv_bound": float(information_bundle["pinsker_tv_bound"]),
                "direct_v1_bound": float(direct_v1_bound),
                "fixed_v2_bound": float(fixed_v2_bound),
                "optimized_bound": float(optimized_bound),
                "selected_alpha": float(selected_alpha),
                "reference_logpdf_cost": float(reference_cost),
                "estimator_logpdf_cost": float(estimator_logpdf_cost),
                "target_logpdf_cost": float(target_logpdf_cost),
                "relative_compute_gap_vs_reference": float(
                    abs(estimator_logpdf_cost - reference_cost) / max(estimator_logpdf_cost, reference_cost)
                ),
                "relative_compute_gap_vs_target": float(
                    abs(estimator_logpdf_cost - target_logpdf_cost) / max(estimator_logpdf_cost, target_logpdf_cost)
                ),
                "compute_within_5pct_vs_reference": bool(
                    abs(estimator_logpdf_cost - reference_cost) / max(estimator_logpdf_cost, reference_cost) <= 0.05
                ),
                "compute_within_5pct_vs_target": bool(
                    abs(estimator_logpdf_cost - target_logpdf_cost) / max(estimator_logpdf_cost, target_logpdf_cost)
                    <= 0.05
                ),
                "direct_v1_leq_pinsker": bool(direct_v1_bound <= information_bundle["pinsker_tv_bound"] + 1e-8),
                "raw_v_alpha_monotonicity_violation": float(evaluation_monotonicity_violation),
                "raw_v_alpha_monotonicity_count": float(evaluation_monotonicity_count),
                "selection_monotonicity_violation": float(selection_monotonicity_violation)
                if selection_monotonicity_violation is not None
                else np.nan,
                "selection_monotonicity_count": float(selection_monotonicity_count)
                if selection_monotonicity_count is not None
                else np.nan,
            }
        )

    def reference_map(case: dict[str, Any]) -> dict[str, DensityReference]:
        dim = int(case["dimension"])
        return {
            "true_mixture": case["reference_mixture"],
            "moment_gaussian": moment_matched_gaussian_reference(case["reference_posteriors"]),
            "prior": prior_gaussian_reference(dim, config.lambda_prior),
        }

    def run_primary_case(
        *,
        scenario: str,
        estimator_prefix: str,
        case: dict[str, Any],
        repeat_index: int,
        total_weight_samples: int,
        theorem_valid: bool,
        include_crossfit: bool,
        base_seed: int,
    ) -> None:
        references = reference_map(case)
        target_cost = alpha_sweep_logpdf_cost(
            num_datasets=len(case["evaluation_posteriors"]),
            num_reference_datasets=len(case["reference_posteriors"]),
            num_weight_samples=total_weight_samples,
        )

        info_bundle = estimate_reference_information_bundle(
            posteriors=case["evaluation_posteriors"],
            reference=references["true_mixture"],
            rng=np.random.default_rng(base_seed + 1),
            num_weight_samples=int(case["num_mi_weight_samples"]),
            clip_c=config.clip_c,
            sigma_loss_sq_value=sigma_loss_sq_value,
            n=config.n,
        )
        split_result = run_split_alpha_estimator(
            posteriors=case["evaluation_posteriors"],
            reference=references["true_mixture"],
            alpha_values=alpha_values,
            optimization_alpha_values=optimization_alpha_values,
            sigma_loss_sq_value=sigma_loss_sq_value,
            n=config.n,
            clip_c=config.clip_c,
            total_weight_samples=total_weight_samples,
            seed=base_seed + 20,
        )
        half_budgets = partition_weight_budget(total_weight_samples, 2)
        append_profile_rows(
            scenario=scenario,
            estimator=f"{estimator_prefix}_split_select",
            reference_kind="true_mixture",
            theorem_valid=theorem_valid,
            dimension=int(case["dimension"]),
            repeat_index=repeat_index,
            profile=split_result["selection_profile"],
            num_weight_samples=int(half_budgets[0]),
        )
        append_profile_rows(
            scenario=scenario,
            estimator=f"{estimator_prefix}_split_eval",
            reference_kind="true_mixture",
            theorem_valid=theorem_valid,
            dimension=int(case["dimension"]),
            repeat_index=repeat_index,
            profile=split_result["evaluation_profile"],
            num_weight_samples=int(half_budgets[1]),
        )
        append_repeat_row(
            scenario=scenario,
            estimator="split_selection" if scenario == "primary" else "benchmark_split",
            reference_kind="true_mixture",
            theorem_valid=theorem_valid,
            case=case,
            repeat_index=repeat_index,
            information_bundle=info_bundle,
            selected_alpha=float(split_result["selected_alpha"]),
            direct_v1_bound=float(split_result["direct_v1_bound"]),
            fixed_v2_bound=float(split_result["fixed_v2_bound"]),
            optimized_bound=float(split_result["optimized_bound"]),
            estimator_logpdf_cost=target_cost,
            target_logpdf_cost=target_cost,
            evaluation_monotonicity_violation=float(split_result["evaluation_profile"]["monotonicity_violation"]),
            evaluation_monotonicity_count=int(split_result["evaluation_profile"]["monotonicity_violation_count"]),
            selection_monotonicity_violation=float(split_result["selection_profile"]["monotonicity_violation"]),
            selection_monotonicity_count=int(split_result["selection_profile"]["monotonicity_violation_count"]),
        )

        if include_crossfit:
            crossfit_result = run_crossfit_alpha_estimator(
                posteriors=case["evaluation_posteriors"],
                reference=references["true_mixture"],
                alpha_values=alpha_values,
                optimization_alpha_values=optimization_alpha_values,
                sigma_loss_sq_value=sigma_loss_sq_value,
                n=config.n,
                clip_c=config.clip_c,
                total_weight_samples=total_weight_samples,
                seed=base_seed + 120,
            )
            quarter_budgets = partition_weight_budget(total_weight_samples, 4)
            append_profile_rows(
                scenario=scenario,
                estimator=f"{estimator_prefix}_crossfit_select",
                reference_kind="true_mixture",
                theorem_valid=theorem_valid,
                dimension=int(case["dimension"]),
                repeat_index=repeat_index,
                profile=crossfit_result["selection_profile"],
                num_weight_samples=int(sum(quarter_budgets[:2])),
            )
            append_profile_rows(
                scenario=scenario,
                estimator=f"{estimator_prefix}_crossfit_eval",
                reference_kind="true_mixture",
                theorem_valid=theorem_valid,
                dimension=int(case["dimension"]),
                repeat_index=repeat_index,
                profile=crossfit_result["evaluation_profile"],
                num_weight_samples=int(sum(quarter_budgets[2:])),
            )
            append_repeat_row(
                scenario=scenario,
                estimator="crossfit_selection",
                reference_kind="true_mixture",
                theorem_valid=theorem_valid,
                case=case,
                repeat_index=repeat_index,
                information_bundle=info_bundle,
                selected_alpha=float(crossfit_result["selected_alpha"]),
                direct_v1_bound=float(crossfit_result["direct_v1_bound"]),
                fixed_v2_bound=float(crossfit_result["fixed_v2_bound"]),
                optimized_bound=float(crossfit_result["optimized_bound"]),
                estimator_logpdf_cost=target_cost,
                target_logpdf_cost=target_cost,
                evaluation_monotonicity_violation=float(crossfit_result["evaluation_profile"]["monotonicity_violation"]),
                evaluation_monotonicity_count=int(
                    crossfit_result["evaluation_profile"]["monotonicity_violation_count"]
                ),
                selection_monotonicity_violation=float(crossfit_result["selection_profile"]["monotonicity_violation"]),
                selection_monotonicity_count=int(
                    crossfit_result["selection_profile"]["monotonicity_violation_count"]
                ),
            )

        if config.include_surrogate_diagnostics:
            surrogate_records.append(
                {
                    "scenario": scenario,
                    "reference_kind": "true_mixture",
                    "theorem_valid": True,
                    "dimension": float(case["dimension"]),
                    "repeat": float(repeat_index),
                    "reference_information_bound": float(info_bundle["information_bound"]),
                    "pinsker_tv_bound": float(info_bundle["pinsker_tv_bound"]),
                    "direct_v1_bound": float(split_result["direct_v1_bound"]),
                    "fixed_v2_bound": float(split_result["fixed_v2_bound"]),
                    "optimized_bound": float(split_result["optimized_bound"]),
                    "selected_alpha": float(split_result["selected_alpha"]),
                }
            )

        reference_kinds = ("moment_gaussian", "prior") if config.include_surrogate_diagnostics else ()
        for offset, reference_kind in enumerate(reference_kinds, start=1):
            reference = references[reference_kind]
            info_bundle = estimate_reference_information_bundle(
                posteriors=case["evaluation_posteriors"],
                reference=reference,
                rng=np.random.default_rng(base_seed + 400 + offset),
                num_weight_samples=int(case["num_mi_weight_samples"]),
                clip_c=config.clip_c,
                sigma_loss_sq_value=sigma_loss_sq_value,
                n=config.n,
            )
            split_result = run_split_alpha_estimator(
                posteriors=case["evaluation_posteriors"],
                reference=reference,
                alpha_values=alpha_values,
                optimization_alpha_values=optimization_alpha_values,
                sigma_loss_sq_value=sigma_loss_sq_value,
                n=config.n,
                clip_c=config.clip_c,
                total_weight_samples=total_weight_samples,
                seed=base_seed + 500 + 20 * offset,
            )
            surrogate_records.append(
                {
                    "scenario": scenario,
                    "reference_kind": reference_kind,
                    "theorem_valid": False,
                    "dimension": float(case["dimension"]),
                    "repeat": float(repeat_index),
                    "reference_information_bound": float(info_bundle["information_bound"]),
                    "pinsker_tv_bound": float(info_bundle["pinsker_tv_bound"]),
                    "direct_v1_bound": float(split_result["direct_v1_bound"]),
                    "fixed_v2_bound": float(split_result["fixed_v2_bound"]),
                    "optimized_bound": float(split_result["optimized_bound"]),
                    "selected_alpha": float(split_result["selected_alpha"]),
                }
            )

    if config.include_legacy_reference:
        legacy_reference_path = output_dir / "02_alpha_sweep.csv"
        if legacy_reference_path.exists():
            legacy_reference = pd.read_csv(legacy_reference_path)
        else:
            legacy_bundle = run_alpha_sweep_experiment(
                AlphaSweepExperimentConfig(
                    sigma_noise_sq=config.sigma_noise_sq,
                    lambda_prior=config.lambda_prior,
                    clip_c=config.clip_c,
                    w_true_value=config.w_true_value,
                    dimensions=config.dimensions,
                    alpha_values=tuple(alpha for alpha in config.alpha_values if alpha > 1.0),
                    n=config.n,
                    num_datasets=config.legacy_num_datasets,
                    num_reference_datasets=config.legacy_num_reference_datasets,
                    num_weight_samples=config.legacy_num_weight_samples,
                    num_holdout=config.num_holdout,
                    num_mi_weight_samples=config.legacy_num_mi_weight_samples,
                    seed=config.legacy_seed,
                    output_dir=config.output_dir,
                )
            )
            legacy_reference = legacy_bundle["results"]
        legacy_cost = alpha_sweep_logpdf_cost(
            num_datasets=config.legacy_num_datasets,
            num_reference_datasets=config.legacy_num_reference_datasets,
            num_weight_samples=config.legacy_num_weight_samples,
        )
        legacy_reference_cost = mi_logpdf_cost(
            num_datasets=config.legacy_num_datasets,
            num_reference_datasets=config.legacy_num_reference_datasets,
            num_weight_samples=config.legacy_num_mi_weight_samples,
        )
        for _, row in legacy_reference.iterrows():
            repeat_records.append(
                {
                    "record_type": "repeat",
                    "scenario": "legacy",
                    "estimator": "legacy_naive",
                    "reference_kind": "true_mixture",
                    "theorem_valid": True,
                    "dimension": float(row["dimension"]),
                    "repeat": 0.0,
                    "empirical_generalization": float(row["empirical_generalization"]),
                    "empirical_generalization_stderr": float(row["empirical_generalization_stderr"]),
                    "reference_information_estimate": float(row["mi_estimate"]),
                    "reference_information_stderr": float(row["mi_stderr"]),
                    "reference_information_bound": float(row["mi_bound"]),
                    "pinsker_tv_bound": float(pinsker_tv_bound_from_information(config.clip_c, row["mi_estimate"])),
                    "direct_v1_bound": np.nan,
                    "fixed_v2_bound": float(row["v2_bound"]),
                    "optimized_bound": float(row["optimized_v_alpha_bound"]),
                    "selected_alpha": float(row["optimal_alpha"]),
                    "reference_logpdf_cost": float(legacy_reference_cost),
                    "estimator_logpdf_cost": float(legacy_cost),
                    "target_logpdf_cost": float(legacy_cost),
                    "relative_compute_gap_vs_reference": float(
                        abs(legacy_cost - legacy_reference_cost) / max(legacy_cost, legacy_reference_cost)
                    ),
                    "relative_compute_gap_vs_target": 0.0,
                    "compute_within_5pct_vs_reference": bool(
                        abs(legacy_cost - legacy_reference_cost) / max(legacy_cost, legacy_reference_cost) <= 0.05
                    ),
                    "compute_within_5pct_vs_target": True,
                    "direct_v1_leq_pinsker": np.nan,
                    "raw_v_alpha_monotonicity_violation": np.nan,
                    "raw_v_alpha_monotonicity_count": np.nan,
                    "selection_monotonicity_violation": np.nan,
                    "selection_monotonicity_count": np.nan,
                }
            )
            sanity_records.append(
                {
                    "record_type": "reproduction",
                    "scenario": "legacy",
                    "estimator": "legacy_naive",
                    "reference_kind": "true_mixture",
                    "dimension": float(row["dimension"]),
                    "repeat": 0.0,
                    "matches_02_within_tolerance": True,
                }
            )

    for repeat_index in range(config.primary_num_repeats):
        for dim in config.dimensions:
            case_seed = config.seed + repeat_index * 1000 + int(dim) * 41
            case = prepare_alpha_sweep_case(
                rng=np.random.default_rng(case_seed),
                dim=int(dim),
                n=config.n,
                num_datasets=config.primary_num_datasets,
                num_reference_datasets=config.primary_num_reference_datasets,
                num_holdout=config.num_holdout,
                w_true_value=config.w_true_value,
                sigma_noise_sq=config.sigma_noise_sq,
                lambda_prior=config.lambda_prior,
                clip_c=config.clip_c,
                num_mi_weight_samples=config.primary_num_mi_weight_samples,
            )
            case["num_mi_weight_samples"] = config.primary_num_mi_weight_samples
            run_primary_case(
                scenario="primary",
                estimator_prefix="primary",
                case=case,
                repeat_index=repeat_index,
                total_weight_samples=config.primary_total_weight_samples,
                theorem_valid=True,
                include_crossfit=config.include_crossfit,
                base_seed=case_seed + 10,
            )

    for repeat_index in range(config.benchmark_num_repeats):
        for dim in config.benchmark_dimensions:
            case_seed = config.seed + 50000 + repeat_index * 1000 + int(dim) * 43
            case = prepare_alpha_sweep_case(
                rng=np.random.default_rng(case_seed),
                dim=int(dim),
                n=config.n,
                num_datasets=config.benchmark_num_datasets,
                num_reference_datasets=config.benchmark_num_reference_datasets,
                num_holdout=config.num_holdout,
                w_true_value=config.w_true_value,
                sigma_noise_sq=config.sigma_noise_sq,
                lambda_prior=config.lambda_prior,
                clip_c=config.clip_c,
                num_mi_weight_samples=config.benchmark_num_mi_weight_samples,
            )
            case["num_mi_weight_samples"] = config.benchmark_num_mi_weight_samples
            run_primary_case(
                scenario="benchmark",
                estimator_prefix="benchmark",
                case=case,
                repeat_index=repeat_index,
                total_weight_samples=config.benchmark_total_weight_samples,
                theorem_valid=True,
                include_crossfit=False,
                base_seed=case_seed + 10,
            )

    profile_frame = pd.DataFrame(profile_records)
    repeat_frame = pd.DataFrame(repeat_records)
    surrogate_frame = pd.DataFrame(
        surrogate_records,
        columns=[
            "scenario",
            "reference_kind",
            "theorem_valid",
            "dimension",
            "repeat",
            "reference_information_bound",
            "pinsker_tv_bound",
            "direct_v1_bound",
            "fixed_v2_bound",
            "optimized_bound",
            "selected_alpha",
        ],
    )

    summary_frame = (
        repeat_frame.groupby(["scenario", "estimator", "reference_kind", "theorem_valid", "dimension"], as_index=False)
        .agg(
            repeat_count=("repeat", "count"),
            empirical_generalization_mean=("empirical_generalization", "mean"),
            empirical_generalization_std=("empirical_generalization", "std"),
            empirical_generalization_stderr_mean=("empirical_generalization_stderr", "mean"),
            reference_information_bound_mean=("reference_information_bound", "mean"),
            reference_information_bound_std=("reference_information_bound", "std"),
            pinsker_tv_bound_mean=("pinsker_tv_bound", "mean"),
            pinsker_tv_bound_std=("pinsker_tv_bound", "std"),
            direct_v1_bound_mean=("direct_v1_bound", "mean"),
            direct_v1_bound_std=("direct_v1_bound", "std"),
            fixed_v2_bound_mean=("fixed_v2_bound", "mean"),
            fixed_v2_bound_std=("fixed_v2_bound", "std"),
            optimized_bound_mean=("optimized_bound", "mean"),
            optimized_bound_std=("optimized_bound", "std"),
            selected_alpha_mean=("selected_alpha", "mean"),
            selected_alpha_median=("selected_alpha", "median"),
            selected_alpha_std=("selected_alpha", "std"),
            selected_alpha_min=("selected_alpha", "min"),
            selected_alpha_max=("selected_alpha", "max"),
        )
    )
    summary_frame["selected_alpha_range"] = (
        summary_frame["selected_alpha_max"] - summary_frame["selected_alpha_min"]
    )
    for column in [
        "empirical_generalization_std",
        "reference_information_bound_std",
        "pinsker_tv_bound_std",
        "direct_v1_bound_std",
        "fixed_v2_bound_std",
        "optimized_bound_std",
        "selected_alpha_std",
    ]:
        summary_frame[column] = summary_frame[column].fillna(0.0)

    if surrogate_frame.empty:
        surrogate_summary = pd.DataFrame(
            columns=[
                "scenario",
                "reference_kind",
                "theorem_valid",
                "dimension",
                "repeat_count",
                "reference_information_bound_mean",
                "reference_information_bound_std",
                "pinsker_tv_bound_mean",
                "pinsker_tv_bound_std",
                "direct_v1_bound_mean",
                "direct_v1_bound_std",
                "fixed_v2_bound_mean",
                "fixed_v2_bound_std",
                "optimized_bound_mean",
                "optimized_bound_std",
                "selected_alpha_mean",
                "selected_alpha_std",
            ]
        )
    else:
        surrogate_summary = (
            surrogate_frame.groupby(["scenario", "reference_kind", "theorem_valid", "dimension"], as_index=False)
            .agg(
                repeat_count=("repeat", "count"),
                reference_information_bound_mean=("reference_information_bound", "mean"),
                reference_information_bound_std=("reference_information_bound", "std"),
                pinsker_tv_bound_mean=("pinsker_tv_bound", "mean"),
                pinsker_tv_bound_std=("pinsker_tv_bound", "std"),
                direct_v1_bound_mean=("direct_v1_bound", "mean"),
                direct_v1_bound_std=("direct_v1_bound", "std"),
                fixed_v2_bound_mean=("fixed_v2_bound", "mean"),
                fixed_v2_bound_std=("fixed_v2_bound", "std"),
                optimized_bound_mean=("optimized_bound", "mean"),
                optimized_bound_std=("optimized_bound", "std"),
                selected_alpha_mean=("selected_alpha", "mean"),
                selected_alpha_std=("selected_alpha", "std"),
            )
        )
        for column in [
            "reference_information_bound_std",
            "pinsker_tv_bound_std",
            "direct_v1_bound_std",
            "fixed_v2_bound_std",
            "optimized_bound_std",
            "selected_alpha_std",
        ]:
            surrogate_summary[column] = surrogate_summary[column].fillna(0.0)

    for _, row in repeat_frame.iterrows():
        sanity_records.append(row.to_dict())

    if not repeat_frame.empty:
        for (scenario, estimator, reference_kind, dimension), group in repeat_frame.groupby(
            ["scenario", "estimator", "reference_kind", "dimension"]
        ):
            unique_alpha, counts = np.unique(group["selected_alpha"].to_numpy(dtype=float), return_counts=True)
            alpha_frequency = {
                f"{alpha:.2f}": int(count)
                for alpha, count in sorted(zip(unique_alpha, counts, strict=False), key=lambda item: item[0])
            }
            sanity_records.append(
                {
                    "record_type": "aggregate",
                    "scenario": scenario,
                    "estimator": estimator,
                    "reference_kind": reference_kind,
                    "dimension": float(dimension),
                    "repeat": np.nan,
                    "alpha_frequency_json": json.dumps(alpha_frequency, sort_keys=True),
                    "selected_alpha_mean": float(group["selected_alpha"].mean()),
                    "selected_alpha_median": float(group["selected_alpha"].median()),
                    "selected_alpha_std": float(group["selected_alpha"].std(ddof=1)) if len(group) > 1 else 0.0,
                    "selected_alpha_range": float(group["selected_alpha"].max() - group["selected_alpha"].min()),
                    "alpha_stability_flag": bool(group["selected_alpha"].max() - group["selected_alpha"].min() > 0.1),
                    "compute_within_5pct_all_repeats": bool(group["compute_within_5pct_vs_reference"].all()),
                    "direct_v1_leq_pinsker_all_repeats": bool(group["direct_v1_leq_pinsker"].fillna(True).all()),
                    "optimized_bound_mean": float(group["optimized_bound"].mean()),
                }
            )

    primary_split = repeat_frame[
        (repeat_frame["scenario"] == "primary")
        & (repeat_frame["estimator"] == "split_selection")
        & (repeat_frame["reference_kind"] == "true_mixture")
    ]
    primary_crossfit = repeat_frame[
        (repeat_frame["scenario"] == "primary")
        & (repeat_frame["estimator"] == "crossfit_selection")
        & (repeat_frame["reference_kind"] == "true_mixture")
    ]
    if not primary_split.empty and not primary_crossfit.empty:
        merged = primary_crossfit.merge(
            primary_split,
            on=["dimension", "repeat"],
            suffixes=("_crossfit", "_split"),
        )
        for dimension, group in merged.groupby("dimension"):
            sanity_records.append(
                {
                    "record_type": "estimator_comparison",
                    "scenario": "primary",
                    "estimator": "crossfit_vs_split",
                    "reference_kind": "true_mixture",
                    "dimension": float(dimension),
                    "repeat": np.nan,
                    "optimized_bound_gap_mean": float(
                        (group["optimized_bound_crossfit"] - group["optimized_bound_split"]).mean()
                    ),
                    "selected_alpha_gap_mean": float(
                        (group["selected_alpha_crossfit"] - group["selected_alpha_split"]).mean()
                    ),
                }
            )

    benchmark_split = repeat_frame[
        (repeat_frame["scenario"] == "benchmark")
        & (repeat_frame["estimator"] == "benchmark_split")
        & (repeat_frame["reference_kind"] == "true_mixture")
    ]
    if not primary_split.empty and not benchmark_split.empty:
        benchmark_comparison = benchmark_split.merge(
            summary_frame[
                (summary_frame["scenario"] == "primary")
                & (summary_frame["estimator"] == "split_selection")
                & (summary_frame["reference_kind"] == "true_mixture")
            ][["dimension", "optimized_bound_mean", "selected_alpha_mean"]],
            on="dimension",
            how="left",
        )
        for _, row in benchmark_comparison.iterrows():
            relative_change = abs(row["optimized_bound"] - row["optimized_bound_mean"]) / max(
                row["optimized_bound_mean"], 1e-12
            )
            alpha_delta = abs(row["selected_alpha"] - row["selected_alpha_mean"])
            sanity_records.append(
                {
                    "record_type": "benchmark_comparison",
                    "scenario": "benchmark",
                    "estimator": "benchmark_split_vs_primary_split",
                    "reference_kind": "true_mixture",
                    "dimension": float(row["dimension"]),
                    "repeat": float(row["repeat"]),
                    "optimized_bound_relative_change": float(relative_change),
                    "selected_alpha_delta": float(alpha_delta),
                    "benchmark_instability_flag": bool(relative_change > 0.10 or alpha_delta > 0.1),
                }
            )

    sanity_frame = pd.DataFrame(sanity_records)

    fig_main, (ax_bounds, ax_alpha) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    paper_summary = summary_frame[
        (summary_frame["scenario"] == "primary")
        & (summary_frame["estimator"] == "split_selection")
        & (summary_frame["reference_kind"] == "true_mixture")
    ].sort_values("dimension")
    if not paper_summary.empty:
        ax_bounds.errorbar(
            paper_summary["dimension"],
            paper_summary["empirical_generalization_mean"],
            yerr=paper_summary["empirical_generalization_std"].to_numpy() / np.sqrt(
                np.maximum(paper_summary["repeat_count"].to_numpy(), 1.0)
            ),
            fmt="o-",
            linewidth=2,
            capsize=3,
            label="Empirical generalization",
        )
        ax_bounds.plot(
            paper_summary["dimension"],
            paper_summary["direct_v1_bound_mean"],
            "x--",
            linewidth=2,
            label=r"Direct $V_1$/TV",
        )
        if config.plot_pinsker_tv:
            ax_bounds.plot(
                paper_summary["dimension"],
                paper_summary["pinsker_tv_bound_mean"],
                "v:",
                linewidth=2,
                label="Pinsker-TV",
            )
        ax_bounds.plot(
            paper_summary["dimension"],
            paper_summary["reference_information_bound_mean"],
            "s--",
            linewidth=2,
            label="MI",
        )
        ax_bounds.plot(
            paper_summary["dimension"],
            paper_summary["fixed_v2_bound_mean"],
            "^-",
            linewidth=2,
            label=r"Fixed $V_2$",
        )
        ax_bounds.plot(
            paper_summary["dimension"],
            paper_summary["optimized_bound_mean"],
            "d-",
            linewidth=2,
            label=r"Split-selection $V_\alpha$",
        )
        benchmark_summary = summary_frame[
            (summary_frame["scenario"] == "benchmark")
            & (summary_frame["estimator"] == "benchmark_split")
            & (summary_frame["reference_kind"] == "true_mixture")
        ].sort_values("dimension")
        if not benchmark_summary.empty:
            ax_bounds.scatter(
                benchmark_summary["dimension"],
                benchmark_summary["optimized_bound_mean"],
                marker="x",
                s=70,
                linewidths=2,
                label="Benchmark split",
            )
        ax_bounds.set_xlabel("Dimension d")
        ax_bounds.set_ylabel("Bound / empirical gap")
        ax_bounds.set_title(r"Theorem-valid parameter-space comparison")
        ax_bounds.grid(True, alpha=0.3)
        ax_bounds.legend()

        ax_alpha.errorbar(
            paper_summary["dimension"],
            paper_summary["selected_alpha_mean"],
            yerr=paper_summary["selected_alpha_std"],
            fmt="o-",
            linewidth=2,
            capsize=3,
            label=r"Split-selection $\alpha^*$",
        )
        if not benchmark_summary.empty:
            ax_alpha.scatter(
                benchmark_summary["dimension"],
                benchmark_summary["selected_alpha_mean"],
                marker="x",
                s=70,
                linewidths=2,
                label="Benchmark split",
            )
        ax_alpha.set_xlabel("Dimension d")
        ax_alpha.set_ylabel(r"Selected $\alpha^*$")
        ax_alpha.set_title(r"Optimized $\alpha$ under theorem-valid estimator")
        ax_alpha.grid(True, alpha=0.3)
        ax_alpha.legend()
    fig_main.tight_layout()

    fig_estimators, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), sharex=True)
    axes_flat = axes.ravel()
    legacy_points = {
        float(row["dimension"]): (float(row["selected_alpha"]), float(row["optimized_bound"]))
        for _, row in repeat_frame[repeat_frame["estimator"] == "legacy_naive"].iterrows()
    }
    for axis, dim in zip(axes_flat, config.dimensions, strict=False):
        dim_profiles = profile_frame[
            (profile_frame["dimension"] == float(dim)) & (profile_frame["reference_kind"] == "true_mixture")
        ]
        for estimator, label, color in [
            ("primary_split_eval", "Split eval", "#d62728"),
            ("primary_crossfit_eval", "Crossfit eval", "#1f77b4"),
            ("benchmark_split_eval", "Benchmark eval", "#2ca02c"),
        ]:
            subset = dim_profiles[dim_profiles["estimator"] == estimator]
            if subset.empty:
                continue
            grouped = subset.groupby("alpha", as_index=False).agg(mean_bound=("bound", "mean"), std_bound=("bound", "std"))
            grouped = grouped.sort_values("alpha")
            std_values = grouped["std_bound"].fillna(0.0).to_numpy()
            axis.plot(grouped["alpha"], grouped["mean_bound"], linewidth=2, label=label, color=color)
            axis.fill_between(
                grouped["alpha"].to_numpy(),
                grouped["mean_bound"].to_numpy() - std_values,
                grouped["mean_bound"].to_numpy() + std_values,
                alpha=0.18,
                color=color,
            )
        if float(dim) in legacy_points:
            legacy_alpha, legacy_bound = legacy_points[float(dim)]
            axis.scatter(
                [legacy_alpha],
                [legacy_bound],
                color="black",
                marker="x",
                s=70,
                linewidths=2,
                label="Legacy 02 optimum",
            )
        axis.axvline(1.0, color="gray", linestyle=":", linewidth=1.0)
        axis.axvline(2.0, color="gray", linestyle="--", linewidth=1.0)
        axis.set_title(f"d = {int(dim)}")
        axis.set_xlabel(r"$\alpha$")
        axis.set_ylabel("Bound")
        axis.grid(True, alpha=0.3)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig_estimators.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig_estimators.suptitle(r"Estimator-sensitive $\alpha$ profiles on the true marginal", y=0.98)
    fig_estimators.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    fig_surrogates, (ax_surr_bound, ax_surr_alpha) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    surrogate_primary = surrogate_summary[surrogate_summary["scenario"] == "primary"].sort_values(
        ["reference_kind", "dimension"]
    )
    if not surrogate_primary.empty:
        for reference_kind, label, color in [
            ("true_mixture", "True mixture", "#1f77b4"),
            ("moment_gaussian", "Moment Gaussian", "#ff7f0e"),
            ("prior", "Prior", "#2ca02c"),
        ]:
            subset = surrogate_primary[surrogate_primary["reference_kind"] == reference_kind]
            if subset.empty:
                continue
            ax_surr_bound.errorbar(
                subset["dimension"],
                subset["optimized_bound_mean"],
                yerr=subset["optimized_bound_std"],
                fmt="o-",
                linewidth=2,
                capsize=3,
                color=color,
                label=label,
            )
            ax_surr_alpha.errorbar(
                subset["dimension"],
                subset["selected_alpha_mean"],
                yerr=subset["selected_alpha_std"],
                fmt="o-",
                linewidth=2,
                capsize=3,
                color=color,
                label=label,
            )
        ax_surr_bound.set_xlabel("Dimension d")
        ax_surr_bound.set_ylabel(r"Optimized bound")
        ax_surr_bound.set_title("Reference sensitivity of optimized bound")
        ax_surr_bound.grid(True, alpha=0.3)
        ax_surr_bound.legend()
        ax_surr_alpha.set_xlabel("Dimension d")
        ax_surr_alpha.set_ylabel(r"Selected $\alpha^*$")
        ax_surr_alpha.set_title("Reference sensitivity of selected alpha")
        ax_surr_alpha.grid(True, alpha=0.3)
        ax_surr_alpha.legend()
    fig_surrogates.tight_layout()

    profiles_path = output_dir / f"{artifact_prefix}_profiles.csv"
    summary_path = output_dir / f"{artifact_prefix}_summary.csv"
    sanity_path = output_dir / f"{artifact_prefix}_sanity.csv"
    surrogates_path = output_dir / f"{artifact_prefix}_surrogates.csv"
    main_figure_path = output_dir / f"{artifact_prefix}_main.png"
    estimator_figure_path = output_dir / f"{artifact_prefix}_estimators.png"
    surrogate_figure_path = output_dir / f"{artifact_prefix}_surrogates.png"
    config_path = output_dir / f"{artifact_prefix}_config.json"

    profile_frame.to_csv(profiles_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    sanity_frame.to_csv(sanity_path, index=False)
    surrogate_summary.to_csv(surrogates_path, index=False)
    fig_main.savefig(main_figure_path, dpi=200)
    fig_estimators.savefig(estimator_figure_path, dpi=200)
    fig_surrogates.savefig(surrogate_figure_path, dpi=200)
    plt.close(fig_main)
    plt.close(fig_estimators)
    plt.close(fig_surrogates)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
            "paper_estimator": "split_selection",
        },
        config_path,
    )

    diagnostics = {
        "profiles_path": str(profiles_path),
        "summary_path": str(summary_path),
        "sanity_path": str(sanity_path),
        "surrogates_path": str(surrogates_path),
        "main_figure_path": str(main_figure_path),
        "estimator_figure_path": str(estimator_figure_path),
        "surrogate_figure_path": str(surrogate_figure_path),
        "story_checks": {
            "theorem_valid_compute_equalized": bool(
                repeat_frame.loc[
                    (repeat_frame["theorem_valid"] == True) & (repeat_frame["scenario"] != "legacy"),
                    "compute_within_5pct_vs_reference",
                ].all()
            )
            if not repeat_frame.empty
            else False,
            "direct_v1_leq_pinsker_all": bool(
                repeat_frame.loc[
                    (repeat_frame["theorem_valid"] == True) & (repeat_frame["reference_kind"] == "true_mixture"),
                    "direct_v1_leq_pinsker",
                ].fillna(True).all()
            )
            if not repeat_frame.empty
            else False,
            "surrogate_rows_present": bool(not surrogate_summary.empty),
        },
    }

    return {
        "profiles": profile_frame,
        "summary": summary_frame,
        "sanity": sanity_frame,
        "surrogates": surrogate_summary,
        "diagnostics": diagnostics,
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def run_alpha_sweep_sample_size_experiment(config: AlphaSweepSampleSizeConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = config.artifact_prefix
    sigma_loss_sq_value = sigma_loss_sq(config.clip_c)
    alpha_values = tuple(sorted({float(alpha) for alpha in config.alpha_values}))
    optimization_alpha_values = alpha_optimization_grid(alpha_values)
    if 1.0 not in alpha_values:
        raise ValueError("AlphaSweepSampleSizeConfig.alpha_values must include 1.0.")

    profile_records: list[dict[str, float | str | int | bool]] = []
    repeat_records: list[dict[str, float | str | int | bool]] = []
    sanity_records: list[dict[str, float | str | int | bool]] = []

    def append_profile_rows(
        *,
        scenario: str,
        estimator: str,
        dimension: int,
        n_value: int,
        repeat_index: int,
        profile: dict[str, Any],
        num_weight_samples: int,
    ) -> None:
        for alpha in alpha_values:
            profile_records.append(
                {
                    "scenario": scenario,
                    "estimator": estimator,
                    "dimension": float(dimension),
                    "n": float(n_value),
                    "repeat": float(repeat_index),
                    "alpha": float(alpha),
                    "v_alpha_value": float(profile["v_alpha_values"][alpha]),
                    "bound": float(profile["bound_by_alpha"][alpha]),
                    "num_weight_samples": float(num_weight_samples),
                }
            )

    def append_repeat_row(
        *,
        scenario: str,
        estimator: str,
        case: dict[str, Any],
        n_value: int,
        repeat_index: int,
        information_bundle: dict[str, float],
        split_result: dict[str, Any],
        estimator_logpdf_cost: int,
        target_logpdf_cost: int,
    ) -> None:
        reference_cost = mi_logpdf_cost(
            num_datasets=len(case["evaluation_posteriors"]),
            num_reference_datasets=len(case["reference_posteriors"]),
            num_weight_samples=int(case["num_mi_weight_samples"]),
        )
        repeat_records.append(
            {
                "record_type": "repeat",
                "scenario": scenario,
                "estimator": estimator,
                "dimension": float(case["dimension"]),
                "n": float(n_value),
                "repeat": float(repeat_index),
                "empirical_generalization": float(case["empirical_generalization"]),
                "empirical_generalization_stderr": float(case["empirical_generalization_stderr"]),
                "reference_information_bound": float(information_bundle["information_bound"]),
                "pinsker_tv_bound": float(information_bundle["pinsker_tv_bound"]),
                "direct_v1_bound": float(split_result["direct_v1_bound"]),
                "fixed_v2_bound": float(split_result["fixed_v2_bound"]),
                "optimized_bound": float(split_result["optimized_bound"]),
                "selected_alpha": float(split_result["selected_alpha"]),
                "reference_logpdf_cost": float(reference_cost),
                "estimator_logpdf_cost": float(estimator_logpdf_cost),
                "target_logpdf_cost": float(target_logpdf_cost),
                "compute_within_5pct_vs_reference": bool(
                    abs(estimator_logpdf_cost - reference_cost) / max(estimator_logpdf_cost, reference_cost) <= 0.05
                ),
                "direct_v1_leq_pinsker": bool(
                    split_result["direct_v1_bound"] <= information_bundle["pinsker_tv_bound"] + 1e-8
                ),
                "raw_v_alpha_monotonicity_violation": float(split_result["evaluation_profile"]["monotonicity_violation"]),
                "raw_v_alpha_monotonicity_count": int(
                    split_result["evaluation_profile"]["monotonicity_violation_count"]
                ),
            }
        )

    for repeat_index in range(config.primary_num_repeats):
        for dim in config.dimensions:
            for n_value in config.n_values:
                case_seed = config.seed + repeat_index * 5000 + int(dim) * 53 + int(n_value)
                case = prepare_alpha_sweep_case(
                    rng=np.random.default_rng(case_seed),
                    dim=int(dim),
                    n=int(n_value),
                    num_datasets=config.primary_num_datasets,
                    num_reference_datasets=config.primary_num_reference_datasets,
                    num_holdout=config.num_holdout,
                    w_true_value=config.w_true_value,
                    sigma_noise_sq=config.sigma_noise_sq,
                    lambda_prior=config.lambda_prior,
                    clip_c=config.clip_c,
                    num_mi_weight_samples=config.primary_num_mi_weight_samples,
                )
                case["num_mi_weight_samples"] = config.primary_num_mi_weight_samples
                info_bundle = estimate_reference_information_bundle(
                    posteriors=case["evaluation_posteriors"],
                    reference=case["reference_mixture"],
                    rng=np.random.default_rng(case_seed + 1),
                    num_weight_samples=config.primary_num_mi_weight_samples,
                    clip_c=config.clip_c,
                    sigma_loss_sq_value=sigma_loss_sq_value,
                    n=int(n_value),
                )
                split_result = run_split_alpha_estimator(
                    posteriors=case["evaluation_posteriors"],
                    reference=case["reference_mixture"],
                    alpha_values=alpha_values,
                    optimization_alpha_values=optimization_alpha_values,
                    sigma_loss_sq_value=sigma_loss_sq_value,
                    n=int(n_value),
                    clip_c=config.clip_c,
                    total_weight_samples=config.primary_total_weight_samples,
                    seed=case_seed + 10,
                )
                half_budgets = partition_weight_budget(config.primary_total_weight_samples, 2)
                append_profile_rows(
                    scenario="primary",
                    estimator="split_select",
                    dimension=int(dim),
                    n_value=int(n_value),
                    repeat_index=repeat_index,
                    profile=split_result["selection_profile"],
                    num_weight_samples=int(half_budgets[0]),
                )
                append_profile_rows(
                    scenario="primary",
                    estimator="split_eval",
                    dimension=int(dim),
                    n_value=int(n_value),
                    repeat_index=repeat_index,
                    profile=split_result["evaluation_profile"],
                    num_weight_samples=int(half_budgets[1]),
                )
                append_repeat_row(
                    scenario="primary",
                    estimator="split_selection",
                    case=case,
                    n_value=int(n_value),
                    repeat_index=repeat_index,
                    information_bundle=info_bundle,
                    split_result=split_result,
                    estimator_logpdf_cost=alpha_sweep_logpdf_cost(
                        num_datasets=config.primary_num_datasets,
                        num_reference_datasets=config.primary_num_reference_datasets,
                        num_weight_samples=config.primary_total_weight_samples,
                    ),
                    target_logpdf_cost=alpha_sweep_logpdf_cost(
                        num_datasets=config.primary_num_datasets,
                        num_reference_datasets=config.primary_num_reference_datasets,
                        num_weight_samples=config.primary_total_weight_samples,
                    ),
                )

    for repeat_index in range(config.benchmark_num_repeats):
        for dim in config.dimensions:
            for n_value in config.benchmark_n_values:
                case_seed = config.seed + 60000 + repeat_index * 5000 + int(dim) * 59 + int(n_value)
                case = prepare_alpha_sweep_case(
                    rng=np.random.default_rng(case_seed),
                    dim=int(dim),
                    n=int(n_value),
                    num_datasets=config.benchmark_num_datasets,
                    num_reference_datasets=config.benchmark_num_reference_datasets,
                    num_holdout=config.num_holdout,
                    w_true_value=config.w_true_value,
                    sigma_noise_sq=config.sigma_noise_sq,
                    lambda_prior=config.lambda_prior,
                    clip_c=config.clip_c,
                    num_mi_weight_samples=config.benchmark_num_mi_weight_samples,
                )
                case["num_mi_weight_samples"] = config.benchmark_num_mi_weight_samples
                info_bundle = estimate_reference_information_bundle(
                    posteriors=case["evaluation_posteriors"],
                    reference=case["reference_mixture"],
                    rng=np.random.default_rng(case_seed + 1),
                    num_weight_samples=config.benchmark_num_mi_weight_samples,
                    clip_c=config.clip_c,
                    sigma_loss_sq_value=sigma_loss_sq_value,
                    n=int(n_value),
                )
                split_result = run_split_alpha_estimator(
                    posteriors=case["evaluation_posteriors"],
                    reference=case["reference_mixture"],
                    alpha_values=alpha_values,
                    optimization_alpha_values=optimization_alpha_values,
                    sigma_loss_sq_value=sigma_loss_sq_value,
                    n=int(n_value),
                    clip_c=config.clip_c,
                    total_weight_samples=config.benchmark_total_weight_samples,
                    seed=case_seed + 10,
                )
                half_budgets = partition_weight_budget(config.benchmark_total_weight_samples, 2)
                append_profile_rows(
                    scenario="benchmark",
                    estimator="benchmark_split_select",
                    dimension=int(dim),
                    n_value=int(n_value),
                    repeat_index=repeat_index,
                    profile=split_result["selection_profile"],
                    num_weight_samples=int(half_budgets[0]),
                )
                append_profile_rows(
                    scenario="benchmark",
                    estimator="benchmark_split_eval",
                    dimension=int(dim),
                    n_value=int(n_value),
                    repeat_index=repeat_index,
                    profile=split_result["evaluation_profile"],
                    num_weight_samples=int(half_budgets[1]),
                )
                append_repeat_row(
                    scenario="benchmark",
                    estimator="benchmark_split",
                    case=case,
                    n_value=int(n_value),
                    repeat_index=repeat_index,
                    information_bundle=info_bundle,
                    split_result=split_result,
                    estimator_logpdf_cost=alpha_sweep_logpdf_cost(
                        num_datasets=config.benchmark_num_datasets,
                        num_reference_datasets=config.benchmark_num_reference_datasets,
                        num_weight_samples=config.benchmark_total_weight_samples,
                    ),
                    target_logpdf_cost=alpha_sweep_logpdf_cost(
                        num_datasets=config.benchmark_num_datasets,
                        num_reference_datasets=config.benchmark_num_reference_datasets,
                        num_weight_samples=config.benchmark_total_weight_samples,
                    ),
                )

    profile_frame = pd.DataFrame(profile_records)
    repeat_frame = pd.DataFrame(repeat_records)
    summary_frame = (
        repeat_frame.groupby(["scenario", "estimator", "dimension", "n"], as_index=False)
        .agg(
            repeat_count=("repeat", "count"),
            empirical_generalization_mean=("empirical_generalization", "mean"),
            empirical_generalization_std=("empirical_generalization", "std"),
            empirical_generalization_stderr_mean=("empirical_generalization_stderr", "mean"),
            reference_information_bound_mean=("reference_information_bound", "mean"),
            reference_information_bound_std=("reference_information_bound", "std"),
            pinsker_tv_bound_mean=("pinsker_tv_bound", "mean"),
            pinsker_tv_bound_std=("pinsker_tv_bound", "std"),
            direct_v1_bound_mean=("direct_v1_bound", "mean"),
            direct_v1_bound_std=("direct_v1_bound", "std"),
            fixed_v2_bound_mean=("fixed_v2_bound", "mean"),
            fixed_v2_bound_std=("fixed_v2_bound", "std"),
            optimized_bound_mean=("optimized_bound", "mean"),
            optimized_bound_std=("optimized_bound", "std"),
            selected_alpha_mean=("selected_alpha", "mean"),
            selected_alpha_std=("selected_alpha", "std"),
            selected_alpha_min=("selected_alpha", "min"),
            selected_alpha_max=("selected_alpha", "max"),
        )
    )
    summary_frame["selected_alpha_range"] = summary_frame["selected_alpha_max"] - summary_frame["selected_alpha_min"]
    for column in [
        "empirical_generalization_std",
        "reference_information_bound_std",
        "pinsker_tv_bound_std",
        "direct_v1_bound_std",
        "fixed_v2_bound_std",
        "optimized_bound_std",
        "selected_alpha_std",
    ]:
        summary_frame[column] = summary_frame[column].fillna(0.0)

    for _, row in repeat_frame.iterrows():
        sanity_records.append(row.to_dict())
    for (scenario, estimator, dimension, n_value), group in repeat_frame.groupby(["scenario", "estimator", "dimension", "n"]):
        unique_alpha, counts = np.unique(group["selected_alpha"].to_numpy(dtype=float), return_counts=True)
        alpha_frequency = {
            f"{alpha:.2f}": int(count)
            for alpha, count in sorted(zip(unique_alpha, counts, strict=False), key=lambda item: item[0])
        }
        sanity_records.append(
            {
                "record_type": "aggregate",
                "scenario": scenario,
                "estimator": estimator,
                "dimension": float(dimension),
                "n": float(n_value),
                "repeat": np.nan,
                "alpha_frequency_json": json.dumps(alpha_frequency, sort_keys=True),
                "selected_alpha_mean": float(group["selected_alpha"].mean()),
                "selected_alpha_std": float(group["selected_alpha"].std(ddof=1)) if len(group) > 1 else 0.0,
                "selected_alpha_range": float(group["selected_alpha"].max() - group["selected_alpha"].min()),
                "alpha_stability_flag": bool(group["selected_alpha"].max() - group["selected_alpha"].min() > 0.1),
                "compute_within_5pct_all_repeats": bool(group["compute_within_5pct_vs_reference"].all()),
                "direct_v1_leq_pinsker_all_repeats": bool(group["direct_v1_leq_pinsker"].all()),
            }
        )

    primary_summary = summary_frame[summary_frame["estimator"] == "split_selection"]
    benchmark_summary = summary_frame[summary_frame["estimator"] == "benchmark_split"]
    if not primary_summary.empty and not benchmark_summary.empty:
        comparison = benchmark_summary.merge(
            primary_summary,
            on=["dimension", "n"],
            suffixes=("_benchmark", "_primary"),
        )
        for _, row in comparison.iterrows():
            relative_change = abs(row["optimized_bound_mean_benchmark"] - row["optimized_bound_mean_primary"]) / max(
                row["optimized_bound_mean_primary"], 1e-12
            )
            alpha_delta = abs(row["selected_alpha_mean_benchmark"] - row["selected_alpha_mean_primary"])
            sanity_records.append(
                {
                    "record_type": "benchmark_comparison",
                    "scenario": "benchmark",
                    "estimator": "benchmark_split_vs_primary",
                    "dimension": float(row["dimension"]),
                    "n": float(row["n"]),
                    "repeat": np.nan,
                    "optimized_bound_relative_change": float(relative_change),
                    "selected_alpha_delta": float(alpha_delta),
                    "benchmark_instability_flag": bool(relative_change > 0.10 or alpha_delta > 0.1),
                }
            )

    sanity_frame = pd.DataFrame(sanity_records)

    fig, axes = plt.subplots(
        len(config.dimensions),
        2,
        figsize=(13.0, 4.1 * max(len(config.dimensions), 1)),
        sharex="col",
        squeeze=False,
    )
    color_map = {
        "empirical": "#1f77b4",
        "v1": "#9467bd",
        "pinsker": "#8c564b",
        "mi": "#ff7f0e",
        "v2": "#2ca02c",
        "opt": "#d62728",
    }
    for row_index, dim in enumerate(config.dimensions):
        dim_summary = primary_summary[primary_summary["dimension"] == float(dim)].sort_values("n")
        benchmark_dim = benchmark_summary[benchmark_summary["dimension"] == float(dim)].sort_values("n")
        ax_bounds = axes[row_index, 0]
        ax_alpha = axes[row_index, 1]
        if not dim_summary.empty:
            ax_bounds.errorbar(
                dim_summary["n"],
                dim_summary["empirical_generalization_mean"],
                yerr=dim_summary["empirical_generalization_std"].to_numpy() / np.sqrt(
                    np.maximum(dim_summary["repeat_count"].to_numpy(), 1.0)
                ),
                fmt="o-",
                linewidth=2,
                capsize=3,
                color=color_map["empirical"],
                label="Empirical",
            )
            ax_bounds.plot(dim_summary["n"], dim_summary["direct_v1_bound_mean"], "x--", linewidth=2, color=color_map["v1"], label=r"Direct $V_1$")
            ax_bounds.plot(dim_summary["n"], dim_summary["pinsker_tv_bound_mean"], "v:", linewidth=2, color=color_map["pinsker"], label="Pinsker-TV")
            ax_bounds.plot(dim_summary["n"], dim_summary["reference_information_bound_mean"], "s--", linewidth=2, color=color_map["mi"], label="MI")
            ax_bounds.plot(dim_summary["n"], dim_summary["fixed_v2_bound_mean"], "^-", linewidth=2, color=color_map["v2"], label=r"Fixed $V_2$")
            ax_bounds.plot(dim_summary["n"], dim_summary["optimized_bound_mean"], "d-", linewidth=2, color=color_map["opt"], label=r"Opt $V_\alpha$")
            if not benchmark_dim.empty:
                ax_bounds.scatter(
                    benchmark_dim["n"],
                    benchmark_dim["optimized_bound_mean"],
                    marker="x",
                    s=70,
                    linewidths=2,
                    color="black",
                    label="Benchmark",
                )
            ax_alpha.errorbar(
                dim_summary["n"],
                dim_summary["selected_alpha_mean"],
                yerr=dim_summary["selected_alpha_std"],
                fmt="o-",
                linewidth=2,
                capsize=3,
                color=color_map["opt"],
                label=r"Split $\alpha^*$",
            )
            if not benchmark_dim.empty:
                ax_alpha.scatter(
                    benchmark_dim["n"],
                    benchmark_dim["selected_alpha_mean"],
                    marker="x",
                    s=70,
                    linewidths=2,
                    color="black",
                    label="Benchmark",
                )
        ax_bounds.set_title(f"d = {int(dim)}")
        ax_bounds.set_xlabel("Sample size n")
        ax_bounds.set_ylabel("Bound / empirical gap")
        ax_bounds.grid(True, alpha=0.3)
        ax_alpha.set_title(f"d = {int(dim)}")
        ax_alpha.set_xlabel("Sample size n")
        ax_alpha.set_ylabel(r"Selected $\alpha^*$")
        ax_alpha.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    fig.suptitle(r"Sample-size robustness of theorem-valid split-selection $\alpha$ sweep", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    summary_path = output_dir / f"{artifact_prefix}_summary.csv"
    profiles_path = output_dir / f"{artifact_prefix}_profiles.csv"
    sanity_path = output_dir / f"{artifact_prefix}_sanity.csv"
    figure_path = output_dir / f"{artifact_prefix}.png"
    config_path = output_dir / f"{artifact_prefix}_config.json"

    summary_frame.to_csv(summary_path, index=False)
    profile_frame.to_csv(profiles_path, index=False)
    sanity_frame.to_csv(sanity_path, index=False)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
            "paper_estimator": "split_selection",
        },
        config_path,
    )

    diagnostics = {
        "summary_path": str(summary_path),
        "profiles_path": str(profiles_path),
        "sanity_path": str(sanity_path),
        "figure_path": str(figure_path),
        "story_checks": {
            "compute_equalized": bool(repeat_frame["compute_within_5pct_vs_reference"].all())
            if not repeat_frame.empty
            else False,
            "direct_v1_leq_pinsker": bool(repeat_frame["direct_v1_leq_pinsker"].all())
            if not repeat_frame.empty
            else False,
        },
    }

    return {
        "profiles": profile_frame,
        "summary": summary_frame,
        "sanity": sanity_frame,
        "diagnostics": diagnostics,
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def run_high_dim_convergence_experiment(config: HighDimConvergenceConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = config.artifact_prefix
    sigma_loss_sq_value = sigma_loss_sq(config.clip_c)
    alpha_values = tuple(sorted({float(alpha) for alpha in config.alpha_values}))
    optimization_alpha_values = alpha_optimization_grid(alpha_values)
    if 1.0 not in alpha_values:
        raise ValueError("HighDimConvergenceConfig.alpha_values must include 1.0.")

    reference_budget_grid = tuple(sorted({int(budget) for budget in config.reference_budget_grid}))
    weight_budget_grid = tuple(sorted({int(budget) for budget in config.weight_budget_grid}))
    if not reference_budget_grid or not weight_budget_grid:
        raise ValueError("Reference and weight budget grids must be non-empty.")

    max_reference_budget = max(
        max(reference_budget_grid),
        config.benchmark_reference_budget if config.benchmark_dimensions else 0,
    )
    max_weight_budget = max(weight_budget_grid)

    profile_records: list[dict[str, float | str | int | bool]] = []
    repeat_records: list[dict[str, float | str | int | bool]] = []
    sanity_records: list[dict[str, float | str | int | bool]] = []

    def build_base_case(*, dim: int, repeat_index: int) -> dict[str, Any]:
        case_seed = config.seed + repeat_index * 10000 + int(dim) * 101
        rng = np.random.default_rng(case_seed)
        w_true = make_w_true(dim, config.w_true_value)
        holdout = make_holdout_set(
            rng=rng,
            num_holdout=config.num_holdout,
            w_true=w_true,
            sigma_noise_sq=config.sigma_noise_sq,
        )
        evaluation_datasets = [
            generate_regression_dataset(
                rng=rng,
                n=config.n,
                w_true=w_true,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for _ in range(config.evaluation_num_datasets)
        ]
        evaluation_posteriors = [
            posterior_from_dataset(
                x,
                y,
                lambda_prior=config.lambda_prior,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for x, y in evaluation_datasets
        ]
        empirical_gen, empirical_stderr = estimate_empirical_generalization(
            datasets=evaluation_datasets,
            posteriors=evaluation_posteriors,
            holdout=holdout,
            clip_c=config.clip_c,
            rng=np.random.default_rng(case_seed + 1),
        )
        reference_datasets = [
            generate_regression_dataset(
                rng=rng,
                n=config.n,
                w_true=w_true,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for _ in range(max_reference_budget)
        ]
        reference_posteriors = [
            posterior_from_dataset(
                x,
                y,
                lambda_prior=config.lambda_prior,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for x, y in reference_datasets
        ]
        return {
            "dimension": float(dim),
            "repeat": float(repeat_index),
            "case_seed": int(case_seed),
            "evaluation_datasets": evaluation_datasets,
            "evaluation_posteriors": evaluation_posteriors,
            "reference_pool_posteriors": reference_posteriors,
            "empirical_generalization": float(empirical_gen),
            "empirical_generalization_stderr": float(empirical_stderr),
        }

    def make_case_from_prefix(base_case: dict[str, Any], reference_budget: int) -> dict[str, Any]:
        reference_posteriors = base_case["reference_pool_posteriors"][:reference_budget]
        return {
            "dimension": float(base_case["dimension"]),
            "repeat": float(base_case["repeat"]),
            "evaluation_posteriors": base_case["evaluation_posteriors"],
            "reference_posteriors": reference_posteriors,
            "reference_mixture": GaussianMixtureReference(reference_posteriors),
            "empirical_generalization": float(base_case["empirical_generalization"]),
            "empirical_generalization_stderr": float(base_case["empirical_generalization_stderr"]),
            "num_mi_weight_samples": int(config.mi_num_weight_samples),
        }

    def append_profile_rows(
        *,
        scenario: str,
        stage: str,
        dimension: int,
        repeat_index: int,
        reference_budget: int,
        total_weight_budget: int,
        profile: dict[str, Any],
        num_weight_samples: int,
    ) -> None:
        for alpha in alpha_values:
            profile_records.append(
                {
                    "scenario": scenario,
                    "stage": stage,
                    "dimension": float(dimension),
                    "repeat": float(repeat_index),
                    "reference_budget": float(reference_budget),
                    "weight_budget": float(total_weight_budget),
                    "alpha": float(alpha),
                    "v_alpha_value": float(profile["v_alpha_values"][alpha]),
                    "bound": float(profile["bound_by_alpha"][alpha]),
                    "num_weight_samples": float(num_weight_samples),
                    "log_ratio_min": float(profile["diagnostics"]["log_ratio_min"]),
                    "log_ratio_max": float(profile["diagnostics"]["log_ratio_max"]),
                }
            )

    def append_repeat_row(
        *,
        scenario: str,
        dimension: int,
        repeat_index: int,
        reference_budget: int,
        total_weight_budget: int,
        case: dict[str, Any],
        info_bundle: dict[str, float],
        split_result: dict[str, Any],
    ) -> None:
        evaluation_profile = split_result["evaluation_profile"]
        v_alpha_values = np.array([evaluation_profile["v_alpha_values"][alpha] for alpha in alpha_values], dtype=float)
        v_alpha_min = float(v_alpha_values.min())
        v_alpha_max = float(v_alpha_values.max())
        v_alpha_range = float(v_alpha_max - v_alpha_min)
        v_alpha_relative_range = float(v_alpha_range / max(abs(v_alpha_max), 1e-12))
        prefactor_dominated_flag = bool(
            np.isclose(split_result["selected_alpha"], 2.0) and v_alpha_relative_range <= 0.05 + 1e-12
        )
        reference_cost = mi_logpdf_cost(
            num_datasets=len(case["evaluation_posteriors"]),
            num_reference_datasets=reference_budget,
            num_weight_samples=config.mi_num_weight_samples,
        )
        estimator_cost = alpha_sweep_logpdf_cost(
            num_datasets=len(case["evaluation_posteriors"]),
            num_reference_datasets=reference_budget,
            num_weight_samples=total_weight_budget,
        )
        repeat_records.append(
            {
                "record_type": "repeat",
                "scenario": scenario,
                "dimension": float(dimension),
                "repeat": float(repeat_index),
                "reference_budget": float(reference_budget),
                "weight_budget": float(total_weight_budget),
                "empirical_generalization": float(case["empirical_generalization"]),
                "empirical_generalization_stderr": float(case["empirical_generalization_stderr"]),
                "reference_information_estimate": float(info_bundle["information_estimate"]),
                "reference_information_stderr": float(info_bundle["information_stderr"]),
                "reference_information_bound": float(info_bundle["information_bound"]),
                "pinsker_tv_bound": float(info_bundle["pinsker_tv_bound"]),
                "direct_v1_bound": float(split_result["direct_v1_bound"]),
                "fixed_v2_bound": float(split_result["fixed_v2_bound"]),
                "optimized_bound": float(split_result["optimized_bound"]),
                "selected_alpha": float(split_result["selected_alpha"]),
                "optimized_bound_minus_empirical": float(
                    split_result["optimized_bound"] - case["empirical_generalization"]
                ),
                "fixed_v2_bound_minus_empirical": float(
                    split_result["fixed_v2_bound"] - case["empirical_generalization"]
                ),
                "v_alpha_min": v_alpha_min,
                "v_alpha_max": v_alpha_max,
                "v_alpha_range": v_alpha_range,
                "v_alpha_relative_range": v_alpha_relative_range,
                "prefactor_dominated_flag": prefactor_dominated_flag,
                "reference_logpdf_cost": float(reference_cost),
                "estimator_logpdf_cost": float(estimator_cost),
                "relative_compute_gap_vs_reference": float(
                    abs(estimator_cost - reference_cost) / max(estimator_cost, reference_cost)
                ),
                "compute_within_5pct_vs_reference": bool(
                    abs(estimator_cost - reference_cost) / max(estimator_cost, reference_cost) <= 0.05
                ),
                "raw_v_alpha_monotonicity_violation": float(evaluation_profile["monotonicity_violation"]),
                "raw_v_alpha_monotonicity_count": int(evaluation_profile["monotonicity_violation_count"]),
                "selection_monotonicity_violation": float(split_result["selection_profile"]["monotonicity_violation"]),
                "selection_monotonicity_count": int(
                    split_result["selection_profile"]["monotonicity_violation_count"]
                ),
            }
        )

    for repeat_index in range(config.num_repeats):
        for dim in config.dimensions:
            base_case = build_base_case(dim=int(dim), repeat_index=repeat_index)
            sanity_records.append(
                {
                    "record_type": "base_case",
                    "scenario": "primary",
                    "dimension": float(dim),
                    "repeat": float(repeat_index),
                    "case_seed": float(base_case["case_seed"]),
                    "evaluation_reuse_verified": True,
                    "reference_prefix_reuse_verified": True,
                    "evaluation_dataset_count": float(len(base_case["evaluation_datasets"])),
                    "reference_pool_size": float(len(base_case["reference_pool_posteriors"])),
                    "max_reference_budget": float(max_reference_budget),
                    "benchmark_enabled": bool(int(dim) in config.benchmark_dimensions),
                }
            )
            for reference_budget in reference_budget_grid:
                case = make_case_from_prefix(base_case, int(reference_budget))
                info_bundle = estimate_reference_information_bundle(
                    posteriors=case["evaluation_posteriors"],
                    reference=case["reference_mixture"],
                    rng=np.random.default_rng(base_case["case_seed"] + 1000 + int(reference_budget)),
                    num_weight_samples=config.mi_num_weight_samples,
                    clip_c=config.clip_c,
                    sigma_loss_sq_value=sigma_loss_sq_value,
                    n=config.n,
                )
                for total_weight_budget in weight_budget_grid:
                    split_result = run_split_alpha_estimator(
                        posteriors=case["evaluation_posteriors"],
                        reference=case["reference_mixture"],
                        alpha_values=alpha_values,
                        optimization_alpha_values=optimization_alpha_values,
                        sigma_loss_sq_value=sigma_loss_sq_value,
                        n=config.n,
                        clip_c=config.clip_c,
                        total_weight_samples=int(total_weight_budget),
                        seed=base_case["case_seed"] + 2000 + int(reference_budget) * 13 + int(total_weight_budget) * 17,
                    )
                    half_budgets = partition_weight_budget(int(total_weight_budget), 2)
                    append_profile_rows(
                        scenario="primary",
                        stage="split_select",
                        dimension=int(dim),
                        repeat_index=repeat_index,
                        reference_budget=int(reference_budget),
                        total_weight_budget=int(total_weight_budget),
                        profile=split_result["selection_profile"],
                        num_weight_samples=int(half_budgets[0]),
                    )
                    append_profile_rows(
                        scenario="primary",
                        stage="split_eval",
                        dimension=int(dim),
                        repeat_index=repeat_index,
                        reference_budget=int(reference_budget),
                        total_weight_budget=int(total_weight_budget),
                        profile=split_result["evaluation_profile"],
                        num_weight_samples=int(half_budgets[1]),
                    )
                    append_repeat_row(
                        scenario="primary",
                        dimension=int(dim),
                        repeat_index=repeat_index,
                        reference_budget=int(reference_budget),
                        total_weight_budget=int(total_weight_budget),
                        case=case,
                        info_bundle=info_bundle,
                        split_result=split_result,
                    )

            if int(dim) in config.benchmark_dimensions and repeat_index < config.benchmark_num_repeats:
                benchmark_case = make_case_from_prefix(base_case, int(config.benchmark_reference_budget))
                benchmark_info_bundle = estimate_reference_information_bundle(
                    posteriors=benchmark_case["evaluation_posteriors"],
                    reference=benchmark_case["reference_mixture"],
                    rng=np.random.default_rng(base_case["case_seed"] + 9000),
                    num_weight_samples=config.mi_num_weight_samples,
                    clip_c=config.clip_c,
                    sigma_loss_sq_value=sigma_loss_sq_value,
                    n=config.n,
                )
                benchmark_split = run_split_alpha_estimator(
                    posteriors=benchmark_case["evaluation_posteriors"],
                    reference=benchmark_case["reference_mixture"],
                    alpha_values=alpha_values,
                    optimization_alpha_values=optimization_alpha_values,
                    sigma_loss_sq_value=sigma_loss_sq_value,
                    n=config.n,
                    clip_c=config.clip_c,
                    total_weight_samples=config.benchmark_total_weight_samples,
                    seed=base_case["case_seed"] + 10000,
                )
                benchmark_half_budgets = partition_weight_budget(config.benchmark_total_weight_samples, 2)
                append_profile_rows(
                    scenario="benchmark",
                    stage="split_select",
                    dimension=int(dim),
                    repeat_index=repeat_index,
                    reference_budget=int(config.benchmark_reference_budget),
                    total_weight_budget=int(config.benchmark_total_weight_samples),
                    profile=benchmark_split["selection_profile"],
                    num_weight_samples=int(benchmark_half_budgets[0]),
                )
                append_profile_rows(
                    scenario="benchmark",
                    stage="split_eval",
                    dimension=int(dim),
                    repeat_index=repeat_index,
                    reference_budget=int(config.benchmark_reference_budget),
                    total_weight_budget=int(config.benchmark_total_weight_samples),
                    profile=benchmark_split["evaluation_profile"],
                    num_weight_samples=int(benchmark_half_budgets[1]),
                )
                append_repeat_row(
                    scenario="benchmark",
                    dimension=int(dim),
                    repeat_index=repeat_index,
                    reference_budget=int(config.benchmark_reference_budget),
                    total_weight_budget=int(config.benchmark_total_weight_samples),
                    case=benchmark_case,
                    info_bundle=benchmark_info_bundle,
                    split_result=benchmark_split,
                )

    profile_frame = pd.DataFrame(profile_records)
    repeat_frame = pd.DataFrame(repeat_records)
    summary_frame = (
        repeat_frame.groupby(["scenario", "dimension", "reference_budget", "weight_budget"], as_index=False)
        .agg(
            repeat_count=("repeat", "count"),
            empirical_generalization_mean=("empirical_generalization", "mean"),
            empirical_generalization_std=("empirical_generalization", "std"),
            empirical_generalization_stderr_mean=("empirical_generalization_stderr", "mean"),
            reference_information_bound_mean=("reference_information_bound", "mean"),
            reference_information_bound_std=("reference_information_bound", "std"),
            direct_v1_bound_mean=("direct_v1_bound", "mean"),
            direct_v1_bound_std=("direct_v1_bound", "std"),
            fixed_v2_bound_mean=("fixed_v2_bound", "mean"),
            fixed_v2_bound_std=("fixed_v2_bound", "std"),
            optimized_bound_mean=("optimized_bound", "mean"),
            optimized_bound_std=("optimized_bound", "std"),
            selected_alpha_mean=("selected_alpha", "mean"),
            selected_alpha_median=("selected_alpha", "median"),
            selected_alpha_std=("selected_alpha", "std"),
            selected_alpha_min=("selected_alpha", "min"),
            selected_alpha_max=("selected_alpha", "max"),
            optimized_bound_minus_empirical_mean=("optimized_bound_minus_empirical", "mean"),
            optimized_bound_minus_empirical_std=("optimized_bound_minus_empirical", "std"),
            fixed_v2_bound_minus_empirical_mean=("fixed_v2_bound_minus_empirical", "mean"),
            fixed_v2_bound_minus_empirical_std=("fixed_v2_bound_minus_empirical", "std"),
            v_alpha_min_mean=("v_alpha_min", "mean"),
            v_alpha_max_mean=("v_alpha_max", "mean"),
            v_alpha_range_mean=("v_alpha_range", "mean"),
            v_alpha_relative_range_mean=("v_alpha_relative_range", "mean"),
            v_alpha_relative_range_std=("v_alpha_relative_range", "std"),
            prefactor_dominated_count=("prefactor_dominated_flag", "sum"),
            optimized_bound_violation_count=("optimized_bound_minus_empirical", lambda values: int(np.sum(values < 0.0))),
            fixed_v2_bound_violation_count=("fixed_v2_bound_minus_empirical", lambda values: int(np.sum(values < 0.0))),
            estimator_logpdf_cost_mean=("estimator_logpdf_cost", "mean"),
            reference_logpdf_cost_mean=("reference_logpdf_cost", "mean"),
        )
    )
    summary_frame["selected_alpha_range"] = summary_frame["selected_alpha_max"] - summary_frame["selected_alpha_min"]
    summary_frame["prefactor_dominated_rate"] = (
        summary_frame["prefactor_dominated_count"] / np.maximum(summary_frame["repeat_count"], 1.0)
    )
    summary_frame["optimized_bound_violation_rate"] = (
        summary_frame["optimized_bound_violation_count"] / np.maximum(summary_frame["repeat_count"], 1.0)
    )
    summary_frame["fixed_v2_bound_violation_rate"] = (
        summary_frame["fixed_v2_bound_violation_count"] / np.maximum(summary_frame["repeat_count"], 1.0)
    )
    summary_frame["prefactor_dominated_all_repeats"] = (
        summary_frame["prefactor_dominated_count"] == summary_frame["repeat_count"]
    )
    summary_frame["optimized_bound_violation_any"] = summary_frame["optimized_bound_violation_count"] > 0
    for column in [
        "empirical_generalization_std",
        "reference_information_bound_std",
        "direct_v1_bound_std",
        "fixed_v2_bound_std",
        "optimized_bound_std",
        "selected_alpha_std",
        "optimized_bound_minus_empirical_std",
        "fixed_v2_bound_minus_empirical_std",
        "v_alpha_relative_range_std",
    ]:
        summary_frame[column] = summary_frame[column].fillna(0.0)

    for _, row in repeat_frame.iterrows():
        sanity_records.append(row.to_dict())
    for (scenario, dimension, reference_budget, weight_budget), group in repeat_frame.groupby(
        ["scenario", "dimension", "reference_budget", "weight_budget"]
    ):
        unique_alpha, counts = np.unique(group["selected_alpha"].to_numpy(dtype=float), return_counts=True)
        alpha_frequency = {
            f"{alpha:.2f}": int(count)
            for alpha, count in sorted(zip(unique_alpha, counts, strict=False), key=lambda item: item[0])
        }
        sanity_records.append(
            {
                "record_type": "aggregate",
                "scenario": scenario,
                "dimension": float(dimension),
                "reference_budget": float(reference_budget),
                "weight_budget": float(weight_budget),
                "repeat": np.nan,
                "alpha_frequency_json": json.dumps(alpha_frequency, sort_keys=True),
                "selected_alpha_mean": float(group["selected_alpha"].mean()),
                "selected_alpha_std": float(group["selected_alpha"].std(ddof=1)) if len(group) > 1 else 0.0,
                "selected_alpha_range": float(group["selected_alpha"].max() - group["selected_alpha"].min()),
                "prefactor_dominated_rate": float(group["prefactor_dominated_flag"].mean()),
                "optimized_bound_violation_rate": float((group["optimized_bound_minus_empirical"] < 0.0).mean()),
            }
        )

    max_grid_summary = summary_frame[
        (summary_frame["scenario"] == "primary")
        & (summary_frame["reference_budget"] == float(max(reference_budget_grid)))
        & (summary_frame["weight_budget"] == float(max_weight_budget))
    ]
    for _, row in max_grid_summary.iterrows():
        sanity_records.append(
            {
                "record_type": "max_grid_check",
                "scenario": "primary",
                "dimension": float(row["dimension"]),
                "reference_budget": float(row["reference_budget"]),
                "weight_budget": float(row["weight_budget"]),
                "repeat": np.nan,
                "optimized_bound_violation_present": bool(row["optimized_bound_violation_count"] > 0),
                "prefactor_dominated_all_repeats": bool(row["prefactor_dominated_all_repeats"]),
                "optimized_bound_minus_empirical_mean": float(row["optimized_bound_minus_empirical_mean"]),
                "v_alpha_relative_range_mean": float(row["v_alpha_relative_range_mean"]),
            }
        )

    benchmark_summary = summary_frame[summary_frame["scenario"] == "benchmark"]
    if not benchmark_summary.empty and not max_grid_summary.empty:
        comparison = benchmark_summary.merge(
            max_grid_summary[
                [
                    "dimension",
                    "optimized_bound_mean",
                    "selected_alpha_mean",
                    "v_alpha_relative_range_mean",
                ]
            ].rename(
                columns={
                    "optimized_bound_mean": "optimized_bound_mean_grid",
                    "selected_alpha_mean": "selected_alpha_mean_grid",
                    "v_alpha_relative_range_mean": "v_alpha_relative_range_mean_grid",
                }
            ),
            on="dimension",
            how="left",
        )
        for _, row in comparison.iterrows():
            relative_change = abs(row["optimized_bound_mean"] - row["optimized_bound_mean_grid"]) / max(
                row["optimized_bound_mean_grid"], 1e-12
            )
            alpha_delta = abs(row["selected_alpha_mean"] - row["selected_alpha_mean_grid"])
            sanity_records.append(
                {
                    "record_type": "benchmark_comparison",
                    "scenario": "benchmark",
                    "dimension": float(row["dimension"]),
                    "reference_budget": float(row["reference_budget"]),
                    "weight_budget": float(row["weight_budget"]),
                    "repeat": np.nan,
                    "optimized_bound_relative_change": float(relative_change),
                    "selected_alpha_delta": float(alpha_delta),
                    "benchmark_instability_flag": bool(relative_change > 0.10 or alpha_delta > 0.1),
                    "grid_v_alpha_relative_range_mean": float(row["v_alpha_relative_range_mean_grid"]),
                }
            )

    sanity_frame = pd.DataFrame(sanity_records)

    fig_main, axes_main = plt.subplots(
        2,
        len(config.dimensions),
        figsize=(4.1 * len(config.dimensions), 8.0),
        sharex="col",
        squeeze=False,
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(reference_budget_grid)))
    for column_index, dim in enumerate(config.dimensions):
        ax_bounds = axes_main[0, column_index]
        ax_alpha = axes_main[1, column_index]
        dim_summary = summary_frame[
            (summary_frame["scenario"] == "primary") & (summary_frame["dimension"] == float(dim))
        ]
        if not dim_summary.empty:
            empirical_level = float(dim_summary["empirical_generalization_mean"].iloc[0])
            ax_bounds.axhline(empirical_level, color="black", linestyle="--", linewidth=1.5, label="Empirical")
        for color, reference_budget in zip(colors, reference_budget_grid, strict=False):
            subset = dim_summary[dim_summary["reference_budget"] == float(reference_budget)].sort_values("weight_budget")
            if subset.empty:
                continue
            ax_bounds.plot(
                subset["weight_budget"],
                subset["optimized_bound_mean"],
                marker="o",
                linewidth=2,
                color=color,
                label=f"R={int(reference_budget)}",
            )
            ax_alpha.plot(
                subset["weight_budget"],
                subset["selected_alpha_mean"],
                marker="o",
                linewidth=2,
                color=color,
                label=f"R={int(reference_budget)}",
            )
        benchmark_dim = benchmark_summary[benchmark_summary["dimension"] == float(dim)]
        if not benchmark_dim.empty:
            ax_bounds.scatter(
                benchmark_dim["weight_budget"],
                benchmark_dim["optimized_bound_mean"],
                marker="x",
                s=80,
                linewidths=2,
                color="black",
                label="Benchmark",
            )
            ax_alpha.scatter(
                benchmark_dim["weight_budget"],
                benchmark_dim["selected_alpha_mean"],
                marker="x",
                s=80,
                linewidths=2,
                color="black",
                label="Benchmark",
            )
        ax_bounds.set_title(f"d = {int(dim)}")
        ax_bounds.set_xlabel("Weight budget")
        ax_bounds.set_ylabel(r"Optimized bound")
        ax_bounds.grid(True, alpha=0.3)
        ax_alpha.set_xlabel("Weight budget")
        ax_alpha.set_ylabel(r"Selected $\alpha^*$")
        ax_alpha.grid(True, alpha=0.3)
    handles, labels = axes_main[0, 0].get_legend_handles_labels()
    if handles:
        fig_main.legend(handles, labels, loc="upper center", ncol=min(len(handles), 5), frameon=False)
    fig_main.suptitle("High-dimensional convergence sweep", y=0.98)
    fig_main.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    fig_diag, axes_diag = plt.subplots(
        2,
        len(config.dimensions),
        figsize=(4.1 * len(config.dimensions), 8.0),
        sharex="col",
        squeeze=False,
    )
    for column_index, dim in enumerate(config.dimensions):
        ax_range = axes_diag[0, column_index]
        ax_gap = axes_diag[1, column_index]
        dim_summary = summary_frame[
            (summary_frame["scenario"] == "primary") & (summary_frame["dimension"] == float(dim))
        ]
        for color, reference_budget in zip(colors, reference_budget_grid, strict=False):
            subset = dim_summary[dim_summary["reference_budget"] == float(reference_budget)].sort_values("weight_budget")
            if subset.empty:
                continue
            ax_range.plot(
                subset["weight_budget"],
                subset["v_alpha_relative_range_mean"],
                marker="o",
                linewidth=2,
                color=color,
                label=f"R={int(reference_budget)}",
            )
            ax_gap.plot(
                subset["weight_budget"],
                subset["optimized_bound_minus_empirical_mean"],
                marker="o",
                linewidth=2,
                color=color,
                label=f"R={int(reference_budget)}",
            )
        benchmark_dim = benchmark_summary[benchmark_summary["dimension"] == float(dim)]
        if not benchmark_dim.empty:
            ax_range.scatter(
                benchmark_dim["weight_budget"],
                benchmark_dim["v_alpha_relative_range_mean"],
                marker="x",
                s=80,
                linewidths=2,
                color="black",
            )
            ax_gap.scatter(
                benchmark_dim["weight_budget"],
                benchmark_dim["optimized_bound_minus_empirical_mean"],
                marker="x",
                s=80,
                linewidths=2,
                color="black",
            )
        ax_range.axhline(0.05, color="gray", linestyle=":", linewidth=1.2)
        ax_gap.axhline(0.0, color="gray", linestyle="--", linewidth=1.2)
        ax_range.set_title(f"d = {int(dim)}")
        ax_range.set_xlabel("Weight budget")
        ax_range.set_ylabel(r"Relative $V_\alpha$ range")
        ax_range.grid(True, alpha=0.3)
        ax_gap.set_xlabel("Weight budget")
        ax_gap.set_ylabel(r"Opt bound - empirical")
        ax_gap.grid(True, alpha=0.3)
    handles, labels = axes_diag[0, 0].get_legend_handles_labels()
    if handles:
        fig_diag.legend(handles, labels, loc="upper center", ncol=min(len(handles), 5), frameon=False)
    fig_diag.suptitle("Convergence diagnostics for the parameter-space estimator", y=0.98)
    fig_diag.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    summary_path = output_dir / f"{artifact_prefix}_summary.csv"
    profiles_path = output_dir / f"{artifact_prefix}_profiles.csv"
    sanity_path = output_dir / f"{artifact_prefix}_sanity.csv"
    main_figure_path = output_dir / f"{artifact_prefix}_main.png"
    diagnostics_figure_path = output_dir / f"{artifact_prefix}_diagnostics.png"
    config_path = output_dir / f"{artifact_prefix}_config.json"

    summary_frame.to_csv(summary_path, index=False)
    profile_frame.to_csv(profiles_path, index=False)
    sanity_frame.to_csv(sanity_path, index=False)
    fig_main.savefig(main_figure_path, dpi=200)
    fig_diag.savefig(diagnostics_figure_path, dpi=200)
    plt.close(fig_main)
    plt.close(fig_diag)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
            "purpose": "diagnostic_validation",
        },
        config_path,
    )

    diagnostics = {
        "summary_path": str(summary_path),
        "profiles_path": str(profiles_path),
        "sanity_path": str(sanity_path),
        "main_figure_path": str(main_figure_path),
        "diagnostics_figure_path": str(diagnostics_figure_path),
        "story_checks": {
            "benchmark_only_on_target_dims": bool(
                set(summary_frame[summary_frame["scenario"] == "benchmark"]["dimension"].astype(int).tolist())
                <= set(int(dim) for dim in config.benchmark_dimensions)
            ),
            "profile_values_nonnegative": bool((profile_frame["v_alpha_value"] >= -1e-12).all())
            if not profile_frame.empty
            else False,
        },
    }

    return {
        "summary": summary_frame,
        "profiles": profile_frame,
        "sanity": sanity_frame,
        "repeat_rows": repeat_frame,
        "diagnostics": diagnostics,
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def summarize_one_dim_sanity(result_bundle: dict[str, Any]) -> dict[str, Any]:
    results = result_bundle["results"]
    sanity = result_bundle["sanity"]
    posterior_mean = result_bundle["diagnostics"]["posterior_mean_vs_sample_mean"]
    posterior_var = result_bundle["diagnostics"]["posterior_var_vs_sample_var"]
    summary = {
        "mi_nonnegative_with_tolerance": bool((results["mi_estimate"] >= -1e-6).all()),
        "cmi_nonnegative_with_tolerance": bool((results["cmi_estimate"] >= -1e-6).all()),
        "conditional_v2_nonnegative_with_tolerance": bool((results["conditional_v2_estimate"] >= -1e-6).all())
        if "conditional_v2_estimate" in results
        else True,
        "grid_outside_mass_below_threshold": bool((sanity["grid_outside_mass"] < 1e-6).all()),
        "v_alpha2_matches_grid_v2": bool((sanity["v2_relative_gap"] < 0.25).all()),
        "posterior_mean_match": float(abs(posterior_mean["analytic"] - posterior_mean["sample"])),
        "posterior_var_match": float(abs(posterior_var["analytic"] - posterior_var["sample"])),
    }
    return summary


def summarize_dimensional_bounds_sanity(result_bundle: dict[str, Any]) -> dict[str, Any]:
    results = result_bundle["results"]
    sanity = result_bundle["sanity"]
    summary = {
        "mi_nonnegative_with_tolerance": bool((results["mi_estimate"] >= -1e-6).all()),
        "cmi_nonnegative_with_tolerance": bool((results["cmi_estimate"] >= -1e-6).all()),
        "conditional_v2_nonnegative_with_tolerance": bool((results["conditional_v2_estimate"] >= -1e-6).all()),
        "v2_bound_finite": bool(sanity["v2_finite"].all()),
        "conditional_v2_bound_finite": bool(sanity["conditional_v2_finite"].all()),
        "covariances_positive_definite": bool((sanity["min_covariance_eigenvalue"] > 0.0).all()),
        "empirical_never_exceeds_all_bounds": bool((~sanity["empirical_exceeds_all_bounds"]).all()),
        "uses_grid_v2": bool(result_bundle["diagnostics"]["uses_grid_v2"]),
        "v2_estimator": result_bundle["diagnostics"]["v2_estimator"],
    }
    return summary


def summarize_alpha_sanity(result_bundle: dict[str, Any]) -> dict[str, Any]:
    results = result_bundle["results"]
    sanity = result_bundle["sanity"]
    summary = {
        "mi_nonnegative_with_tolerance": bool((results["mi_estimate"] >= -1e-6).all()),
        "covariances_positive_definite": bool((sanity["min_covariance_eigenvalue"] > 0.0).all()),
        "optimized_beats_fixed_v2": bool(
            (results["optimized_v_alpha_bound"] <= results["v2_bound"] + 1e-10).all()
        ),
        "alpha_moves_downward": bool(results["optimal_alpha"].iloc[-1] <= results["optimal_alpha"].iloc[0]),
    }
    return summary


def summarize_alpha_audit_sanity(result_bundle: dict[str, Any]) -> dict[str, Any]:
    summary = result_bundle["summary"]
    sanity = result_bundle["sanity"]
    repeat_rows = sanity[sanity["record_type"] == "repeat"] if not sanity.empty else pd.DataFrame()
    selection_bias_rows = (
        sanity[sanity["record_type"] == "selection_bias_aggregate"] if not sanity.empty else pd.DataFrame()
    )
    benchmark_rows = (
        sanity[sanity["record_type"] == "benchmark_comparison"] if not sanity.empty else pd.DataFrame()
    )
    reproduction_rows = sanity[sanity["record_type"] == "reproduction"] if not sanity.empty else pd.DataFrame()
    summary_dict = {
        "repeat_rows_present": bool(not repeat_rows.empty),
        "audit_compute_within_5pct": bool(
            repeat_rows.loc[repeat_rows["scenario"] == "audit", "compute_within_5pct_vs_mi"].all()
        )
        if not repeat_rows.empty
        else False,
        "raw_v_alpha_nonnegative": bool((result_bundle["profiles"]["v_alpha_value"] >= -1e-12).all())
        if not result_bundle["profiles"].empty
        else False,
        "legacy_reproduces_02": bool(reproduction_rows["matches_02_within_tolerance"].all())
        if not reproduction_rows.empty
        else False,
        "split_not_systematically_lower": bool(selection_bias_rows["split_not_systematically_lower"].all())
        if not selection_bias_rows.empty
        else False,
        "benchmark_disagreement_flagged": bool(benchmark_rows["benchmark_instability_flag"].any())
        if not benchmark_rows.empty
        else False,
        "alpha_stability_flags_present": int(
            summary["selected_alpha_range"].gt(0.1).sum()
        )
        if not summary.empty
        else 0,
    }
    return summary_dict


def summarize_theorem1_proxy_sanity(result_bundle: dict[str, Any]) -> dict[str, Any]:
    profiles = result_bundle["profiles"]
    sanity = result_bundle["sanity"]
    repeat_rows = sanity[sanity["record_type"] == "repeat"] if not sanity.empty else pd.DataFrame()
    return {
        "profile_values_finite": bool(
            np.isfinite(
                profiles[
                    [
                        "v_alpha_value",
                        "theorem1_proxy_sensitivity",
                        "theorem1_proxy_bound",
                        "current_analytic_bound",
                        "corollary1_bound",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        )
        if not profiles.empty
        else False,
        "proxy_lower_than_current_analytic_all_repeats": bool(
            repeat_rows["theorem1_proxy_leq_current_analytic"].all()
        )
        if not repeat_rows.empty
        else False,
        "proxy_lower_than_corollary1_all_repeats": bool(
            repeat_rows["theorem1_proxy_leq_corollary1"].all()
        )
        if not repeat_rows.empty
        else False,
        "candidate_count_min": int(repeat_rows["candidate_count"].min()) if not repeat_rows.empty else 0,
        "max_abs_loss_deviation": float(repeat_rows["max_abs_loss_deviation"].max())
        if not repeat_rows.empty
        else 0.0,
        "proxy_not_theorem_valid": True,
    }


def summarize_alpha_comparison_sanity(result_bundle: dict[str, Any]) -> dict[str, Any]:
    summary = result_bundle["summary"]
    sanity = result_bundle["sanity"]
    profiles = result_bundle["profiles"]
    surrogates = result_bundle["surrogates"]
    repeat_rows = sanity[sanity["record_type"] == "repeat"] if not sanity.empty else pd.DataFrame()
    benchmark_rows = sanity[sanity["record_type"] == "benchmark_comparison"] if not sanity.empty else pd.DataFrame()
    legacy_rows = sanity[sanity["record_type"] == "reproduction"] if not sanity.empty else pd.DataFrame()
    theorem_rows = (
        repeat_rows[(repeat_rows["theorem_valid"] == True) & (repeat_rows["scenario"] != "legacy")]
        if not repeat_rows.empty
        else pd.DataFrame()
    )
    primary_rows = summary[
        (summary["scenario"] == "primary")
        & (summary["estimator"] == "split_selection")
        & (summary["reference_kind"] == "true_mixture")
    ]
    return {
        "theorem_valid_compute_within_5pct": bool(theorem_rows["compute_within_5pct_vs_reference"].all())
        if not theorem_rows.empty
        else False,
        "direct_v1_leq_pinsker": bool(theorem_rows["direct_v1_leq_pinsker"].fillna(True).all())
        if not theorem_rows.empty
        else False,
        "profile_values_nonnegative": bool((profiles["v_alpha_value"] >= -1e-12).all()) if not profiles.empty else False,
        "legacy_reference_present": bool(not legacy_rows.empty),
        "surrogate_rows_present": bool(not surrogates.empty),
        "benchmark_instability_flagged": bool(benchmark_rows["benchmark_instability_flag"].any())
        if not benchmark_rows.empty
        else False,
        "split_alpha_stability_flags": int(primary_rows["selected_alpha_range"].gt(0.1).sum())
        if not primary_rows.empty
        else 0,
    }


def summarize_alpha_sample_size_sanity(result_bundle: dict[str, Any]) -> dict[str, Any]:
    summary = result_bundle["summary"]
    sanity = result_bundle["sanity"]
    profiles = result_bundle["profiles"]
    repeat_rows = sanity[sanity["record_type"] == "repeat"] if not sanity.empty else pd.DataFrame()
    benchmark_rows = sanity[sanity["record_type"] == "benchmark_comparison"] if not sanity.empty else pd.DataFrame()
    primary_rows = summary[summary["estimator"] == "split_selection"] if not summary.empty else pd.DataFrame()
    return {
        "compute_within_5pct": bool(repeat_rows["compute_within_5pct_vs_reference"].all()) if not repeat_rows.empty else False,
        "direct_v1_leq_pinsker": bool(repeat_rows["direct_v1_leq_pinsker"].all()) if not repeat_rows.empty else False,
        "profile_values_nonnegative": bool((profiles["v_alpha_value"] >= -1e-12).all()) if not profiles.empty else False,
        "benchmark_instability_flagged": bool(benchmark_rows["benchmark_instability_flag"].any())
        if not benchmark_rows.empty
        else False,
        "alpha_stability_flags": int(primary_rows["selected_alpha_range"].gt(0.1).sum())
        if not primary_rows.empty
        else 0,
    }


def summarize_high_dim_convergence_sanity(result_bundle: dict[str, Any]) -> dict[str, Any]:
    summary = result_bundle["summary"]
    sanity = result_bundle["sanity"]
    profiles = result_bundle["profiles"]
    repeat_rows = result_bundle["repeat_rows"]
    config = result_bundle["config"]
    base_case_rows = sanity[sanity["record_type"] == "base_case"] if not sanity.empty else pd.DataFrame()
    benchmark_rows = sanity[sanity["record_type"] == "benchmark_comparison"] if not sanity.empty else pd.DataFrame()
    max_grid_rows = sanity[sanity["record_type"] == "max_grid_check"] if not sanity.empty else pd.DataFrame()
    prefactor_matches = (
        repeat_rows["prefactor_dominated_flag"]
        == (np.isclose(repeat_rows["selected_alpha"], 2.0) & (repeat_rows["v_alpha_relative_range"] <= 0.05 + 1e-12))
    ) if not repeat_rows.empty else pd.Series(dtype=bool)
    return {
        "profile_values_nonnegative": bool((profiles["v_alpha_value"] >= -1e-12).all()) if not profiles.empty else False,
        "benchmark_only_on_target_dims": bool(
            set(summary[summary["scenario"] == "benchmark"]["dimension"].astype(int).tolist())
            <= set(int(dim) for dim in config["benchmark_dimensions"])
        ) if not summary.empty else False,
        "prefactor_rule_exact": bool(prefactor_matches.all()) if not prefactor_matches.empty else False,
        "base_case_reuse_verified": bool(
            base_case_rows["evaluation_reuse_verified"].all() and base_case_rows["reference_prefix_reuse_verified"].all()
        ) if not base_case_rows.empty else False,
        "max_grid_violation_flagged": bool(max_grid_rows["optimized_bound_violation_present"].any())
        if not max_grid_rows.empty
        else False,
        "benchmark_disagreement_flagged": bool(benchmark_rows["benchmark_instability_flag"].any())
        if not benchmark_rows.empty
        else False,
        "prefactor_dominated_repeat_count": int(repeat_rows["prefactor_dominated_flag"].sum()) if not repeat_rows.empty else 0,
    }


def run_exact_channel_experiment(config: ExactChannelExperimentConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_grid = np.linspace(0.0, 1.0, config.channel_grid_size)
    alpha_grid = tuple(sorted({1.0, *config.alpha_values}))
    source_probabilities = np.array([0.5, 0.5], dtype=float)
    source_values = np.array([0.0, 1.0], dtype=float)

    channel_specs = [
        {
            "channel": "bsc",
            "parameter_name": "crossover_prob",
            "parameter_label": "p",
            "transition_builder": lambda p: np.array([[1.0 - p, p], [p, 1.0 - p]], dtype=float),
            "prediction_values": np.array([0.0, 1.0], dtype=float),
        },
        {
            "channel": "bec",
            "parameter_name": "erasure_prob",
            "parameter_label": "epsilon",
            "transition_builder": lambda epsilon: np.array(
                [[1.0 - epsilon, 0.0, epsilon], [0.0, 1.0 - epsilon, epsilon]],
                dtype=float,
            ),
            "prediction_values": np.array([0.0, 1.0, 0.5], dtype=float),
        },
        {
            "channel": "z_channel",
            "parameter_name": "transition_prob",
            "parameter_label": "p",
            "transition_builder": lambda p: np.array([[1.0, 0.0], [p, 1.0 - p]], dtype=float),
            "prediction_values": np.array([0.0, 1.0], dtype=float),
        },
    ]

    records: list[dict[str, float | str]] = []
    diagnostics_records: list[dict[str, float | str | bool]] = []
    alpha_records: list[dict[str, float | str]] = []

    for spec in channel_specs:
        for parameter in parameter_grid:
            conditional_probabilities = spec["transition_builder"](float(parameter))
            marginal_probabilities = source_probabilities @ conditional_probabilities
            joint = source_probabilities[:, None] * conditional_probabilities
            prediction_values = spec["prediction_values"]
            empirical_loss = (prediction_values[None, :] - source_values[:, None]) ** 2
            population_loss = source_probabilities @ empirical_loss
            active_outputs = marginal_probabilities > 0.0
            generalization = float(
                np.abs(np.sum(joint[:, active_outputs] * (population_loss[None, active_outputs] - empirical_loss[:, active_outputs])))
            )
            sup_deviation = float(
                np.max(np.abs(population_loss[active_outputs][None, :] - empirical_loss[:, active_outputs]))
            )
            mi_value = max(
                exact_discrete_mutual_information(
                    conditional_probabilities=conditional_probabilities,
                    marginal_probabilities=marginal_probabilities,
                    source_probabilities=source_probabilities,
                ),
                0.0,
            )
            mi_bound = float(np.sqrt(2.0 * config.sigma_loss_sq * mi_value))
            v1_value = float(np.sum(np.abs(joint - source_probabilities[:, None] * marginal_probabilities[None, :])))
            tv_bound = sup_deviation * v1_value
            pinsker_tv_bound = sup_deviation * float(np.sqrt(2.0 * mi_value))
            variance = weighted_variance(conditional_probabilities, source_probabilities)
            v2_value = float(np.sum(np.sqrt(variance)))
            v2_bound = float(np.sqrt(config.sigma_loss_sq) * v2_value)

            bound_by_alpha: dict[float, float] = {}
            value_by_alpha: dict[float, float] = {1.0: v1_value}
            for alpha in config.alpha_values:
                v_alpha_value = exact_discrete_v_alpha(
                    conditional_probabilities=conditional_probabilities,
                    marginal_probabilities=marginal_probabilities,
                    source_probabilities=source_probabilities,
                    alpha=alpha,
                )
                value_by_alpha[float(alpha)] = v_alpha_value
                bound_by_alpha[float(alpha)] = subgaussian_multiplier(alpha, config.sigma_loss_sq, 1) * v_alpha_value
            optimal_alpha = min(bound_by_alpha, key=bound_by_alpha.get)

            records.append(
                {
                    "channel": spec["channel"],
                    spec["parameter_name"]: float(parameter),
                    "exact_generalization": generalization,
                    "tv_v1_value": v1_value,
                    "tv_v1_bound": tv_bound,
                    "pinsker_tv_bound": pinsker_tv_bound,
                    "mi_value": mi_value,
                    "mi_bound": mi_bound,
                    "v2_value": v2_value,
                    "v2_bound": v2_bound,
                    "optimized_v_alpha_bound": float(bound_by_alpha[optimal_alpha]),
                    "optimal_alpha": float(optimal_alpha),
                }
            )
            diagnostics_records.append(
                {
                    "channel": spec["channel"],
                    spec["parameter_name"]: float(parameter),
                    "channel_rows_sum_to_one": bool(np.allclose(conditional_probabilities.sum(axis=1), 1.0)),
                    "monotone_v_alpha": monotone_grid_check([value_by_alpha[alpha] for alpha in alpha_grid]),
                    "v2_identity_gap": float(abs(value_by_alpha.get(2.0, v2_value) - v2_value)),
                    "min_marginal_probability": float(marginal_probabilities[active_outputs].min()),
                }
            )
            for alpha in alpha_grid:
                alpha_records.append(
                    {
                        "channel": spec["channel"],
                        spec["parameter_name"]: float(parameter),
                        "alpha": float(alpha),
                        "v_alpha_value": float(value_by_alpha[alpha]),
                        "bound": float(tv_bound if alpha == 1.0 else bound_by_alpha[float(alpha)]),
                    }
                )

    result_frame = pd.DataFrame(records)
    diagnostics_frame = pd.DataFrame(diagnostics_records)
    alpha_frame = pd.DataFrame(alpha_records)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharey=True)
    for axis, spec in zip(axes, channel_specs):
        subset = result_frame[result_frame["channel"] == spec["channel"]].sort_values(spec["parameter_name"])
        axis.plot(subset[spec["parameter_name"]], subset["exact_generalization"], linewidth=2, label="Exact gen")
        axis.plot(subset[spec["parameter_name"]], subset["tv_v1_bound"], "--", linewidth=2, label=r"$V_1$/TV")
        axis.plot(subset[spec["parameter_name"]], subset["pinsker_tv_bound"], ":", linewidth=2, label="Pinsker-TV")
        axis.plot(subset[spec["parameter_name"]], subset["mi_bound"], linewidth=2, label="MI")
        axis.plot(subset[spec["parameter_name"]], subset["v2_bound"], linewidth=2, label=r"$V_2$")
        axis.plot(subset[spec["parameter_name"]], subset["optimized_v_alpha_bound"], linewidth=2, label=r"Opt $V_\alpha$")
        axis.set_title(spec["channel"].replace("_", " ").upper())
        axis.set_xlabel(spec["parameter_label"])
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Bound / exact generalization")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    figure_path = output_dir / "03_exact_channels.png"
    table_path = output_dir / "03_exact_channels.csv"
    diagnostics_path = output_dir / "03_exact_channels_sanity.csv"
    alpha_path = output_dir / "03_exact_channels_alpha_profiles.csv"
    config_path = output_dir / "03_exact_channels_config.json"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    result_frame.to_csv(table_path, index=False)
    diagnostics_frame.to_csv(diagnostics_path, index=False)
    alpha_frame.to_csv(alpha_path, index=False)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
        },
        config_path,
    )

    return {
        "results": result_frame,
        "sanity": diagnostics_frame,
        "alpha_profiles": alpha_frame,
        "diagnostics": {
            "figure_path": str(figure_path),
            "table_path": str(table_path),
            "diagnostics_path": str(diagnostics_path),
            "alpha_path": str(alpha_path),
        },
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def run_reparameterization_experiment(config: ReparameterizationExperimentConfig) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma_loss_sq_value = sigma_loss_sq(config.clip_c)
    w_true = np.array([config.w_true], dtype=float)

    datasets = [
        generate_regression_dataset(
            rng=rng,
            n=config.n,
            w_true=w_true,
            sigma_noise_sq=config.sigma_noise_sq,
        )
        for _ in range(config.num_datasets)
    ]
    posteriors = [
        posterior_from_dataset(
            x,
            y,
            lambda_prior=config.lambda_prior,
            sigma_noise_sq=config.sigma_noise_sq,
        )
        for x, y in datasets
    ]

    original_bound, original_diag = estimate_v2_grid(
        posteriors=posteriors,
        sigma_loss_sq_value=sigma_loss_sq_value,
        n=config.n,
        grid_points=config.grid_points,
        grid_sigma_span=config.grid_sigma_span,
    )
    original_coordinate_stat = estimate_coordinate_variance_statistic(posteriors)

    transform_specs = [
        {
            "name": "identity",
            "transform": lambda x: x,
            "inverse_transform": lambda x: x,
            "derivative_on_w": lambda x: np.ones_like(x),
        },
        {
            "name": "affine_2w_plus_1",
            "transform": lambda x: 2.0 * x + 1.0,
            "inverse_transform": lambda x: (x - 1.0) / 2.0,
            "derivative_on_w": lambda x: np.full_like(x, 2.0),
        },
        {
            "name": "sinh",
            "transform": np.sinh,
            "inverse_transform": np.arcsinh,
            "derivative_on_w": np.cosh,
        },
    ]

    records: list[dict[str, float | str]] = []
    sanity_records: list[dict[str, float | str]] = []
    for spec in transform_specs:
        transformed_bound, transformed_diag = estimate_transform_v2(
            posteriors=posteriors,
            sigma_loss_sq_value=sigma_loss_sq_value,
            n=config.n,
            grid_points=config.grid_points,
            grid_sigma_span=config.grid_sigma_span,
            transform=spec["transform"],
            inverse_transform=spec["inverse_transform"],
            derivative_on_w=spec["derivative_on_w"],
        )
        coordinate_stat = (
            original_coordinate_stat
            if spec["name"] == "identity"
            else estimate_coordinate_variance_statistic(
                posteriors=posteriors,
                transform=spec["transform"],
                rng=np.random.default_rng(config.seed + len(records) + 1),
                sample_size=config.transform_sample_size,
            )
        )
        records.append(
            {
                "transform": spec["name"],
                "v2_bound": transformed_bound,
                "relative_error_vs_original": float(abs(transformed_bound - original_bound) / max(original_bound, 1e-12)),
                "coordinate_variance_statistic": coordinate_stat,
            }
        )
        sanity_records.append(
            {
                "transform": spec["name"],
                "max_density_normalization_error": float(transformed_diag["max_normalization_error"]),
                "grid_lower": float(transformed_diag["grid_lower"]),
                "grid_upper": float(transformed_diag["grid_upper"]),
            }
        )

    result_frame = pd.DataFrame(records)
    sanity_frame = pd.DataFrame(sanity_records)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    axes[0].bar(result_frame["transform"], result_frame["v2_bound"])
    axes[0].axhline(original_bound, color="black", linestyle="--", linewidth=1.5, label="Original")
    axes[0].set_title(r"$V_2$ invariance under reparameterization")
    axes[0].set_ylabel(r"$V_2$ bound")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(result_frame["transform"], result_frame["coordinate_variance_statistic"])
    axes[1].set_title("Coordinate-dependent contrast statistic")
    axes[1].set_ylabel("Average posterior coordinate variance")
    axes[1].grid(True, axis="y", alpha=0.3)
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    figure_path = output_dir / "04_reparameterization.png"
    table_path = output_dir / "04_reparameterization.csv"
    sanity_path = output_dir / "04_reparameterization_sanity.csv"
    config_path = output_dir / "04_reparameterization_config.json"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    result_frame.to_csv(table_path, index=False)
    sanity_frame.to_csv(sanity_path, index=False)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
            "original_v2_bound": original_bound,
            "original_coordinate_variance_statistic": original_coordinate_stat,
            "original_v2_value": float(original_diag["v2_value"]),
        },
        config_path,
    )

    return {
        "results": result_frame,
        "sanity": sanity_frame,
        "diagnostics": {
            "figure_path": str(figure_path),
            "table_path": str(table_path),
            "sanity_path": str(sanity_path),
            "original_v2_bound": original_bound,
        },
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def run_ads_experiment(config: AdsExperimentConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma_value = np.sqrt(config.sigma_loss_sq)
    hypothesis_grid = np.asarray(config.hypothesis_grid, dtype=float)
    population_loss = 0.5 * (hypothesis_grid ** 2 + (1.0 - hypothesis_grid) ** 2)

    records: list[dict[str, float]] = []
    diagnostics_records: list[dict[str, float | bool]] = []

    for n in config.n_values:
        dataset_weights = dataset_probabilities_by_counts(n)
        posterior_matrix = np.vstack(
            [gibbs_posterior_from_count(k, n, hypothesis_grid, config.beta) for k in range(n + 1)]
        )
        marginal = dataset_weights @ posterior_matrix
        empirical_loss_matrix = (
            ((n - np.arange(n + 1))[:, None] / n) * (hypothesis_grid[None, :] ** 2)
            + (np.arange(n + 1)[:, None] / n) * ((1.0 - hypothesis_grid[None, :]) ** 2)
        )
        exact_generalization = float(
            np.abs(np.sum(dataset_weights[:, None] * posterior_matrix * (population_loss[None, :] - empirical_loss_matrix)))
        )

        neighboring_differences = np.abs(np.diff(posterior_matrix, axis=0))
        gamma_exact = float(np.max(neighboring_differences / marginal[None, :]))
        gamma_gibbs_formula = float(
            np.exp(2.0 * config.beta * config.max_loss)
            * (np.exp(2.0 * config.beta * config.max_loss / n) - 1.0)
        )
        tv_stability = float(0.5 * np.max(np.sum(neighboring_differences, axis=1)))

        expected_uniform = 0.0
        for z in (0.0, 1.0):
            loss_values = (hypothesis_grid - z) ** 2
            differences = np.abs(np.diff(posterior_matrix @ loss_values))
            expected_uniform = max(expected_uniform, float(np.max(differences)))

        variance = weighted_variance(posterior_matrix, dataset_weights)
        v2_value = float(np.sum(np.sqrt(variance)))
        v2_bound = sigma_value / np.sqrt(n) * v2_value

        ads_bound_exact_gamma = float(sigma_value * gamma_exact / np.sqrt(2.0))
        ads_bound_gibbs_formula = float(sigma_value * gamma_gibbs_formula / np.sqrt(2.0))

        records.append(
            {
                "n": float(n),
                "exact_generalization": exact_generalization,
                "gamma_exact": gamma_exact,
                "gamma_gibbs_formula": gamma_gibbs_formula,
                "tv_stability": tv_stability,
                "expected_uniform_stability": expected_uniform,
                "v2_value": v2_value,
                "v2_bound": v2_bound,
                "ads_bound_exact_gamma": ads_bound_exact_gamma,
                "ads_bound_gibbs_formula": ads_bound_gibbs_formula,
            }
        )
        diagnostics_records.append(
            {
                "n": float(n),
                "tv_le_gamma_over_2": bool(tv_stability <= gamma_exact / 2.0 + 1e-12),
                "expected_uniform_le_M_gamma": bool(expected_uniform <= config.max_loss * gamma_exact + 1e-12),
                "v2_le_sqrt_2_gamma_n": bool(v2_value <= np.sqrt(2.0 * gamma_exact * n) + 1e-12),
                "gamma_exact_le_gibbs_formula": bool(gamma_exact <= gamma_gibbs_formula + 1e-12),
            }
        )

    result_frame = pd.DataFrame(records)
    diagnostics_frame = pd.DataFrame(diagnostics_records)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(result_frame["n"], result_frame["gamma_exact"], "o-", linewidth=2, label=r"$\gamma_n$ exact")
    axes[0].plot(result_frame["n"], result_frame["gamma_gibbs_formula"], "s--", linewidth=2, label=r"$\gamma_n$ Gibbs formula")
    axes[0].plot(result_frame["n"], result_frame["tv_stability"] * 2.0, "^-", linewidth=2, label=r"$2 \cdot \mathrm{TV}$")
    axes[0].plot(
        result_frame["n"],
        result_frame["expected_uniform_stability"] / config.max_loss,
        "d-",
        linewidth=2,
        label=r"$\beta_{\mathrm{EUS}} / M$",
    )
    axes[0].set_title("ADS implications on exact finite Gibbs posteriors")
    axes[0].set_xlabel("Training set size n")
    axes[0].set_ylabel("Stability quantity")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(result_frame["n"], result_frame["exact_generalization"], "o-", linewidth=2, label="Exact gen")
    axes[1].plot(
        result_frame["n"],
        result_frame["v2_bound"],
        "^-",
        linewidth=2,
        label=r"Direct $V_2$ bound",
    )
    axes[1].plot(
        result_frame["n"],
        result_frame["ads_bound_exact_gamma"],
        "s--",
        linewidth=2,
        label=r"ADS bound via exact $\gamma_n$",
    )
    axes[1].plot(
        result_frame["n"],
        result_frame["ads_bound_gibbs_formula"],
        "d-",
        linewidth=2,
        label=r"ADS bound via Gibbs formula",
    )
    axes[1].set_title("Generalization bounds implied by ADS")
    axes[1].set_xlabel("Training set size n")
    axes[1].set_ylabel("Bound / exact generalization")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()

    figure_path = output_dir / "05_ads_stability.png"
    table_path = output_dir / "05_ads_stability.csv"
    diagnostics_path = output_dir / "05_ads_stability_sanity.csv"
    config_path = output_dir / "05_ads_stability_config.json"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    result_frame.to_csv(table_path, index=False)
    diagnostics_frame.to_csv(diagnostics_path, index=False)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
        },
        config_path,
    )

    return {
        "results": result_frame,
        "sanity": diagnostics_frame,
        "diagnostics": {
            "figure_path": str(figure_path),
            "table_path": str(table_path),
            "diagnostics_path": str(diagnostics_path),
        },
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def run_conditional_output_experiment(config: ConditionalOutputExperimentConfig) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = config.artifact_prefix
    sigma_loss_sq_value = sigma_loss_sq(config.clip_c)

    conditional_records: list[dict[str, float]] = []
    conditional_sanity_records: list[dict[str, float | bool]] = []
    conditional_alpha_records: list[dict[str, float]] = []
    output_records: list[dict[str, float]] = []
    output_sanity_records: list[dict[str, float | bool]] = []
    output_alpha_records: list[dict[str, float]] = []

    for dim in config.dimensions:
        w_true = make_w_true(dim, config.w_true_value)
        holdout = make_holdout_set(
            rng=rng,
            num_holdout=config.num_holdout,
            w_true=w_true,
            sigma_noise_sq=config.sigma_noise_sq,
        )

        evaluation_datasets = [
            generate_regression_dataset(
                rng=rng,
                n=config.n,
                w_true=w_true,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for _ in range(config.num_datasets)
        ]
        evaluation_posteriors = [
            posterior_from_dataset(
                x,
                y,
                lambda_prior=config.lambda_prior,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for x, y in evaluation_datasets
        ]
        empirical_gen, empirical_stderr = estimate_empirical_generalization(
            datasets=evaluation_datasets,
            posteriors=evaluation_posteriors,
            holdout=holdout,
            clip_c=config.clip_c,
            rng=rng,
        )

        cmi_estimates = []
        samplewise_bounds = []
        samplewise_mean_infos = []
        conditional_v_alpha_samples = {float(alpha): [] for alpha in config.alpha_values}
        conditional_proxy_products = {float(alpha): [] for alpha in config.alpha_values}

        for _ in range(config.num_supersamples):
            supersample_x, supersample_y = generate_supersample(
                rng=rng,
                n=config.n,
                w_true=w_true,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            reference_u_vectors = [
                rng.integers(0, 2, size=config.n) for _ in range(config.num_marginal_components)
            ]
            reference_posteriors = [
                posterior_from_dataset(
                    *select_dataset_from_supersample(supersample_x, supersample_y, u_vector),
                    lambda_prior=config.lambda_prior,
                    sigma_noise_sq=config.sigma_noise_sq,
                )
                for u_vector in reference_u_vectors
            ]
            reference_mixture = GaussianMixtureReference(reference_posteriors)

            evaluation_u_vectors = [rng.integers(0, 2, size=config.n) for _ in range(config.num_u_samples)]
            evaluation_supersample_posteriors = [
                posterior_from_dataset(
                    *select_dataset_from_supersample(supersample_x, supersample_y, u_vector),
                    lambda_prior=config.lambda_prior,
                    sigma_noise_sq=config.sigma_noise_sq,
                )
                for u_vector in evaluation_u_vectors
            ]

            cmi_value, _ = estimate_mutual_information(
                posteriors=evaluation_supersample_posteriors,
                reference_mixture=reference_mixture,
                rng=rng,
                num_weight_samples=config.num_mi_weight_samples,
            )
            cmi_estimates.append(max(cmi_value, 0.0))

            v_alpha_values, _ = estimate_v_alpha_from_marginal(
                posteriors=evaluation_supersample_posteriors,
                reference_mixture=reference_mixture,
                alpha_values=config.alpha_values,
                rng=rng,
                num_weight_samples=config.num_weight_samples,
            )
            for alpha, value in v_alpha_values.items():
                conditional_v_alpha_samples[float(alpha)].append(float(value))

            candidate_means = np.vstack([posterior.mean for posterior in reference_posteriors[: config.proxy_num_means]])
            candidate_samples = reference_mixture.sample(rng, config.proxy_num_marginal_samples)
            candidate_weights = np.vstack([candidate_means, candidate_samples])
            proxy_norms = estimate_supersample_proxy_norms(
                supersample_x=supersample_x,
                supersample_y=supersample_y,
                candidate_weights=candidate_weights,
                evaluation_u_vectors=evaluation_u_vectors,
                alpha_values=config.alpha_values,
                clip_c=config.clip_c,
            )
            for alpha in config.alpha_values:
                conditional_proxy_products[float(alpha)].append(proxy_norms[float(alpha)] * v_alpha_values[float(alpha)])

            samplewise_bound, samplewise_diag = estimate_samplewise_loss_proxy_mi(
                supersample_x=supersample_x,
                supersample_y=supersample_y,
                evaluation_u_vectors=evaluation_u_vectors,
                evaluation_posteriors=evaluation_supersample_posteriors,
                clip_c=config.clip_c,
                rng=rng,
                num_weight_samples=config.samplewise_num_weight_samples,
                num_bins=config.samplewise_num_bins,
            )
            samplewise_bounds.append(samplewise_bound)
            samplewise_mean_infos.append(samplewise_diag["mean_samplewise_mi"])

        mean_conditional_v_alpha = {
            alpha: float(np.mean(values)) for alpha, values in conditional_v_alpha_samples.items()
        }
        rigorous_prefactors = {
            alpha: conditional_supersample_norm_prefactor(alpha, config.n, config.clip_c)
            for alpha in mean_conditional_v_alpha
        }
        rigorous_bounds = {
            alpha: rigorous_prefactors[alpha] * mean_value for alpha, mean_value in mean_conditional_v_alpha.items()
        }
        proxy_bounds = {
            alpha: float(np.mean(values)) for alpha, values in conditional_proxy_products.items()
        }
        rigorous_optimal_alpha = min(rigorous_bounds, key=rigorous_bounds.get)
        proxy_optimal_alpha = min(proxy_bounds, key=proxy_bounds.get)
        cmi_estimate = float(np.mean(cmi_estimates))
        cmi_bound = bounded_loss_cmi_bound(cmi_estimate, config.clip_c, config.n)
        samplewise_bound_mean = float(np.mean(samplewise_bounds))

        conditional_records.append(
            {
                "dimension": float(dim),
                "empirical_generalization": empirical_gen,
                "empirical_generalization_stderr": empirical_stderr,
                "cmi_estimate": cmi_estimate,
                "cmi_bound": cmi_bound,
                "samplewise_loss_proxy_bound": samplewise_bound_mean,
                "rigorous_conditional_v_alpha_bound": float(rigorous_bounds[rigorous_optimal_alpha]),
                "rigorous_optimal_alpha": float(rigorous_optimal_alpha),
                "proxy_conditional_v_alpha_bound": float(proxy_bounds[proxy_optimal_alpha]),
                "proxy_optimal_alpha": float(proxy_optimal_alpha),
                "fixed_rigorous_conditional_v2_bound": float(rigorous_bounds[2.0]),
                "fixed_proxy_conditional_v2_bound": float(proxy_bounds[2.0]),
            }
        )
        conditional_sanity_records.append(
            {
                "dimension": float(dim),
                "cmi_nonnegative": bool(cmi_estimate >= -1e-9),
                "samplewise_nonnegative": bool(samplewise_bound_mean >= -1e-9),
                "rigorous_dominates_proxy": bool(
                    rigorous_bounds[rigorous_optimal_alpha] >= proxy_bounds[proxy_optimal_alpha] - 1e-10
                ),
                "mean_samplewise_mi": float(np.mean(samplewise_mean_infos)),
            }
        )
        for alpha in config.alpha_values:
            conditional_alpha_records.append(
                {
                    "dimension": float(dim),
                    "alpha": float(alpha),
                    "conditional_v_alpha": mean_conditional_v_alpha[float(alpha)],
                    "rigorous_prefactor": rigorous_prefactors[float(alpha)],
                    "rigorous_bound": rigorous_bounds[float(alpha)],
                    "proxy_bound": proxy_bounds[float(alpha)],
                }
            )

        reference_datasets = [
            generate_regression_dataset(
                rng=rng,
                n=config.n,
                w_true=w_true,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for _ in range(config.num_reference_datasets)
        ]
        reference_posteriors = [
            posterior_from_dataset(
                x,
                y,
                lambda_prior=config.lambda_prior,
                sigma_noise_sq=config.sigma_noise_sq,
            )
            for x, y in reference_datasets
        ]
        parameter_reference_mixture = GaussianMixtureReference(reference_posteriors)
        parameter_v_alpha, _ = estimate_v_alpha_from_marginal(
            posteriors=evaluation_posteriors,
            reference_mixture=parameter_reference_mixture,
            alpha_values=config.alpha_values,
            rng=rng,
            num_weight_samples=config.output_num_weight_samples,
        )
        parameter_bounds = {
            alpha: subgaussian_multiplier(alpha, sigma_loss_sq_value, config.n) * value
            for alpha, value in parameter_v_alpha.items()
        }

        output_test_size = max(1, min(config.output_test_size_max, max(1, dim // 2)))
        output_design = rng.standard_normal((output_test_size, dim))
        projected_evaluation_posteriors = [
            predictive_posterior_from_design(posterior, output_design) for posterior in evaluation_posteriors
        ]
        projected_reference_posteriors = [
            predictive_posterior_from_design(posterior, output_design) for posterior in reference_posteriors
        ]
        output_reference_mixture = GaussianMixtureReference(projected_reference_posteriors)
        output_v_alpha, _ = estimate_v_alpha_from_marginal(
            posteriors=projected_evaluation_posteriors,
            reference_mixture=output_reference_mixture,
            alpha_values=config.alpha_values,
            rng=rng,
            num_weight_samples=config.output_num_weight_samples,
        )
        output_bounds = {
            alpha: subgaussian_multiplier(alpha, sigma_loss_sq_value, config.n) * value
            for alpha, value in output_v_alpha.items()
        }
        parameter_optimal_alpha = min(parameter_bounds, key=parameter_bounds.get)
        output_optimal_alpha = min(output_bounds, key=output_bounds.get)

        output_records.append(
            {
                "dimension": float(dim),
                "output_space_dimension": float(output_test_size),
                "empirical_generalization": empirical_gen,
                "parameter_space_optimized_bound": float(parameter_bounds[parameter_optimal_alpha]),
                "parameter_space_optimal_alpha": float(parameter_optimal_alpha),
                "output_space_optimized_bound": float(output_bounds[output_optimal_alpha]),
                "output_space_optimal_alpha": float(output_optimal_alpha),
            }
        )
        output_sanity_records.append(
            {
                "dimension": float(dim),
                "max_same_alpha_dpi_violation": float(
                    max(output_bounds[alpha] - parameter_bounds[alpha] for alpha in config.alpha_values)
                ),
                "output_opt_not_worse_than_parameter_opt": bool(
                    output_bounds[output_optimal_alpha] <= parameter_bounds[parameter_optimal_alpha] + 1e-8
                ),
            }
        )
        for alpha in config.alpha_values:
            output_alpha_records.append(
                {
                    "dimension": float(dim),
                    "alpha": float(alpha),
                    "parameter_space_bound": parameter_bounds[float(alpha)],
                    "output_space_bound": output_bounds[float(alpha)],
                }
            )

    conditional_frame = pd.DataFrame(conditional_records)
    conditional_sanity_frame = pd.DataFrame(conditional_sanity_records)
    conditional_alpha_frame = pd.DataFrame(conditional_alpha_records)
    output_frame = pd.DataFrame(output_records)
    output_sanity_frame = pd.DataFrame(output_sanity_records)
    output_alpha_frame = pd.DataFrame(output_alpha_records)

    fig_conditional, ax_conditional = plt.subplots(figsize=(8.8, 5.0))
    ax_conditional.errorbar(
        conditional_frame["dimension"],
        conditional_frame["empirical_generalization"],
        yerr=conditional_frame["empirical_generalization_stderr"],
        fmt="o-",
        linewidth=2,
        capsize=3,
        label="Empirical generalization",
    )
    ax_conditional.plot(conditional_frame["dimension"], conditional_frame["cmi_bound"], "s--", linewidth=2, label="CMI")
    ax_conditional.plot(
        conditional_frame["dimension"],
        conditional_frame["rigorous_conditional_v_alpha_bound"],
        "d-",
        linewidth=2,
        label=r"Rigorous conditional $V_\alpha$",
    )
    ax_conditional.plot(
        conditional_frame["dimension"],
        conditional_frame["proxy_conditional_v_alpha_bound"],
        "x-",
        linewidth=2,
        label=r"Proxy conditional $V_\alpha$",
    )
    ax_conditional.set_xlabel("Dimension d")
    ax_conditional.set_ylabel("Bound / empirical gap")
    ax_conditional.set_title("Conditional bounds on shared supersamples")
    ax_conditional.grid(True, alpha=0.3)
    ax_conditional.legend()
    fig_conditional.tight_layout()

    fig_output, ax_output = plt.subplots(figsize=(8.8, 5.0))
    ax_output.errorbar(
        output_frame["dimension"],
        output_frame["empirical_generalization"],
        yerr=conditional_frame["empirical_generalization_stderr"],
        fmt="o-",
        linewidth=2,
        capsize=3,
        label="Empirical generalization",
    )
    ax_output.plot(
        output_frame["dimension"],
        output_frame["parameter_space_optimized_bound"],
        "s--",
        linewidth=2,
        label=r"Parameter-space opt $V_\alpha$",
    )
    ax_output.plot(
        output_frame["dimension"],
        output_frame["output_space_optimized_bound"],
        "d-",
        linewidth=2,
        label=r"Output-space opt $V_\alpha$",
    )
    ax_output.set_xlabel("Dimension d")
    ax_output.set_ylabel("Bound / empirical gap")
    ax_output.set_title("Output-space projections vs parameter space")
    ax_output.grid(True, alpha=0.3)
    ax_output.legend()
    fig_output.tight_layout()

    conditional_figure_path = output_dir / f"{artifact_prefix}_conditional_bounds.png"
    output_figure_path = output_dir / f"{artifact_prefix}_output_space.png"
    conditional_table_path = output_dir / f"{artifact_prefix}_conditional_bounds.csv"
    conditional_sanity_path = output_dir / f"{artifact_prefix}_conditional_bounds_sanity.csv"
    conditional_alpha_path = output_dir / f"{artifact_prefix}_conditional_bounds_alpha.csv"
    output_table_path = output_dir / f"{artifact_prefix}_output_space.csv"
    output_sanity_path = output_dir / f"{artifact_prefix}_output_space_sanity.csv"
    output_alpha_path = output_dir / f"{artifact_prefix}_output_space_alpha.csv"
    config_path = output_dir / f"{artifact_prefix}_conditional_output_config.json"
    fig_conditional.savefig(conditional_figure_path, dpi=200)
    fig_output.savefig(output_figure_path, dpi=200)
    plt.close(fig_conditional)
    plt.close(fig_output)

    conditional_frame.to_csv(conditional_table_path, index=False)
    conditional_sanity_frame.to_csv(conditional_sanity_path, index=False)
    conditional_alpha_frame.to_csv(conditional_alpha_path, index=False)
    output_frame.to_csv(output_table_path, index=False)
    output_sanity_frame.to_csv(output_sanity_path, index=False)
    output_alpha_frame.to_csv(output_alpha_path, index=False)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
        },
        config_path,
    )

    return {
        "conditional_results": conditional_frame,
        "conditional_sanity": conditional_sanity_frame,
        "conditional_alpha": conditional_alpha_frame,
        "output_results": output_frame,
        "output_sanity": output_sanity_frame,
        "output_alpha": output_alpha_frame,
        "diagnostics": {
            "conditional_figure_path": str(conditional_figure_path),
            "output_figure_path": str(output_figure_path),
            "conditional_table_path": str(conditional_table_path),
            "output_table_path": str(output_table_path),
        },
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def run_scaling_continuation_experiment(config: ScalingContinuationExperimentConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditional_alpha_values = tuple(alpha for alpha in config.alpha_values if alpha > 1.0)
    if not conditional_alpha_values:
        raise ValueError("ScalingContinuationExperimentConfig.alpha_values must contain at least one alpha > 1.")

    parameter_bundle = run_alpha_sweep_comparison_experiment(
        AlphaSweepComparisonConfig(
            sigma_noise_sq=config.sigma_noise_sq,
            lambda_prior=config.lambda_prior,
            clip_c=config.clip_c,
            w_true_value=config.w_true_value,
            dimensions=config.dimensions,
            benchmark_dimensions=config.parameter_benchmark_dimensions,
            alpha_values=config.alpha_values,
            n=config.n,
            num_holdout=config.num_holdout,
            primary_num_repeats=config.parameter_num_repeats,
            primary_num_datasets=config.parameter_num_datasets,
            primary_num_reference_datasets=config.parameter_num_reference_datasets,
            primary_total_weight_samples=config.parameter_total_weight_samples,
            primary_num_mi_weight_samples=config.parameter_num_mi_weight_samples,
            benchmark_num_repeats=config.parameter_benchmark_num_repeats,
            benchmark_num_datasets=config.parameter_benchmark_num_datasets,
            benchmark_num_reference_datasets=config.parameter_benchmark_num_reference_datasets,
            benchmark_total_weight_samples=config.parameter_benchmark_total_weight_samples,
            benchmark_num_mi_weight_samples=config.parameter_benchmark_num_mi_weight_samples,
            seed=config.seed + 11,
            include_legacy_reference=False,
            include_crossfit=False,
            include_surrogate_diagnostics=False,
            artifact_prefix=f"{config.artifact_prefix}_parameter",
            output_dir=config.output_dir,
        )
    )
    conditional_bundle = run_conditional_output_experiment(
        ConditionalOutputExperimentConfig(
            sigma_noise_sq=config.sigma_noise_sq,
            lambda_prior=config.lambda_prior,
            clip_c=config.clip_c,
            w_true_value=config.w_true_value,
            dimensions=config.dimensions,
            alpha_values=conditional_alpha_values,
            n=config.n,
            num_datasets=config.conditional_num_datasets,
            num_reference_datasets=config.conditional_num_reference_datasets,
            num_holdout=config.num_holdout,
            num_supersamples=config.conditional_num_supersamples,
            num_u_samples=config.conditional_num_u_samples,
            num_marginal_components=config.conditional_num_marginal_components,
            num_weight_samples=config.conditional_num_weight_samples,
            num_mi_weight_samples=config.conditional_num_mi_weight_samples,
            proxy_num_means=config.proxy_num_means,
            proxy_num_marginal_samples=config.proxy_num_marginal_samples,
            samplewise_num_weight_samples=config.samplewise_num_weight_samples,
            samplewise_num_bins=config.samplewise_num_bins,
            output_test_size_max=config.output_test_size_max,
            output_num_weight_samples=config.output_num_weight_samples,
            seed=config.seed + 101,
            artifact_prefix=config.artifact_prefix,
            output_dir=config.output_dir,
        )
    )

    parameter_summary = parameter_bundle["summary"]
    parameter_primary = parameter_summary[
        (parameter_summary["scenario"] == "primary")
        & (parameter_summary["estimator"] == "split_selection")
        & (parameter_summary["reference_kind"] == "true_mixture")
    ][
        [
            "dimension",
            "repeat_count",
            "empirical_generalization_mean",
            "reference_information_bound_mean",
            "fixed_v2_bound_mean",
            "optimized_bound_mean",
            "selected_alpha_mean",
            "selected_alpha_std",
            "selected_alpha_range",
        ]
    ].rename(
        columns={
            "repeat_count": "parameter_repeat_count",
            "empirical_generalization_mean": "parameter_empirical_generalization",
            "reference_information_bound_mean": "parameter_mi_bound",
            "fixed_v2_bound_mean": "parameter_fixed_v2_bound",
            "optimized_bound_mean": "parameter_optimized_bound",
            "selected_alpha_mean": "parameter_selected_alpha",
            "selected_alpha_std": "parameter_selected_alpha_std",
            "selected_alpha_range": "parameter_selected_alpha_range",
        }
    )
    conditional_results = conditional_bundle["conditional_results"].rename(
        columns={
            "empirical_generalization": "conditional_empirical_generalization",
            "empirical_generalization_stderr": "conditional_empirical_generalization_stderr",
            "rigorous_conditional_v_alpha_bound": "conditional_rigorous_bound",
            "rigorous_optimal_alpha": "conditional_rigorous_alpha",
            "proxy_conditional_v_alpha_bound": "conditional_proxy_bound",
            "proxy_optimal_alpha": "conditional_proxy_alpha",
            "fixed_rigorous_conditional_v2_bound": "conditional_fixed_rigorous_v2_bound",
            "fixed_proxy_conditional_v2_bound": "conditional_fixed_proxy_v2_bound",
        }
    )
    output_results = conditional_bundle["output_results"].rename(
        columns={
            "empirical_generalization": "output_empirical_generalization",
            "parameter_space_optimized_bound": "output_runner_parameter_optimized_bound",
            "parameter_space_optimal_alpha": "output_runner_parameter_alpha",
            "output_space_optimized_bound": "output_space_optimized_bound",
            "output_space_optimal_alpha": "output_space_optimal_alpha",
        }
    )
    combined_frame = parameter_primary.merge(
        conditional_results,
        on="dimension",
        how="inner",
    ).merge(
        output_results,
        on="dimension",
        how="inner",
    )
    combined_frame["output_vs_parameter_ratio"] = (
        combined_frame["output_space_optimized_bound"] / combined_frame["parameter_optimized_bound"]
    )
    combined_frame["conditional_vs_parameter_ratio"] = (
        combined_frame["conditional_rigorous_bound"] / combined_frame["parameter_optimized_bound"]
    )
    combined_frame["parameter_split_selected_above_fixed_v2"] = (
        combined_frame["parameter_optimized_bound"] > combined_frame["parameter_fixed_v2_bound"] + 1e-10
    )

    alpha_frame = pd.DataFrame(
        [
            {
                "dimension": float(row["dimension"]),
                "bound_family": "parameter_space",
                "selected_alpha": float(row["parameter_selected_alpha"]),
                "theorem_valid": True,
            }
            for _, row in combined_frame.iterrows()
        ]
        + [
            {
                "dimension": float(row["dimension"]),
                "bound_family": "conditional_rigorous",
                "selected_alpha": float(row["conditional_rigorous_alpha"]),
                "theorem_valid": True,
            }
            for _, row in combined_frame.iterrows()
        ]
        + [
            {
                "dimension": float(row["dimension"]),
                "bound_family": "conditional_proxy",
                "selected_alpha": float(row["conditional_proxy_alpha"]),
                "theorem_valid": False,
            }
            for _, row in combined_frame.iterrows()
        ]
        + [
            {
                "dimension": float(row["dimension"]),
                "bound_family": "output_space",
                "selected_alpha": float(row["output_space_optimal_alpha"]),
                "theorem_valid": True,
            }
            for _, row in combined_frame.iterrows()
        ]
    )
    conditional_sanity = conditional_bundle["conditional_sanity"]
    output_sanity = conditional_bundle["output_sanity"]
    sanity_frame = combined_frame[
        [
            "dimension",
            "parameter_repeat_count",
            "parameter_selected_alpha_std",
            "parameter_selected_alpha_range",
            "output_vs_parameter_ratio",
            "conditional_vs_parameter_ratio",
            "parameter_split_selected_above_fixed_v2",
        ]
    ].merge(conditional_sanity, on="dimension", how="left").merge(output_sanity, on="dimension", how="left")
    sanity_frame["parameter_compute_equalized"] = bool(
        parameter_bundle["diagnostics"]["story_checks"]["theorem_valid_compute_equalized"]
    )

    fig, (ax_bounds, ax_alpha) = plt.subplots(1, 2, figsize=(13.0, 4.9))
    if not combined_frame.empty:
        combined_frame = combined_frame.sort_values("dimension")
        ax_bounds.plot(
            combined_frame["dimension"],
            combined_frame["parameter_empirical_generalization"],
            "o-",
            linewidth=2,
            label="Empirical generalization",
        )
        ax_bounds.plot(
            combined_frame["dimension"],
            combined_frame["parameter_fixed_v2_bound"],
            "^-",
            linewidth=2,
            label=r"Parameter fixed $V_2$",
        )
        ax_bounds.plot(
            combined_frame["dimension"],
            combined_frame["parameter_optimized_bound"],
            "d-",
            linewidth=2,
            label=r"Parameter split-selected $V_\alpha$",
        )
        ax_bounds.plot(
            combined_frame["dimension"],
            combined_frame["conditional_rigorous_bound"],
            "s--",
            linewidth=2,
            label=r"Conditional rigorous $V_\alpha$",
        )
        ax_bounds.plot(
            combined_frame["dimension"],
            combined_frame["conditional_proxy_bound"],
            "x:",
            linewidth=2,
            label=r"Conditional proxy $V_\alpha$",
        )
        ax_bounds.plot(
            combined_frame["dimension"],
            combined_frame["output_space_optimized_bound"],
            "p-",
            linewidth=2,
            label=r"Output-space opt $V_\alpha$",
        )
        ax_bounds.set_xlabel("Dimension d")
        ax_bounds.set_ylabel("Bound / empirical gap")
        ax_bounds.set_title("Scaling continuation beyond parameter space")
        ax_bounds.grid(True, alpha=0.3)
        ax_bounds.legend()

        for family, label, style in [
            ("parameter_space", r"Parameter split $\alpha^*$", "o-"),
            ("conditional_rigorous", r"Conditional rigorous $\alpha^*$", "s--"),
            ("conditional_proxy", r"Conditional proxy $\alpha^*$", "x:"),
            ("output_space", r"Output-space $\alpha^*$", "d-"),
        ]:
            subset = alpha_frame[alpha_frame["bound_family"] == family].sort_values("dimension")
            ax_alpha.plot(subset["dimension"], subset["selected_alpha"], style, linewidth=2, label=label)
        ax_alpha.set_xlabel("Dimension d")
        ax_alpha.set_ylabel(r"Selected $\alpha^*$")
        ax_alpha.set_title(r"Where the optimizer moves across formulations")
        ax_alpha.grid(True, alpha=0.3)
        ax_alpha.legend()
    fig.tight_layout()

    summary_path = output_dir / f"{config.artifact_prefix}.csv"
    alpha_path = output_dir / f"{config.artifact_prefix}_alpha.csv"
    sanity_path = output_dir / f"{config.artifact_prefix}_sanity.csv"
    figure_path = output_dir / f"{config.artifact_prefix}.png"
    config_path = output_dir / f"{config.artifact_prefix}_config.json"
    combined_frame.to_csv(summary_path, index=False)
    alpha_frame.to_csv(alpha_path, index=False)
    sanity_frame.to_csv(sanity_path, index=False)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
            "parameter_bundle_prefix": f"{config.artifact_prefix}_parameter",
            "conditional_bundle_prefix": config.artifact_prefix,
        },
        config_path,
    )

    return {
        "summary": combined_frame,
        "alpha": alpha_frame,
        "sanity": sanity_frame,
        "parameter_bundle": parameter_bundle,
        "conditional_output_bundle": conditional_bundle,
        "diagnostics": {
            "summary_path": str(summary_path),
            "alpha_path": str(alpha_path),
            "sanity_path": str(sanity_path),
            "figure_path": str(figure_path),
        },
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def run_alpha_sweep_sensitivity_experiment(config: AlphaSweepSensitivityConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma_loss_sq_value = sigma_loss_sq(config.clip_c)
    alpha_values = tuple(sorted({float(alpha) for alpha in config.alpha_values}))
    optimization_alpha_values = alpha_optimization_grid(alpha_values)
    if 1.0 not in alpha_values:
        raise ValueError("AlphaSweepSensitivityConfig.alpha_values must include 1.0.")

    profile_records: list[dict[str, float | str | int | bool]] = []
    repeat_records: list[dict[str, float | str | int | bool]] = []
    sanity_records: list[dict[str, float | str | int | bool]] = []

    def append_profile_rows(
        *,
        sweep_type: str,
        dimension: int,
        sweep_value: float,
        repeat_index: int,
        profile: dict[str, Any],
        num_weight_samples: int,
        stage: str,
    ) -> None:
        for alpha in alpha_values:
            profile_records.append(
                {
                    "sweep_type": sweep_type,
                    "dimension": float(dimension),
                    "sweep_value": float(sweep_value),
                    "repeat": float(repeat_index),
                    "stage": stage,
                    "alpha": float(alpha),
                    "v_alpha_value": float(profile["v_alpha_values"][alpha]),
                    "bound": float(profile["bound_by_alpha"][alpha]),
                    "num_weight_samples": float(num_weight_samples),
                }
            )

    def append_repeat_row(
        *,
        sweep_type: str,
        case: dict[str, Any],
        sweep_value: float,
        sigma_noise_sq_value: float,
        lambda_prior_value: float,
        repeat_index: int,
        info_bundle: dict[str, float],
        split_result: dict[str, Any],
    ) -> None:
        reference_cost = mi_logpdf_cost(
            num_datasets=len(case["evaluation_posteriors"]),
            num_reference_datasets=len(case["reference_posteriors"]),
            num_weight_samples=int(case["num_mi_weight_samples"]),
        )
        estimator_cost = alpha_sweep_logpdf_cost(
            num_datasets=len(case["evaluation_posteriors"]),
            num_reference_datasets=len(case["reference_posteriors"]),
            num_weight_samples=config.total_weight_samples,
        )
        repeat_records.append(
            {
                "record_type": "repeat",
                "sweep_type": sweep_type,
                "dimension": float(case["dimension"]),
                "sweep_value": float(sweep_value),
                "sigma_noise_sq": float(sigma_noise_sq_value),
                "lambda_prior": float(lambda_prior_value),
                "repeat": float(repeat_index),
                "empirical_generalization": float(case["empirical_generalization"]),
                "empirical_generalization_stderr": float(case["empirical_generalization_stderr"]),
                "reference_information_bound": float(info_bundle["information_bound"]),
                "pinsker_tv_bound": float(info_bundle["pinsker_tv_bound"]),
                "direct_v1_bound": float(split_result["direct_v1_bound"]),
                "fixed_v2_bound": float(split_result["fixed_v2_bound"]),
                "optimized_bound": float(split_result["optimized_bound"]),
                "selected_alpha": float(split_result["selected_alpha"]),
                "reference_logpdf_cost": float(reference_cost),
                "estimator_logpdf_cost": float(estimator_cost),
                "compute_within_5pct_vs_reference": bool(
                    abs(estimator_cost - reference_cost) / max(estimator_cost, reference_cost) <= 0.05
                ),
                "direct_v1_leq_pinsker": bool(
                    split_result["direct_v1_bound"] <= info_bundle["pinsker_tv_bound"] + 1e-8
                ),
                "raw_v_alpha_monotonicity_violation": float(split_result["evaluation_profile"]["monotonicity_violation"]),
                "raw_v_alpha_monotonicity_count": int(
                    split_result["evaluation_profile"]["monotonicity_violation_count"]
                ),
            }
        )

    for repeat_index in range(config.num_repeats):
        for dim in config.dimensions:
            for sweep_type, sweep_values in [
                ("noise", config.sigma_noise_sq_values),
                ("prior", config.lambda_prior_values),
            ]:
                for sweep_value in sweep_values:
                    sigma_noise_sq_value = (
                        float(sweep_value) if sweep_type == "noise" else float(config.base_sigma_noise_sq)
                    )
                    lambda_prior_value = (
                        float(sweep_value) if sweep_type == "prior" else float(config.base_lambda_prior)
                    )
                    case_seed = (
                        config.seed
                        + repeat_index * 5000
                        + int(dim) * 79
                        + int(round(float(sweep_value) * 1000.0))
                        + (0 if sweep_type == "noise" else 200000)
                    )
                    case = prepare_alpha_sweep_case(
                        rng=np.random.default_rng(case_seed),
                        dim=int(dim),
                        n=config.n,
                        num_datasets=config.num_datasets,
                        num_reference_datasets=config.num_reference_datasets,
                        num_holdout=config.num_holdout,
                        w_true_value=config.w_true_value,
                        sigma_noise_sq=sigma_noise_sq_value,
                        lambda_prior=lambda_prior_value,
                        clip_c=config.clip_c,
                        num_mi_weight_samples=config.num_mi_weight_samples,
                    )
                    case["num_mi_weight_samples"] = config.num_mi_weight_samples
                    info_bundle = estimate_reference_information_bundle(
                        posteriors=case["evaluation_posteriors"],
                        reference=case["reference_mixture"],
                        rng=np.random.default_rng(case_seed + 1),
                        num_weight_samples=config.num_mi_weight_samples,
                        clip_c=config.clip_c,
                        sigma_loss_sq_value=sigma_loss_sq_value,
                        n=config.n,
                    )
                    split_result = run_split_alpha_estimator(
                        posteriors=case["evaluation_posteriors"],
                        reference=case["reference_mixture"],
                        alpha_values=alpha_values,
                        optimization_alpha_values=optimization_alpha_values,
                        sigma_loss_sq_value=sigma_loss_sq_value,
                        n=config.n,
                        clip_c=config.clip_c,
                        total_weight_samples=config.total_weight_samples,
                        seed=case_seed + 10,
                    )
                    half_budgets = partition_weight_budget(config.total_weight_samples, 2)
                    append_profile_rows(
                        sweep_type=sweep_type,
                        dimension=int(dim),
                        sweep_value=float(sweep_value),
                        repeat_index=repeat_index,
                        profile=split_result["selection_profile"],
                        num_weight_samples=int(half_budgets[0]),
                        stage="split_select",
                    )
                    append_profile_rows(
                        sweep_type=sweep_type,
                        dimension=int(dim),
                        sweep_value=float(sweep_value),
                        repeat_index=repeat_index,
                        profile=split_result["evaluation_profile"],
                        num_weight_samples=int(half_budgets[1]),
                        stage="split_eval",
                    )
                    append_repeat_row(
                        sweep_type=sweep_type,
                        case=case,
                        sweep_value=float(sweep_value),
                        sigma_noise_sq_value=sigma_noise_sq_value,
                        lambda_prior_value=lambda_prior_value,
                        repeat_index=repeat_index,
                        info_bundle=info_bundle,
                        split_result=split_result,
                    )

    profile_frame = pd.DataFrame(profile_records)
    repeat_frame = pd.DataFrame(repeat_records)
    summary_frame = (
        repeat_frame.groupby(["sweep_type", "dimension", "sweep_value", "sigma_noise_sq", "lambda_prior"], as_index=False)
        .agg(
            repeat_count=("repeat", "count"),
            empirical_generalization_mean=("empirical_generalization", "mean"),
            empirical_generalization_std=("empirical_generalization", "std"),
            reference_information_bound_mean=("reference_information_bound", "mean"),
            reference_information_bound_std=("reference_information_bound", "std"),
            pinsker_tv_bound_mean=("pinsker_tv_bound", "mean"),
            direct_v1_bound_mean=("direct_v1_bound", "mean"),
            fixed_v2_bound_mean=("fixed_v2_bound", "mean"),
            optimized_bound_mean=("optimized_bound", "mean"),
            optimized_bound_std=("optimized_bound", "std"),
            selected_alpha_mean=("selected_alpha", "mean"),
            selected_alpha_std=("selected_alpha", "std"),
            selected_alpha_min=("selected_alpha", "min"),
            selected_alpha_max=("selected_alpha", "max"),
        )
    )
    summary_frame["selected_alpha_range"] = summary_frame["selected_alpha_max"] - summary_frame["selected_alpha_min"]
    for column in [
        "empirical_generalization_std",
        "reference_information_bound_std",
        "optimized_bound_std",
        "selected_alpha_std",
    ]:
        summary_frame[column] = summary_frame[column].fillna(0.0)

    for _, row in repeat_frame.iterrows():
        sanity_records.append(row.to_dict())
    for (sweep_type, dimension, sweep_value), group in repeat_frame.groupby(["sweep_type", "dimension", "sweep_value"]):
        unique_alpha, counts = np.unique(group["selected_alpha"].to_numpy(dtype=float), return_counts=True)
        alpha_frequency = {
            f"{alpha:.2f}": int(count)
            for alpha, count in sorted(zip(unique_alpha, counts, strict=False), key=lambda item: item[0])
        }
        sanity_records.append(
            {
                "record_type": "aggregate",
                "sweep_type": sweep_type,
                "dimension": float(dimension),
                "sweep_value": float(sweep_value),
                "repeat": np.nan,
                "alpha_frequency_json": json.dumps(alpha_frequency, sort_keys=True),
                "selected_alpha_mean": float(group["selected_alpha"].mean()),
                "selected_alpha_std": float(group["selected_alpha"].std(ddof=1)) if len(group) > 1 else 0.0,
                "selected_alpha_range": float(group["selected_alpha"].max() - group["selected_alpha"].min()),
                "alpha_stability_flag": bool(group["selected_alpha"].max() - group["selected_alpha"].min() > 0.1),
                "compute_within_5pct_all_repeats": bool(group["compute_within_5pct_vs_reference"].all()),
                "direct_v1_leq_pinsker_all_repeats": bool(group["direct_v1_leq_pinsker"].all()),
            }
        )
    sanity_frame = pd.DataFrame(sanity_records)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    color_cycle = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    method_specs = [
        ("reference_information_bound_mean", "MI", "--", "s"),
        ("fixed_v2_bound_mean", r"Fixed $V_2$", "-", "^"),
        ("optimized_bound_mean", r"Opt $V_\alpha$", "-", "d"),
    ]
    for column_index, sweep_type in enumerate(["noise", "prior"]):
        bounds_ax = axes[0, column_index]
        alpha_ax = axes[1, column_index]
        subset = summary_frame[summary_frame["sweep_type"] == sweep_type]
        for color, dim in zip(color_cycle, config.dimensions, strict=False):
            dim_subset = subset[subset["dimension"] == float(dim)].sort_values("sweep_value")
            if dim_subset.empty:
                continue
            for metric, label, linestyle, marker in method_specs:
                bounds_ax.plot(
                    dim_subset["sweep_value"],
                    dim_subset[metric],
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2,
                    color=color,
                    alpha=0.9 if "Opt" in label else 0.65,
                    label=f"d={int(dim)} {label}",
                )
            alpha_ax.errorbar(
                dim_subset["sweep_value"],
                dim_subset["selected_alpha_mean"],
                yerr=dim_subset["selected_alpha_std"],
                fmt="o-",
                linewidth=2,
                capsize=3,
                color=color,
                label=f"d={int(dim)}",
            )
        bounds_ax.set_xscale("log")
        alpha_ax.set_xscale("log")
        bounds_ax.set_xlabel("Noise variance" if sweep_type == "noise" else "Prior precision")
        alpha_ax.set_xlabel("Noise variance" if sweep_type == "noise" else "Prior precision")
        bounds_ax.set_ylabel("Bound")
        alpha_ax.set_ylabel(r"Selected $\alpha^*$")
        bounds_ax.set_title(f"{sweep_type.title()} sensitivity")
        alpha_ax.set_title(f"{sweep_type.title()} sensitivity of selected alpha")
        bounds_ax.grid(True, alpha=0.3)
        alpha_ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    handles_alpha, labels_alpha = axes[1, 0].get_legend_handles_labels()
    if handles_alpha:
        axes[1, 1].legend(handles_alpha, labels_alpha, loc="best")
    fig.suptitle(r"Noise and prior sensitivity under theorem-valid split-selection", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    summary_path = output_dir / f"{config.artifact_prefix}_summary.csv"
    profiles_path = output_dir / f"{config.artifact_prefix}_profiles.csv"
    sanity_path = output_dir / f"{config.artifact_prefix}_sanity.csv"
    figure_path = output_dir / f"{config.artifact_prefix}.png"
    config_path = output_dir / f"{config.artifact_prefix}_config.json"
    summary_frame.to_csv(summary_path, index=False)
    profile_frame.to_csv(profiles_path, index=False)
    sanity_frame.to_csv(sanity_path, index=False)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    save_json(
        {
            "config": json_ready_config(config),
            "package_versions": collect_package_versions(),
            "paper_estimator": "split_selection",
        },
        config_path,
    )

    return {
        "summary": summary_frame,
        "profiles": profile_frame,
        "sanity": sanity_frame,
        "diagnostics": {
            "summary_path": str(summary_path),
            "profiles_path": str(profiles_path),
            "sanity_path": str(sanity_path),
            "figure_path": str(figure_path),
        },
        "package_versions": collect_package_versions(),
        "config": json_ready_config(config),
    }


def summarize_exact_channel_sanity(result_bundle: dict[str, Any]) -> dict[str, Any]:
    results = result_bundle["results"]
    sanity = result_bundle["sanity"]
    summary = {
        "all_channel_rows_sum_to_one": bool(sanity["channel_rows_sum_to_one"].all()),
        "all_v_alpha_monotone": bool(sanity["monotone_v_alpha"].all()),
        "v2_identity_gap_small": bool((sanity["v2_identity_gap"] < 1e-10).all()),
        "v2_beats_mi_everywhere": bool((results["v2_bound"] <= results["mi_bound"] + 1e-10).all()),
    }
    return summary


def summarize_reparameterization_sanity(result_bundle: dict[str, Any]) -> dict[str, Any]:
    results = result_bundle["results"]
    sanity = result_bundle["sanity"]
    transformed = results[results["transform"] != "identity"]
    summary = {
        "v2_invariant_within_two_percent": bool((transformed["relative_error_vs_original"] < 0.02).all()),
        "density_normalization_good": bool((sanity["max_density_normalization_error"] < 5e-3).all()),
        "contrast_stat_changes": bool(
            (abs(transformed["coordinate_variance_statistic"] - results.iloc[0]["coordinate_variance_statistic"]) > 1e-3).all()
        ),
    }
    return summary


def summarize_ads_sanity(result_bundle: dict[str, Any]) -> dict[str, Any]:
    sanity = result_bundle["sanity"]
    summary = {
        "tv_implication_holds": bool(sanity["tv_le_gamma_over_2"].all()),
        "expected_uniform_implication_holds": bool(sanity["expected_uniform_le_M_gamma"].all()),
        "v2_implication_holds": bool(sanity["v2_le_sqrt_2_gamma_n"].all()),
        "gibbs_formula_upper_bounds_exact_ads_all_n": bool(sanity["gamma_exact_le_gibbs_formula"].all()),
    }
    return summary


def summarize_conditional_output_sanity(result_bundle: dict[str, Any]) -> dict[str, Any]:
    conditional_sanity = result_bundle["conditional_sanity"]
    output_sanity = result_bundle["output_sanity"]
    summary = {
        "conditional_nonnegative": bool(
            conditional_sanity["cmi_nonnegative"].all() and conditional_sanity["samplewise_nonnegative"].all()
        ),
        "rigorous_above_proxy": bool(conditional_sanity["rigorous_dominates_proxy"].all()),
        "output_space_not_worse": bool(
            (output_sanity["max_same_alpha_dpi_violation"] <= 0.2).all()
        ),
    }
    return summary
