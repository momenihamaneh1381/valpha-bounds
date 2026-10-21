import numpy as np

# Import added
import scipy.optimize as optimize
import scipy.special as special
import scipy.integrate as integrate


def bkl(p, q):
    return p * np.log(p / q) + (1-p) * np.log((1-p) / (1-q))


def bjs(p, q):
    return (bkl(p, (p+q) / 2) + bkl(q, (p+q) / 2)) / 2


def optimize_fast(X, Y):
    def func(c2):
        c1 = -np.log(2 - np.exp(c2)) / c2 - 1
        return (c1 + 1) * Y + 2 * X / c2

    def grad(c2):
        dc1 = np.exp(c2) / (2 - np.exp(c2)) / c2 + np.log(2 - np.exp(c2)) / (c2 ** 2)
        return dc1 * Y - 2 * X / (c2 ** 2)

    res = optimize.minimize(func, np.array([0.1]), jac=grad, method="L-BFGS-B", bounds=((1e-9, np.log(2) - 1e-9),))
    return res.fun


def binary_kl_bound(q, c):
    q = max(q, 1e-9)

    def func(x):
        return bkl(q, (q+x) / 2) - c

    try:
        res = optimize.root_scalar(func, bracket=[q, 1], method='brentq')
        return res.root
    except:
        return 1.01


def binary_js_bound(q, c):
    q = max(q, 1e-9)

    def func(x):
        return bjs(q, x) - c

    try:
        res = optimize.root_scalar(func, bracket=[q, 1-1e-9], method='brentq')
        return res.root
    except:
        return 1.01


def f_kl(t):
    return t * np.log(t)


def f_sh(t):
    return np.square(np.sqrt(t) - 1)


def f_js(t):
    return t * np.log(2 * t / (1 + t)) + np.log(2 / (1 + t))


def disc2_mi(xs, nx, ys, ny, f):
    pxy = np.zeros((nx, ny), dtype=np.float64)
    for x in range(nx):
        for y in range(ny):
            pxy[x, y] = (np.logical_and(xs == x, ys == y).sum() / xs.shape[0])
    pxy = pxy.clip(1e-9, None)

    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    pxpy = px[:, None] @ py[None, :]
    mi = np.sum(f(pxy / pxpy) * pxpy)
    return mi.clip(1e-9, None)


def binarize_mi(xs, ys, ny, f):
    pxy = np.zeros((2, ny), dtype=np.float64)
    for y in range(ny):
        pxy[1, y] = (xs[ys == y].sum() / xs.shape[0])
        pxy[0, y] = ((1 - xs[ys == y]).sum() / xs.shape[0])
    pxy = pxy.clip(1e-9, None)

    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    pxpy = px[:, None] @ py[None, :]
    mi = np.sum(f(pxy / pxpy) * pxpy)
    return mi.clip(1e-9, None)


def phi(x):
    return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)


def Phi(x):
    return (1 + special.erf(x / np.sqrt(2))) / 2


def trunc_gauss_kde(xs, xmn, xmx, s):
    def pdf(x):
        return np.clip(np.sum(phi((x - xs) / s) / (Phi((xmx - xs) / s) - Phi((xmn - xs) / s)) / s) / xs.shape[0], 1e-9, None)

    return pdf


def f_divergence(pdf0, pdf1, xmn, xmx, f):
    def integral(x):
        return f(pdf0(x) / pdf1(x)) * pdf1(x)

    result = integrate.quad(integral, xmn, xmx, epsabs=1e-4)[0]
    return result


def cont_bin_mi(xs, xmn, xmx, ys, sigma, f):
    pdf = trunc_gauss_kde(xs, xmn, xmx, sigma)
    pdf0 = trunc_gauss_kde(xs[ys == 0], xmn, xmx, sigma)
    pdf1 = trunc_gauss_kde(xs[ys == 1], xmn, xmx, sigma)

    return (f_divergence(pdf0, pdf, xmn, xmx, f) + f_divergence(pdf1, pdf, xmn, xmx, f)) / 2


def estimate_mi_bound_discrete(n, masks, L0, L1, Lt, Lv):
    masks = masks.reshape(-1)
    L0 = L0.reshape(-1)
    L1 = L1.reshape(-1)

    loss_diff = L1 - L0
    ld_mi = disc2_mi(loss_diff + 1, 3, masks, 2, f_kl)
    bound_square = np.sqrt(2 * ld_mi) + Lt

    # use both L0 and L1 to reduce variance
    single_mi = disc2_mi(np.concatenate([L0, L1]), 2, np.concatenate([masks, 1-masks]), 2, f_kl)
    bound_fast = optimize_fast(single_mi, Lt)

    loss_pair = L1 * 2 + L0
    pair_mi = disc2_mi(loss_pair, 4, masks, 2, f_kl)
    bound_bkl = binary_kl_bound(Lt, pair_mi)

    ld_square = np.square(loss_diff).mean()
    err_abs = np.abs(((masks * 2 - 1) * loss_diff).mean())
    bound_cmi = np.sqrt(2 * (ld_square + err_abs) * ld_mi) + Lt

    ld_mi_sh = disc2_mi(loss_diff + 1, 3, masks, 2, f_sh)
    bound_cshi = np.sqrt((4 * ld_square + 2 * err_abs) * ld_mi_sh) + Lt

    ld_mi_js = disc2_mi(loss_diff + 1, 3, masks, 2, f_js)
    bound_cjsi = 2 * np.sqrt((4 * ld_square + err_abs) * ld_mi_js) + Lt

    bound_bjs = binary_js_bound(Lt, single_mi)

    return bound_square, bound_fast, bound_bkl, bound_cmi, bound_cshi, bound_cjsi, bound_bjs


def estimate_mi_bound_continuous(n, masks, L0, L1, Lt, Lv):
    masks = masks.reshape(-1)
    L0 = L0.reshape(-1)
    L1 = L1.reshape(-1)

    # rule-of-thumb bandwidth estimator
    kde_s = np.std(np.concatenate([L0, L1])) * 1.06 / np.power(n * 2, 1 / 5)

    loss_diff = L1 - L0
    ld_mi = cont_bin_mi(loss_diff, -1, 1, masks, kde_s, f_kl)
    bound_square = np.sqrt(2 * ld_mi) + Lt

    # use both L0 and L1 to reduce variance
    single_mi = cont_bin_mi(np.concatenate([L0, L1]), 0, 1, np.concatenate([masks, 1-masks]), kde_s, f_kl)
    bound_fast = optimize_fast(single_mi, Lt)

    # approximate pair_mi as the double of single_mi
    pair_mi = single_mi * 2
    bound_bkl = binary_kl_bound(Lt, pair_mi)

    ld_square = np.square(loss_diff).mean()
    err_abs = np.abs(((masks * 2 - 1) * loss_diff).mean())
    bound_cmi = np.sqrt(2 * (ld_square + err_abs) * ld_mi) + Lt

    ld_mi_sh = cont_bin_mi(loss_diff, -1, 1, masks, kde_s, f_sh)
    bound_cshi = np.sqrt((4 * ld_square + 2 * err_abs) * ld_mi_sh) + Lt

    ld_mi_js = cont_bin_mi(loss_diff, -1, 1, masks, kde_s, f_js)
    bound_cjsi = 2 * np.sqrt((4 * ld_square + err_abs) * ld_mi_js) + Lt

    bl_mi = binarize_mi(np.concatenate([L0, L1]), np.concatenate([masks, 1-masks]), 2, f_kl)
    bound_bjs = binary_js_bound(Lt, bl_mi)

    return bound_square, bound_fast, bound_bkl, bound_cmi, bound_cshi, bound_cjsi, bound_bjs


def _integer_codes_from_rows(rows):
    """Map a 2D array-like object to integer codes by unique rows."""
    rows = np.asarray(rows)
    if rows.ndim == 1:
        rows = rows.reshape(-1, 1)

    mapping = {}
    codes = np.empty(rows.shape[0], dtype=np.int64)
    next_code = 0
    for i, row in enumerate(rows):
        key = tuple(row.tolist())
        if key not in mapping:
            mapping[key] = next_code
            next_code += 1
        codes[i] = mapping[key]
    return codes, mapping


def _vector_projection_codes(values, num_bins, max_exact_values, value_range=None):
    """Discretize vector-valued output projections into empirical support codes.

    This shared helper is used *within one fixed supersample coordinate* for
    loss-pair statistics (T10). It bins vector components and assigns an exact
    code to each observed row; it does not pool distinct sample coordinates.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"projection values must have shape (num_models, dim); got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("projection values must be finite")

    unique_rows = np.unique(values, axis=0)
    if unique_rows.shape[0] <= max_exact_values:
        codes, mapping = _integer_codes_from_rows(values)
        return codes, {
            'mode': 'exact_rows',
            'support_size': len(mapping),
            'bin_edges': None,
            'value_range': value_range,
        }

    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2 for vector projection; got {num_bins}")

    if value_range is not None:
        value_min, value_max = value_range
        if not value_min < value_max:
            raise ValueError(f"value_range must be increasing; got {value_range}")
        bin_edges = np.linspace(value_min, value_max, num_bins + 1)
        binned = np.digitize(values, bin_edges[1:-1], right=False)
        codes, mapping = _integer_codes_from_rows(binned)
        return codes, {
            'mode': 'shared_histogram_rows',
            'support_size': len(mapping),
            'bin_edges': bin_edges,
            'value_range': value_range,
        }

    binned = np.zeros_like(values, dtype=np.int64)
    bin_edges = []
    for j in range(values.shape[1]):
        col = values[:, j]
        col_min = col.min()
        col_max = col.max()
        if np.isclose(col_min, col_max):
            edges = None
            binned[:, j] = 0
        else:
            edges = np.linspace(col_min, col_max, num_bins + 1)
            binned[:, j] = np.digitize(col, edges[1:-1], right=False)
        bin_edges.append(edges)

    codes, mapping = _integer_codes_from_rows(binned)
    return codes, {
        'mode': 'coordinate_histogram_rows',
        'support_size': len(mapping),
        'bin_edges': bin_edges,
        'value_range': value_range,
    }


def _validate_samplewise_coordinate_inputs(n, masks, **arrays):
    """Validate inputs for Theorems 8--13 and require both selector states per i."""
    masks = np.asarray(masks, dtype=np.int64)
    if masks.ndim != 2 or masks.shape[1] != n:
        raise ValueError(
            f"masks must have shape (num_selector_seeds, {n}); got {masks.shape}"
        )
    if masks.shape[0] < 2:
        raise ValueError("at least two selector seeds/models are required")
    if np.any((masks != 0) & (masks != 1)):
        raise ValueError("masks must contain only 0/1 selector values")

    checked = {}
    for name, values in arrays.items():
        if values is None:
            checked[name] = None
            continue
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != masks.shape:
            raise ValueError(
                f"{name} must match masks with shape {masks.shape}; got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        checked[name] = arr

    ones = masks.sum(axis=0)
    missing = np.flatnonzero((ones == 0) | (ones == masks.shape[0]))
    if missing.size:
        preview = ", ".join(str(int(i)) for i in missing[:10])
        raise ValueError(
            "The coordinate-wise Theorem 8--13 estimators require both U_i=0 "
            "and U_i=1 among the selector runs for every coordinate. Missing both "
            f"states at {missing.size} coordinate(s), first indices: {preview}. "
            "Increase the number/diversity of selector seeds."
        )
    return masks, checked


def _coordinatewise_codes(values, num_bins, max_exact_values, value_range=None):
    """Code one scalar/vector statistic independently for every supersample index.

    values has shape (num_selector_seeds, n) for scalar statistics or
    (num_selector_seeds, n, dim) for vector statistics. Codes are local to each
    coordinate i; equal integer codes at different i do *not* identify states.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in (2, 3):
        raise ValueError(
            "coordinate-wise coding expects shape (num_models, n) or "
            f"(num_models, n, dim); got {values.shape}"
        )
    num_models, n = values.shape[:2]
    codes = np.empty((num_models, n), dtype=np.int64)
    mode_counts = {}
    support_sizes = []

    for i in range(n):
        coordinate_values = values[:, i]
        if coordinate_values.ndim == 1:
            coordinate_values = coordinate_values.reshape(-1, 1)
        coordinate_codes, info = _vector_projection_codes(
            values=coordinate_values,
            num_bins=num_bins,
            max_exact_values=max_exact_values,
            value_range=value_range,
        )
        codes[:, i] = coordinate_codes
        mode = info.get('mode', 'unknown')
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        support_sizes.append(int(info.get('support_size', np.unique(coordinate_codes).size)))

    support_sizes = np.asarray(support_sizes, dtype=np.int64)
    return codes, {
        'coordinatewise': True,
        'num_coordinates': int(n),
        'mode_counts': mode_counts,
        'support_size_min': int(support_sizes.min()),
        'support_size_median': float(np.median(support_sizes)),
        'support_size_max': int(support_sizes.max()),
        'num_bins': int(num_bins),
        'max_exact_values': int(max_exact_values),
        'value_range': value_range,
    }


def _coordinatewise_joint_counts_from_codes(codes, masks):
    """Build one empirical (T_i,U_i) table per coordinate without pooling i."""
    codes = np.asarray(codes, dtype=np.int64)
    masks = np.asarray(masks, dtype=np.int64)
    if codes.shape != masks.shape:
        raise ValueError(f"codes and masks must match; got {codes.shape} and {masks.shape}")
    if np.any(codes < 0):
        raise ValueError("codes must be nonnegative")

    num_models, n = codes.shape
    num_states = int(codes.max()) + 1
    joint = np.zeros((n, num_states, 2), dtype=np.float64)
    coord = np.arange(n)
    for r in range(num_models):
        np.add.at(joint, (coord, codes[r], masks[r]), 1.0)
    return joint


def _coordinatewise_valpha_from_joint_counts(joint_counts, alpha, state_weights=None):
    """Compute conditional coordinate-wise V_alpha for a FAIR binary selector.

    P(U_i=0)=P(U_i=1)=1/2 is known by design. For any alpha >= 1,
        p(t) * lambda_alpha(t; U_i) = |p(t|0)-p(t|1)| / 2.
    Thus both V_alpha and the weighted expression are alpha-independent here.
    Do not substitute empirical selector frequencies for the prior: those
    frequencies are only used to estimate P(T_i=t | U_i=u).
    """
    joint_counts = np.asarray(joint_counts, dtype=np.float64)
    if joint_counts.ndim != 3 or joint_counts.shape[2] != 2:
        raise ValueError(
            "joint_counts must have shape (n, num_t_states, 2); "
            f"got {joint_counts.shape}"
        )
    alpha = float(alpha)
    if alpha < 1:
        raise ValueError(f"alpha must be at least 1; got {alpha}")
    if np.any(joint_counts < 0):
        raise ValueError("joint_counts must be nonnegative")

    u_mass = joint_counts.sum(axis=1)  # (n, 2)
    if np.any(u_mass <= 0):
        bad = np.argwhere(u_mass <= 0)
        raise ValueError(
            "cannot estimate a coordinate-wise conditional law because at least "
            f"one selector state has zero samples; first bad (i,u)={bad[0].tolist()}"
        )

    p_t_given_u = joint_counts / u_mass[:, None, :]
    selector_prior = np.array([0.5, 0.5], dtype=np.float64)
    p_t = np.sum(p_t_given_u * selector_prior[None, None, :], axis=2)

    # Exact identity for the Bernoulli(1/2) selector, including alpha=1.
    # Use the analytic form rather than powers of density ratios to eliminate
    # artificial alpha preferences from floating-point rounding.
    delta = np.abs(p_t_given_u[:, :, 0] - p_t_given_u[:, :, 1])
    r_alpha = np.zeros_like(p_t)
    np.divide(delta, 2.0 * p_t, out=r_alpha, where=p_t > 0)
    valpha_per_coordinate = np.sum(p_t * r_alpha, axis=1)

    weighted_per_coordinate = None
    if state_weights is not None:
        weights = np.asarray(state_weights, dtype=np.float64)
        if weights.ndim == 1:
            if weights.shape[0] != joint_counts.shape[1]:
                raise ValueError("state_weights length must match num_t_states")
            weights = np.broadcast_to(weights[None, :], p_t.shape)
        elif weights.shape != p_t.shape:
            raise ValueError(
                f"state_weights must have shape {p_t.shape} or ({p_t.shape[1]},); "
                f"got {weights.shape}"
            )
        weighted_per_coordinate = np.sum(p_t * weights * r_alpha, axis=1)

    return valpha_per_coordinate, r_alpha, weighted_per_coordinate


def _coordinatewise_weighted_from_samples(codes, masks, sample_weights, r_alpha):
    """Estimate E_{T_i}[w(T_i) R_alpha(T_i)] for each i with P(U_i)=1/2."""
    codes = np.asarray(codes, dtype=np.int64)
    masks = np.asarray(masks, dtype=np.int64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    r_alpha = np.asarray(r_alpha, dtype=np.float64)
    if codes.shape != masks.shape or sample_weights.shape != masks.shape:
        raise ValueError("codes, masks, and sample_weights must have matching shapes")
    if np.any(sample_weights < 0) or not np.all(np.isfinite(sample_weights)):
        raise ValueError("sample_weights must be finite and nonnegative")

    n = masks.shape[1]
    coord = np.arange(n)[None, :]
    r_samples = r_alpha[coord, codes]
    out = np.zeros(n, dtype=np.float64)
    for u in (0, 1):
        select = masks == u
        denom = select.sum(axis=0)
        if np.any(denom == 0):
            raise ValueError("both selector states are required at every coordinate")
        out += 0.5 * np.sum(select * sample_weights * r_samples, axis=0) / denom
    return out


def _coordinatewise_binary_expected_joint_counts(masks, probs):
    """Expected (B_i,U_i) tables, one 2x2 table per coordinate."""
    masks = np.asarray(masks, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if masks.shape != probs.shape:
        raise ValueError("masks and probs must have matching shapes")
    n = masks.shape[1]
    joint = np.zeros((n, 2, 2), dtype=np.float64)
    for u in (0, 1):
        select = masks == u
        joint[:, 0, u] = np.sum(select * (1.0 - probs), axis=0)
        joint[:, 1, u] = np.sum(select * probs, axis=0)
    return joint


def _coordinatewise_binarized_difference_joint_counts(masks, q0, q1):
    """Expected (Dbar_i,U_i) tables for Dbar_i in {-1,0,1}, per coordinate."""
    masks = np.asarray(masks, dtype=np.int64)
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if masks.shape != q0.shape or q0.shape != q1.shape:
        raise ValueError("masks, q0, and q1 must have matching shapes")
    p_m1 = q0 * (1.0 - q1)
    p_p1 = (1.0 - q0) * q1
    p_0 = (1.0 - q0) * (1.0 - q1) + q0 * q1
    n = masks.shape[1]
    joint = np.zeros((n, 3, 2), dtype=np.float64)
    for u in (0, 1):
        select = masks == u
        joint[:, 0, u] = np.sum(select * p_m1, axis=0)
        joint[:, 1, u] = np.sum(select * p_0, axis=0)
        joint[:, 2, u] = np.sum(select * p_p1, axis=0)
    return joint


def _coordinatewise_binarized_pair_joint_counts(masks, q0, q1):
    """Expected (Pbar_i,U_i) tables for Pbar_i in {0,1}^2, per coordinate."""
    masks = np.asarray(masks, dtype=np.int64)
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if masks.shape != q0.shape or q0.shape != q1.shape:
        raise ValueError("masks, q0, and q1 must have matching shapes")
    probs = (
        (1.0 - q0) * (1.0 - q1),
        (1.0 - q0) * q1,
        q0 * (1.0 - q1),
        q0 * q1,
    )
    n = masks.shape[1]
    joint = np.zeros((n, 4, 2), dtype=np.float64)
    for u in (0, 1):
        select = masks == u
        for state, prob in enumerate(probs):
            joint[:, state, u] = np.sum(select * prob, axis=0)
    return joint


def _coordinate_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(np.mean(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
    }


def estimate_theorem8_samplewise_loss_difference_valpha_bound(
        n,
        masks,
        L0_01,
        L1_01,
        L0_ce,
        L1_ce,
        train_loss_01=None,
        train_loss_ce=None,
        alphas=(1.05, 1.2, 1.5, 2.0, 3.0),
        num_bins=30,
        max_exact_values=20,
        bounded_loss_M_01=1.0,
        bounded_loss_M_ce=1.0,
        use_weighted=False):
    """Coordinate-wise plug-in estimator for Theorem 8.

    Crucially, coordinate i is never pooled with coordinate j. For each i we
    estimate P(D_i | U_i=u) from selector runs, compute its local V_alpha term,
    and only then average the theorem contribution over i. The exact selector
    prior P(U_i=0)=P(U_i=1)=1/2 is used.
    """
    masks, arrays = _validate_samplewise_coordinate_inputs(
        n, masks,
        L0_01=L0_01, L1_01=L1_01, L0_ce=L0_ce, L1_ce=L1_ce,
    )
    L0_01, L1_01 = arrays['L0_01'], arrays['L1_01']
    L0_ce, L1_ce = arrays['L0_ce'], arrays['L1_ce']
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(a < 1 for a in alphas):
        raise ValueError(f"Theorem 8 requires nonempty alphas >= 1; got {alphas}")

    D_01 = L1_01 - L0_01
    D_ce = L1_ce - L0_ce
    codes_01, code_info_01 = _coordinatewise_codes(
        D_01, num_bins, max_exact_values,
        value_range=(-float(bounded_loss_M_01), float(bounded_loss_M_01)),
    )
    codes_ce, code_info_ce = _coordinatewise_codes(
        D_ce, num_bins, max_exact_values,
        value_range=(-float(bounded_loss_M_ce), float(bounded_loss_M_ce)),
    )
    joint_01 = _coordinatewise_joint_counts_from_codes(codes_01, masks)
    joint_ce = _coordinatewise_joint_counts_from_codes(codes_ce, masks)

    by_alpha = []
    for alpha in alphas:
        v01_i, r01, _ = _coordinatewise_valpha_from_joint_counts(joint_01, alpha)
        vce_i, rce, _ = _coordinatewise_valpha_from_joint_counts(joint_ce, alpha)
        w01_i = _coordinatewise_weighted_from_samples(codes_01, masks, np.abs(D_01), r01)
        wce_i = _coordinatewise_weighted_from_samples(codes_ce, masks, np.abs(D_ce), rce)

        valpha_01 = float(np.mean(v01_i))
        valpha_ce = float(np.mean(vce_i))
        weighted_gap_bound_01 = float(np.mean(w01_i))
        weighted_gap_bound_ce = float(np.mean(wce_i))
        bounded_gap_bound_01 = float(bounded_loss_M_01 * valpha_01)
        bounded_gap_bound_ce = float(bounded_loss_M_ce * valpha_ce)
        gap_bound_01 = weighted_gap_bound_01 if use_weighted else bounded_gap_bound_01
        gap_bound_ce = weighted_gap_bound_ce if use_weighted else bounded_gap_bound_ce

        by_alpha.append({
            'alpha': float(alpha),
            'valpha_01': valpha_01,
            'weighted_gap_bound_01': weighted_gap_bound_01,
            'bounded_gap_bound_01': bounded_gap_bound_01,
            'gap_bound_01': gap_bound_01,
            'bound_with_train_loss_01': None if train_loss_01 is None else float(train_loss_01 + gap_bound_01),
            'valpha_ce': valpha_ce,
            'weighted_gap_bound_ce': weighted_gap_bound_ce,
            'bounded_gap_bound_ce': bounded_gap_bound_ce,
            'gap_bound_ce': gap_bound_ce,
            'bound_with_train_loss_ce': None if train_loss_ce is None else float(train_loss_ce + gap_bound_ce),
            'selected_form': 'weighted' if use_weighted else 'bounded',
            'valpha_coordinate_summary_01': _coordinate_summary(v01_i),
            'valpha_coordinate_summary_ce': _coordinate_summary(vce_i),
        })

    # For a fair binary U_i all entries have the SAME theoretical value.
    # Use the first alpha as a representative, not a tuned optimum.
    best_01 = by_alpha[0]
    best_ce = by_alpha[0]
    return {
        'method': 'Theorem 8',
        'result_key': 'theorem8_samplewise_loss_difference_valpha_bound',
        'is_theorem8_estimator': True,
        'coordinatewise': True,
        'selector_prior': [0.5, 0.5],
        'alpha_selection': 'first_grid_value_representative_not_optimized',
        'conditions_on': 'coordinate-wise scalar loss differences (U_i,D_i) on fixed supersample tilde{S}',
        'selected_form': 'weighted' if use_weighted else 'bounded',
        'num_selector_seeds': int(masks.shape[0]),
        'num_supersample_pairs': int(n),
        'bounded_loss_M_01': float(bounded_loss_M_01),
        'bounded_loss_M_ce': float(bounded_loss_M_ce),
        'loss_difference_info_01': code_info_01,
        'loss_difference_info_ce': code_info_ce,
        'by_alpha': by_alpha,
        'best_alpha_01': best_01['alpha'],
        'best_gap_bound_01': best_01['gap_bound_01'],
        'best_weighted_gap_bound_01': best_01['weighted_gap_bound_01'],
        'best_bounded_gap_bound_01': best_01['bounded_gap_bound_01'],
        'best_bound_with_train_loss_01': best_01['bound_with_train_loss_01'],
        'best_alpha_ce': best_ce['alpha'],
        'best_gap_bound_ce': best_ce['gap_bound_ce'],
        'best_weighted_gap_bound_ce': best_ce['weighted_gap_bound_ce'],
        'best_bounded_gap_bound_ce': best_ce['bounded_gap_bound_ce'],
        'best_bound_with_train_loss_ce': best_ce['bound_with_train_loss_ce'],
        'warnings': [],
    }


def estimate_theorem9_samplewise_single_loss_valpha_bound(
        n,
        masks,
        L0_01,
        L0_ce,
        L1_01=None,
        L1_ce=None,
        train_loss_01=None,
        train_loss_ce=None,
        alphas=(1.05, 1.2, 1.5, 2.0, 3.0),
        num_bins=30,
        max_exact_values=20,
        bounded_loss_M_01=1.0,
        bounded_loss_M_ce=1.0,
        center_01=0.5,
        center_ce=0.5,
        use_weighted=True,
        symmetrization='average'):
    """Conditional coordinate-wise T9 plug-in, with both sides by default.

    With symmetrization='average', the population bound equals the sum of
    the side-0 and side-1 conditional weighted terms (or M/2 times their
    V_alpha sum). 'max' is a looser variant; 'none' is a one-sided diagnostic,
    NOT a certified conditional bound on a fixed supersample.
    """
    masks, arrays = _validate_samplewise_coordinate_inputs(
        n, masks,
        L0_01=L0_01, L0_ce=L0_ce, L1_01=L1_01, L1_ce=L1_ce,
    )
    L0_01, L0_ce = arrays['L0_01'], arrays['L0_ce']
    L1_01, L1_ce = arrays['L1_01'], arrays['L1_ce']
    if symmetrization not in ('none', 'average', 'max'):
        raise ValueError("symmetrization must be one of {'none','average','max'}")
    if symmetrization != 'none' and (L1_01 is None or L1_ce is None):
        raise ValueError("symmetrization requires both side-1 loss arrays")
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(a < 1 for a in alphas):
        raise ValueError(f"Theorem 9 requires nonempty alphas >= 1; got {alphas}")

    def side_estimate(loss_01, loss_ce):
        codes_01, info_01 = _coordinatewise_codes(
            loss_01, num_bins, max_exact_values,
            value_range=(0.0, float(bounded_loss_M_01)),
        )
        codes_ce, info_ce = _coordinatewise_codes(
            loss_ce, num_bins, max_exact_values,
            value_range=(0.0, float(bounded_loss_M_ce)),
        )
        joint_01 = _coordinatewise_joint_counts_from_codes(codes_01, masks)
        joint_ce = _coordinatewise_joint_counts_from_codes(codes_ce, masks)
        out = {}
        for alpha in alphas:
            v01_i, r01, _ = _coordinatewise_valpha_from_joint_counts(joint_01, alpha)
            vce_i, rce, _ = _coordinatewise_valpha_from_joint_counts(joint_ce, alpha)
            w01_i = _coordinatewise_weighted_from_samples(
                codes_01, masks, np.abs(loss_01 - center_01), r01,
            )
            wce_i = _coordinatewise_weighted_from_samples(
                codes_ce, masks, np.abs(loss_ce - center_ce), rce,
            )
            valpha_01 = float(np.mean(v01_i))
            valpha_ce = float(np.mean(vce_i))
            weighted_gap_bound_01 = float(2.0 * np.mean(w01_i))
            weighted_gap_bound_ce = float(2.0 * np.mean(wce_i))
            bounded_gap_bound_01 = float(bounded_loss_M_01 * valpha_01)
            bounded_gap_bound_ce = float(bounded_loss_M_ce * valpha_ce)
            out[alpha] = {
                'alpha': float(alpha),
                'valpha_01': valpha_01,
                'weighted_gap_bound_01': weighted_gap_bound_01,
                'bounded_gap_bound_01': bounded_gap_bound_01,
                'gap_bound_01': weighted_gap_bound_01 if use_weighted else bounded_gap_bound_01,
                'valpha_ce': valpha_ce,
                'weighted_gap_bound_ce': weighted_gap_bound_ce,
                'bounded_gap_bound_ce': bounded_gap_bound_ce,
                'gap_bound_ce': weighted_gap_bound_ce if use_weighted else bounded_gap_bound_ce,
                'selected_form': 'weighted' if use_weighted else 'bounded',
                'valpha_coordinate_summary_01': _coordinate_summary(v01_i),
                'valpha_coordinate_summary_ce': _coordinate_summary(vce_i),
            }
        return out, info_01, info_ce

    side0, info0_01, info0_ce = side_estimate(L0_01, L0_ce)
    side1 = info1_01 = info1_ce = None
    if symmetrization != 'none':
        side1, info1_01, info1_ce = side_estimate(L1_01, L1_ce)

    by_alpha = []
    combine_keys = (
        'valpha_01', 'weighted_gap_bound_01', 'bounded_gap_bound_01', 'gap_bound_01',
        'valpha_ce', 'weighted_gap_bound_ce', 'bounded_gap_bound_ce', 'gap_bound_ce',
    )
    for alpha in alphas:
        item0 = side0[alpha]
        item = dict(item0)
        if side1 is not None:
            item1 = side1[alpha]
            reducer = (lambda a, b: 0.5 * (a + b)) if symmetrization == 'average' else max
            for key in combine_keys:
                item[key] = float(reducer(item0[key], item1[key]))
            item.update({
                'side0_gap_bound_01': item0['gap_bound_01'],
                'side1_gap_bound_01': item1['gap_bound_01'],
                'side0_gap_bound_ce': item0['gap_bound_ce'],
                'side1_gap_bound_ce': item1['gap_bound_ce'],
                'side0_valpha_01': item0['valpha_01'],
                'side1_valpha_01': item1['valpha_01'],
                'side0_valpha_ce': item0['valpha_ce'],
                'side1_valpha_ce': item1['valpha_ce'],
            })
        else:
            item.update({
                'side0_gap_bound_01': item0['gap_bound_01'],
                'side1_gap_bound_01': None,
                'side0_gap_bound_ce': item0['gap_bound_ce'],
                'side1_gap_bound_ce': None,
            })
        item['bound_with_train_loss_01'] = None if train_loss_01 is None else float(train_loss_01 + item['gap_bound_01'])
        item['bound_with_train_loss_ce'] = None if train_loss_ce is None else float(train_loss_ce + item['gap_bound_ce'])
        item['symmetrization'] = symmetrization
        by_alpha.append(item)

    # For a fair binary U_i all entries have the SAME theoretical value.
    # Use the first alpha as a representative, not a tuned optimum.
    best_01 = by_alpha[0]
    best_ce = by_alpha[0]
    warnings = []
    if symmetrization == 'none':
        warnings.append(
            'One-sided estimation is not a certified conditional bound on a fixed supersample. '
            'Use symmetrization=average for the conditional corollary.'
        )
    elif symmetrization == 'max':
        warnings.append(
            'Side-max is a conservative relaxation of the certified '
            'side-averaged conditional expression.'
        )
    return {
        'method': 'Theorem 9',
        'result_key': 'theorem9_samplewise_single_loss_valpha_bound',
        'is_theorem9_estimator': True,
        'coordinatewise': True,
        'selector_prior': [0.5, 0.5],
        'alpha_selection': 'first_grid_value_representative_not_optimized',
        'conditions_on': 'fixed supersample tilde{S}; coordinate-wise scalar losses (U_i,L_i^b)',
        'selected_form': 'weighted' if use_weighted else 'bounded',
        'symmetrization': symmetrization,
        'bound_interpretation': ('conditional_two_sided' if symmetrization != 'none'
                                 else 'one_sided_diagnostic_only'),
        'num_selector_seeds': int(masks.shape[0]),
        'num_supersample_pairs': int(n),
        'bounded_loss_M_01': float(bounded_loss_M_01),
        'bounded_loss_M_ce': float(bounded_loss_M_ce),
        'center_01': float(center_01),
        'center_ce': float(center_ce),
        'single_loss_info_01': info0_01,
        'single_loss_info_ce': info0_ce,
        'single_loss_info_side0_01': info0_01,
        'single_loss_info_side0_ce': info0_ce,
        'single_loss_info_side1_01': info1_01,
        'single_loss_info_side1_ce': info1_ce,
        'by_alpha': by_alpha,
        'best_alpha_01': best_01['alpha'],
        'best_gap_bound_01': best_01['gap_bound_01'],
        'best_weighted_gap_bound_01': best_01['weighted_gap_bound_01'],
        'best_bounded_gap_bound_01': best_01['bounded_gap_bound_01'],
        'best_bound_with_train_loss_01': best_01['bound_with_train_loss_01'],
        'best_alpha_ce': best_ce['alpha'],
        'best_gap_bound_ce': best_ce['gap_bound_ce'],
        'best_weighted_gap_bound_ce': best_ce['weighted_gap_bound_ce'],
        'best_bounded_gap_bound_ce': best_ce['bounded_gap_bound_ce'],
        'best_bound_with_train_loss_ce': best_ce['bound_with_train_loss_ce'],
        'warnings': warnings,
    }


def _normalize_bounded_losses(values, M, name):
    values = np.asarray(values, dtype=np.float64)
    if M <= 0:
        raise ValueError(f"{name}: M must be positive; got {M}")
    q = values / float(M)
    # Numerical guard for tiny clipping errors.
    q = np.clip(q, 0.0, 1.0)
    return q


def estimate_theorem10_samplewise_loss_pair_valpha_bound(
        n,
        masks,
        L0_01,
        L1_01,
        L0_ce,
        L1_ce,
        train_loss_01=None,
        train_loss_ce=None,
        alphas=(1.05, 1.2, 1.5, 2.0, 3.0),
        num_bins=30,
        max_exact_values=20,
        bounded_loss_M_01=1.0,
        bounded_loss_M_ce=1.0,
        use_weighted=True):
    """Coordinate-wise plug-in estimator for Theorem 10 loss pairs."""
    masks, arrays = _validate_samplewise_coordinate_inputs(
        n, masks,
        L0_01=L0_01, L1_01=L1_01, L0_ce=L0_ce, L1_ce=L1_ce,
    )
    L0_01, L1_01 = arrays['L0_01'], arrays['L1_01']
    L0_ce, L1_ce = arrays['L0_ce'], arrays['L1_ce']
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(a < 1 for a in alphas):
        raise ValueError(f"Theorem 10 requires nonempty alphas >= 1; got {alphas}")

    pair_01 = np.stack([L0_01, L1_01], axis=2)
    pair_ce = np.stack([L0_ce, L1_ce], axis=2)
    codes_01, info_01 = _coordinatewise_codes(
        pair_01, num_bins, max_exact_values,
        value_range=(0.0, float(bounded_loss_M_01)),
    )
    codes_ce, info_ce = _coordinatewise_codes(
        pair_ce, num_bins, max_exact_values,
        value_range=(0.0, float(bounded_loss_M_ce)),
    )
    joint_01 = _coordinatewise_joint_counts_from_codes(codes_01, masks)
    joint_ce = _coordinatewise_joint_counts_from_codes(codes_ce, masks)
    weights_01 = np.abs(L1_01 - L0_01)
    weights_ce = np.abs(L1_ce - L0_ce)

    by_alpha = []
    for alpha in alphas:
        v01_i, r01, _ = _coordinatewise_valpha_from_joint_counts(joint_01, alpha)
        vce_i, rce, _ = _coordinatewise_valpha_from_joint_counts(joint_ce, alpha)
        w01_i = _coordinatewise_weighted_from_samples(codes_01, masks, weights_01, r01)
        wce_i = _coordinatewise_weighted_from_samples(codes_ce, masks, weights_ce, rce)
        valpha_01, valpha_ce = float(np.mean(v01_i)), float(np.mean(vce_i))
        weighted_01, weighted_ce = float(np.mean(w01_i)), float(np.mean(wce_i))
        bounded_01 = float(bounded_loss_M_01 * valpha_01)
        bounded_ce = float(bounded_loss_M_ce * valpha_ce)
        gap_01 = weighted_01 if use_weighted else bounded_01
        gap_ce = weighted_ce if use_weighted else bounded_ce
        by_alpha.append({
            'alpha': float(alpha),
            'valpha_01': valpha_01,
            'weighted_gap_bound_01': weighted_01,
            'bounded_gap_bound_01': bounded_01,
            'gap_bound_01': gap_01,
            'bound_with_train_loss_01': None if train_loss_01 is None else float(train_loss_01 + gap_01),
            'valpha_ce': valpha_ce,
            'weighted_gap_bound_ce': weighted_ce,
            'bounded_gap_bound_ce': bounded_ce,
            'gap_bound_ce': gap_ce,
            'bound_with_train_loss_ce': None if train_loss_ce is None else float(train_loss_ce + gap_ce),
            'selected_form': 'weighted' if use_weighted else 'bounded',
            'valpha_coordinate_summary_01': _coordinate_summary(v01_i),
            'valpha_coordinate_summary_ce': _coordinate_summary(vce_i),
        })

    # For a fair binary U_i all entries have the SAME theoretical value.
    # Use the first alpha as a representative, not a tuned optimum.
    best_01 = by_alpha[0]
    best_ce = by_alpha[0]
    return {
        'method': 'Theorem 10',
        'result_key': 'theorem10_samplewise_loss_pair_valpha_bound',
        'is_theorem10_estimator': True,
        'coordinatewise': True,
        'selector_prior': [0.5, 0.5],
        'alpha_selection': 'first_grid_value_representative_not_optimized',
        'conditions_on': 'fixed supersample tilde{S}; coordinate-wise loss pairs (U_i,(L_i^0,L_i^1))',
        'selected_form': 'weighted' if use_weighted else 'bounded',
        'num_selector_seeds': int(masks.shape[0]),
        'num_supersample_pairs': int(n),
        'bounded_loss_M_01': float(bounded_loss_M_01),
        'bounded_loss_M_ce': float(bounded_loss_M_ce),
        'loss_pair_info_01': info_01,
        'loss_pair_info_ce': info_ce,
        'by_alpha': by_alpha,
        'best_alpha_01': best_01['alpha'],
        'best_gap_bound_01': best_01['gap_bound_01'],
        'best_weighted_gap_bound_01': best_01['weighted_gap_bound_01'],
        'best_bounded_gap_bound_01': best_01['bounded_gap_bound_01'],
        'best_bound_with_train_loss_01': best_01['bound_with_train_loss_01'],
        'best_alpha_ce': best_ce['alpha'],
        'best_gap_bound_ce': best_ce['gap_bound_ce'],
        'best_weighted_gap_bound_ce': best_ce['weighted_gap_bound_ce'],
        'best_bounded_gap_bound_ce': best_ce['bounded_gap_bound_ce'],
        'best_bound_with_train_loss_ce': best_ce['bound_with_train_loss_ce'],
        'warnings': [],
    }


def estimate_theorem11_samplewise_binarized_single_loss_valpha_bound(
        n,
        masks,
        L0_01,
        L0_ce,
        L1_01=None,
        L1_ce=None,
        train_loss_01=None,
        train_loss_ce=None,
        alphas=(1.05, 1.2, 1.5, 2.0, 3.0),
        bounded_loss_M_01=1.0,
        bounded_loss_M_ce=1.0,
        symmetrization='average'):
    """Conditional coordinate-wise T11 plug-in, with both sides by default.

    The two-sided 'average' has a conditional bound proof. 'max' is looser;
    'none' is an unprotected one-sided diagnostic for fixed supersamples.
    """
    masks, arrays = _validate_samplewise_coordinate_inputs(
        n, masks,
        L0_01=L0_01, L0_ce=L0_ce, L1_01=L1_01, L1_ce=L1_ce,
    )
    L0_01, L0_ce = arrays['L0_01'], arrays['L0_ce']
    L1_01, L1_ce = arrays['L1_01'], arrays['L1_ce']
    if symmetrization not in ('none', 'average', 'max'):
        raise ValueError("symmetrization must be one of {'none','average','max'}")
    if symmetrization != 'none' and (L1_01 is None or L1_ce is None):
        raise ValueError("symmetrization requires both side-1 loss arrays")
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(a < 1 for a in alphas):
        raise ValueError(f"Theorem 11 requires nonempty alphas >= 1; got {alphas}")

    def side_estimate(loss_01, loss_ce):
        q01 = _normalize_bounded_losses(loss_01, bounded_loss_M_01, 'Theorem 11 0-1')
        qce = _normalize_bounded_losses(loss_ce, bounded_loss_M_ce, 'Theorem 11 CE')
        joint01 = _coordinatewise_binary_expected_joint_counts(masks, q01)
        jointce = _coordinatewise_binary_expected_joint_counts(masks, qce)
        out = {}
        for alpha in alphas:
            v01_i, _, _ = _coordinatewise_valpha_from_joint_counts(joint01, alpha)
            vce_i, _, _ = _coordinatewise_valpha_from_joint_counts(jointce, alpha)
            valpha_01, valpha_ce = float(np.mean(v01_i)), float(np.mean(vce_i))
            out[alpha] = {
                'alpha': float(alpha),
                'valpha_01': valpha_01,
                'gap_bound_01': float(bounded_loss_M_01 * valpha_01),
                'valpha_ce': valpha_ce,
                'gap_bound_ce': float(bounded_loss_M_ce * valpha_ce),
                'valpha_coordinate_summary_01': _coordinate_summary(v01_i),
                'valpha_coordinate_summary_ce': _coordinate_summary(vce_i),
            }
        return out

    side0 = side_estimate(L0_01, L0_ce)
    side1 = side_estimate(L1_01, L1_ce) if symmetrization != 'none' else None
    by_alpha = []
    for alpha in alphas:
        item0 = side0[alpha]
        item = dict(item0)
        if side1 is not None:
            item1 = side1[alpha]
            reducer = (lambda a, b: 0.5 * (a + b)) if symmetrization == 'average' else max
            for key in ('valpha_01', 'gap_bound_01', 'valpha_ce', 'gap_bound_ce'):
                item[key] = float(reducer(item0[key], item1[key]))
            item.update({
                'side0_gap_bound_01': item0['gap_bound_01'],
                'side1_gap_bound_01': item1['gap_bound_01'],
                'side0_gap_bound_ce': item0['gap_bound_ce'],
                'side1_gap_bound_ce': item1['gap_bound_ce'],
                'side0_valpha_01': item0['valpha_01'],
                'side1_valpha_01': item1['valpha_01'],
                'side0_valpha_ce': item0['valpha_ce'],
                'side1_valpha_ce': item1['valpha_ce'],
            })
        else:
            item.update({
                'side0_gap_bound_01': item0['gap_bound_01'],
                'side1_gap_bound_01': None,
                'side0_gap_bound_ce': item0['gap_bound_ce'],
                'side1_gap_bound_ce': None,
            })
        item['bound_with_train_loss_01'] = None if train_loss_01 is None else float(train_loss_01 + item['gap_bound_01'])
        item['bound_with_train_loss_ce'] = None if train_loss_ce is None else float(train_loss_ce + item['gap_bound_ce'])
        item['symmetrization'] = symmetrization
        by_alpha.append(item)

    # For a fair binary U_i all entries have the SAME theoretical value.
    # Use the first alpha as a representative, not a tuned optimum.
    best_01 = by_alpha[0]
    best_ce = by_alpha[0]
    warnings = []
    if symmetrization == 'none':
        warnings.append(
            'One-sided estimation is not a certified conditional bound on a fixed supersample. '
            'Use symmetrization=average for the conditional corollary.'
        )
    elif symmetrization == 'max':
        warnings.append(
            'Side-max is a conservative relaxation of the certified '
            'side-averaged conditional expression.'
        )
    return {
        'method': 'Theorem 11',
        'result_key': 'theorem11_samplewise_binarized_single_loss_valpha_bound',
        'is_theorem11_estimator': True,
        'coordinatewise': True,
        'selector_prior': [0.5, 0.5],
        'alpha_selection': 'first_grid_value_representative_not_optimized',
        'conditions_on': 'fixed supersample tilde{S}; coordinate-wise binarized losses (U_i,L_i^b)',
        'symmetrization': symmetrization,
        'bound_interpretation': ('conditional_two_sided' if symmetrization != 'none'
                                 else 'one_sided_diagnostic_only'),
        'num_selector_seeds': int(masks.shape[0]),
        'num_supersample_pairs': int(n),
        'bounded_loss_M_01': float(bounded_loss_M_01),
        'bounded_loss_M_ce': float(bounded_loss_M_ce),
        'by_alpha': by_alpha,
        'best_alpha_01': best_01['alpha'],
        'best_gap_bound_01': best_01['gap_bound_01'],
        'best_bound_with_train_loss_01': best_01['bound_with_train_loss_01'],
        'best_alpha_ce': best_ce['alpha'],
        'best_gap_bound_ce': best_ce['gap_bound_ce'],
        'best_bound_with_train_loss_ce': best_ce['bound_with_train_loss_ce'],
        'warnings': warnings,
    }


def estimate_theorem12_samplewise_binarized_loss_difference_valpha_bound(
        n,
        masks,
        L0_01,
        L1_01,
        L0_ce,
        L1_ce,
        train_loss_01=None,
        train_loss_ce=None,
        alphas=(1.05, 1.2, 1.5, 2.0, 3.0),
        bounded_loss_M_01=1.0,
        bounded_loss_M_ce=1.0,
        use_weighted=True):
    """Coordinate-wise expected-table estimator for Theorem 12."""
    masks, arrays = _validate_samplewise_coordinate_inputs(
        n, masks,
        L0_01=L0_01, L1_01=L1_01, L0_ce=L0_ce, L1_ce=L1_ce,
    )
    L0_01, L1_01 = arrays['L0_01'], arrays['L1_01']
    L0_ce, L1_ce = arrays['L0_ce'], arrays['L1_ce']
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(a < 1 for a in alphas):
        raise ValueError(f"Theorem 12 requires nonempty alphas >= 1; got {alphas}")

    q0_01 = _normalize_bounded_losses(L0_01, bounded_loss_M_01, 'Theorem 12 0-1 L0')
    q1_01 = _normalize_bounded_losses(L1_01, bounded_loss_M_01, 'Theorem 12 0-1 L1')
    q0_ce = _normalize_bounded_losses(L0_ce, bounded_loss_M_ce, 'Theorem 12 CE L0')
    q1_ce = _normalize_bounded_losses(L1_ce, bounded_loss_M_ce, 'Theorem 12 CE L1')
    joint_01 = _coordinatewise_binarized_difference_joint_counts(masks, q0_01, q1_01)
    joint_ce = _coordinatewise_binarized_difference_joint_counts(masks, q0_ce, q1_ce)
    state_weights = np.array([1.0, 0.0, 1.0], dtype=np.float64)

    by_alpha = []
    for alpha in alphas:
        v01_i, _, w01_i = _coordinatewise_valpha_from_joint_counts(joint_01, alpha, state_weights)
        vce_i, _, wce_i = _coordinatewise_valpha_from_joint_counts(joint_ce, alpha, state_weights)
        valpha_01, valpha_ce = float(np.mean(v01_i)), float(np.mean(vce_i))
        weighted_01 = float(bounded_loss_M_01 * np.mean(w01_i))
        weighted_ce = float(bounded_loss_M_ce * np.mean(wce_i))
        bounded_01 = float(bounded_loss_M_01 * valpha_01)
        bounded_ce = float(bounded_loss_M_ce * valpha_ce)
        gap_01 = weighted_01 if use_weighted else bounded_01
        gap_ce = weighted_ce if use_weighted else bounded_ce
        by_alpha.append({
            'alpha': float(alpha),
            'valpha_01': valpha_01,
            'weighted_gap_bound_01': weighted_01,
            'bounded_gap_bound_01': bounded_01,
            'gap_bound_01': gap_01,
            'bound_with_train_loss_01': None if train_loss_01 is None else float(train_loss_01 + gap_01),
            'valpha_ce': valpha_ce,
            'weighted_gap_bound_ce': weighted_ce,
            'bounded_gap_bound_ce': bounded_ce,
            'gap_bound_ce': gap_ce,
            'bound_with_train_loss_ce': None if train_loss_ce is None else float(train_loss_ce + gap_ce),
            'selected_form': 'weighted' if use_weighted else 'bounded',
            'valpha_coordinate_summary_01': _coordinate_summary(v01_i),
            'valpha_coordinate_summary_ce': _coordinate_summary(vce_i),
        })

    # For a fair binary U_i all entries have the SAME theoretical value.
    # Use the first alpha as a representative, not a tuned optimum.
    best_01 = by_alpha[0]
    best_ce = by_alpha[0]
    return {
        'method': 'Theorem 12',
        'result_key': 'theorem12_samplewise_binarized_loss_difference_valpha_bound',
        'is_theorem12_estimator': True,
        'coordinatewise': True,
        'selector_prior': [0.5, 0.5],
        'alpha_selection': 'first_grid_value_representative_not_optimized',
        'conditions_on': 'fixed supersample tilde{S}; coordinate-wise binarized loss differences (U_i,Dbar_i)',
        'selected_form': 'weighted' if use_weighted else 'bounded',
        'num_selector_seeds': int(masks.shape[0]),
        'num_supersample_pairs': int(n),
        'bounded_loss_M_01': float(bounded_loss_M_01),
        'bounded_loss_M_ce': float(bounded_loss_M_ce),
        'by_alpha': by_alpha,
        'best_alpha_01': best_01['alpha'],
        'best_gap_bound_01': best_01['gap_bound_01'],
        'best_weighted_gap_bound_01': best_01['weighted_gap_bound_01'],
        'best_bounded_gap_bound_01': best_01['bounded_gap_bound_01'],
        'best_bound_with_train_loss_01': best_01['bound_with_train_loss_01'],
        'best_alpha_ce': best_ce['alpha'],
        'best_gap_bound_ce': best_ce['gap_bound_ce'],
        'best_weighted_gap_bound_ce': best_ce['weighted_gap_bound_ce'],
        'best_bounded_gap_bound_ce': best_ce['bounded_gap_bound_ce'],
        'best_bound_with_train_loss_ce': best_ce['bound_with_train_loss_ce'],
        'warnings': [],
    }


def estimate_theorem13_samplewise_binarized_loss_pair_valpha_bound(
        n,
        masks,
        L0_01,
        L1_01,
        L0_ce,
        L1_ce,
        train_loss_01=None,
        train_loss_ce=None,
        alphas=(1.05, 1.2, 1.5, 2.0, 3.0),
        bounded_loss_M_01=1.0,
        bounded_loss_M_ce=1.0,
        use_weighted=True):
    """Coordinate-wise expected-table estimator for Theorem 13."""
    masks, arrays = _validate_samplewise_coordinate_inputs(
        n, masks,
        L0_01=L0_01, L1_01=L1_01, L0_ce=L0_ce, L1_ce=L1_ce,
    )
    L0_01, L1_01 = arrays['L0_01'], arrays['L1_01']
    L0_ce, L1_ce = arrays['L0_ce'], arrays['L1_ce']
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(a < 1 for a in alphas):
        raise ValueError(f"Theorem 13 requires nonempty alphas >= 1; got {alphas}")

    q0_01 = _normalize_bounded_losses(L0_01, bounded_loss_M_01, 'Theorem 13 0-1 L0')
    q1_01 = _normalize_bounded_losses(L1_01, bounded_loss_M_01, 'Theorem 13 0-1 L1')
    q0_ce = _normalize_bounded_losses(L0_ce, bounded_loss_M_ce, 'Theorem 13 CE L0')
    q1_ce = _normalize_bounded_losses(L1_ce, bounded_loss_M_ce, 'Theorem 13 CE L1')
    joint_01 = _coordinatewise_binarized_pair_joint_counts(masks, q0_01, q1_01)
    joint_ce = _coordinatewise_binarized_pair_joint_counts(masks, q0_ce, q1_ce)
    state_weights = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)

    by_alpha = []
    for alpha in alphas:
        v01_i, _, w01_i = _coordinatewise_valpha_from_joint_counts(joint_01, alpha, state_weights)
        vce_i, _, wce_i = _coordinatewise_valpha_from_joint_counts(joint_ce, alpha, state_weights)
        valpha_01, valpha_ce = float(np.mean(v01_i)), float(np.mean(vce_i))
        weighted_01 = float(bounded_loss_M_01 * np.mean(w01_i))
        weighted_ce = float(bounded_loss_M_ce * np.mean(wce_i))
        bounded_01 = float(bounded_loss_M_01 * valpha_01)
        bounded_ce = float(bounded_loss_M_ce * valpha_ce)
        gap_01 = weighted_01 if use_weighted else bounded_01
        gap_ce = weighted_ce if use_weighted else bounded_ce
        by_alpha.append({
            'alpha': float(alpha),
            'valpha_01': valpha_01,
            'weighted_gap_bound_01': weighted_01,
            'bounded_gap_bound_01': bounded_01,
            'gap_bound_01': gap_01,
            'bound_with_train_loss_01': None if train_loss_01 is None else float(train_loss_01 + gap_01),
            'valpha_ce': valpha_ce,
            'weighted_gap_bound_ce': weighted_ce,
            'bounded_gap_bound_ce': bounded_ce,
            'gap_bound_ce': gap_ce,
            'bound_with_train_loss_ce': None if train_loss_ce is None else float(train_loss_ce + gap_ce),
            'selected_form': 'weighted' if use_weighted else 'bounded',
            'valpha_coordinate_summary_01': _coordinate_summary(v01_i),
            'valpha_coordinate_summary_ce': _coordinate_summary(vce_i),
        })

    # For a fair binary U_i all entries have the SAME theoretical value.
    # Use the first alpha as a representative, not a tuned optimum.
    best_01 = by_alpha[0]
    best_ce = by_alpha[0]
    return {
        'method': 'Theorem 13',
        'result_key': 'theorem13_samplewise_binarized_loss_pair_valpha_bound',
        'is_theorem13_estimator': True,
        'coordinatewise': True,
        'selector_prior': [0.5, 0.5],
        'alpha_selection': 'first_grid_value_representative_not_optimized',
        'conditions_on': 'fixed supersample tilde{S}; coordinate-wise binarized loss pairs (U_i,Pbar_i)',
        'selected_form': 'weighted' if use_weighted else 'bounded',
        'num_selector_seeds': int(masks.shape[0]),
        'num_supersample_pairs': int(n),
        'bounded_loss_M_01': float(bounded_loss_M_01),
        'bounded_loss_M_ce': float(bounded_loss_M_ce),
        'by_alpha': by_alpha,
        'best_alpha_01': best_01['alpha'],
        'best_gap_bound_01': best_01['gap_bound_01'],
        'best_weighted_gap_bound_01': best_01['weighted_gap_bound_01'],
        'best_bounded_gap_bound_01': best_01['bounded_gap_bound_01'],
        'best_bound_with_train_loss_01': best_01['bound_with_train_loss_01'],
        'best_alpha_ce': best_ce['alpha'],
        'best_gap_bound_ce': best_ce['gap_bound_ce'],
        'best_weighted_gap_bound_ce': best_ce['weighted_gap_bound_ce'],
        'best_bounded_gap_bound_ce': best_ce['bounded_gap_bound_ce'],
        'best_bound_with_train_loss_ce': best_ce['bound_with_train_loss_ce'],
        'warnings': [],
    }
