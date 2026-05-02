# report_bundle.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Sequence, Optional, Tuple, List

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re


# ==============================
# Saving utilities (NO SHOW)
# ==============================
def _reports_dir(out_dir: Optional[Path] = None) -> Path:
    d = Path(out_dir) if out_dir is not None else (Path.cwd() / "reports")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_close(fig, filepath: Path, dpi: int = 200) -> Path:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ==============================
# Style helpers (FONT SIZES)
# ==============================
def _apply_axes_style(
    ax,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel_fs: int = 14,
    ylabel_fs: int = 14,
    xtick_fs: int = 12,
    ytick_fs: int = 12,
):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xlabel_fs)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_fs)

    ax.tick_params(axis="x", labelsize=xtick_fs)
    ax.tick_params(axis="y", labelsize=ytick_fs)


def _legend(
    ax,
    *,
    handles=None,
    labels=None,
    loc: str = "best",
    ncol: int = 1,
    legend_fs: int = 12,
    legend_title: Optional[str] = None,
    legend_title_fs: int = 12,
):
    """
    Reliable legend fontsize control:
      - fontsize=... controls label fontsize
      - title_fontsize=... controls title fontsize
      - AND we also set title.set_fontsize(...) explicitly (some backends can be finicky)
    """
    leg = ax.legend(
        handles=handles,
        labels=labels,
        loc=loc,
        ncol=ncol,
        fontsize=legend_fs,
        title=legend_title,
        title_fontsize=legend_title_fs,
        frameon=True,
    )
    if leg is not None and leg.get_title() is not None:
        leg.get_title().set_fontsize(legend_title_fs)
    return leg


# ==============================
# helpers
# ==============================
def to_1d_timeseries(mu: np.ndarray) -> np.ndarray:
    """
    mu: [T, d] expected value over time
    returns: [T] mean across dimensions per time step
    """
    mu = np.asarray(mu)
    assert mu.ndim == 2, f"Expected [T,d], got shape {mu.shape}"
    return mu.mean(axis=1)



def to_1d_variance(cov: np.ndarray) -> np.ndarray:
    """
    cov: [T, d, d] covariance over time
    returns: [T] mean of diagonal (mean marginal variance) per time step
    """
    cov = np.asarray(cov)
    assert cov.ndim == 3, f"Expected [T,d,d], got shape {cov.shape}"
    return np.diagonal(cov, axis1=1, axis2=2).mean(axis=1)



def clip_T(y: np.ndarray, v: np.ndarray, T_plot: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    T = min(len(y), len(v))
    if T_plot is not None:
        T = min(T, int(T_plot))
    return y[:T], v[:T], T


def _safe_std_from_var(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    return np.sqrt(np.maximum(v, 0.0))


def _tail_slice(y: np.ndarray, v: np.ndarray, K: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if K is None:
        return y, v
    K = int(K)
    if K <= 0:
        raise ValueError("K must be a positive integer.")
    if len(y) <= K:
        return y, v
    return y[-K:], v[-K:]


def _slug_filename(s: str, maxlen: int = 140) -> str:
    s = re.sub(r"[^\w.\-]+", "_", s)
    s = s.strip("._-")
    return s[:maxlen] if len(s) > maxlen else s


# ==============================
# scanning: best run per constraint
# ==============================
def compute_best_for_kx(
    parent_dir: Path,
    kx_value: int,
    E_pi_y_constraints: Sequence[float],
) -> Dict[float, Dict[str, Any]]:
    sub_dir = Path(parent_dir) / f'kx={kx_value}'

    best: Dict[float, Dict[str, Any]] = {
        E: {
            "min_fa": float("inf"),
            "mse": None,
            "accuracy": None,
            "complexity": None,

            "lambda_x": None,
            "lambda_y": None,
            "theta": None,

            "cov_lambda_x": None,
            "cov_lambda_y": None,
            "cov_theta": None,

            "kappa_x": None,
            "kappa_x_all": [],
            "theta_interval": None,
            "theta_beta": None,
            "lambda_beta": None,
            "folder": None,
        }
        for E in E_pi_y_constraints
    }

    if not sub_dir.exists():
        return best

    for folder in sub_dir.iterdir():
        if not folder.is_dir():
            continue

        combo_path = folder / 'combo.json'
        if not combo_path.exists():
            continue

        with open(combo_path, "r") as f:
            combo = json.load(f)

        (kx, ky, gp_name, gp_integration_method,
         dt, T_total, f_name, g_name,
         E_theta, sigma_theta,
         E_pi_x, sigma_lambda_x,
         E_pi_y, sigma_lambda_y,
         nu_x, kappa_x,
         lambda_eta_adapt, lambda_eta_rate, lambda_eta_t_0, lambda_eta_gamma,
         lambda_interval, lambda_beta,
         theta_eta_adapt, theta_eta_rate, theta_eta_t_0, theta_eta_gamma, theta_interval, theta_beta, carry_cov,
         jitter, algorithm_name, device) = combo

        # NOTE: no exclusion of theta_interval (per your request)

        if E_pi_y not in best:
            continue

        best[E_pi_y]["kappa_x_all"].append(float(kappa_x))

        vfe_path = folder / "vfe.npy"
        acc_path = folder / "accuracy.npy"
        comp_path = folder / "complexity.npy"
        if not (vfe_path.exists() and acc_path.exists() and comp_path.exists()):
            continue

        fa = float(np.sum(np.load(vfe_path)))
        accuracy = float(np.sum(np.load(acc_path)))
        complexity = float(np.sum(np.load(comp_path)))

        if fa < best[E_pi_y]["min_fa"]:
            # MSE(x, x_hat)
            x_path = folder / "x_noisy.npy"
            xhat_path = folder / "gen_x_estimates.npy"
            if x_path.exists() and xhat_path.exists():
                x = np.load(x_path)
                x_hat = np.load(xhat_path)[:, 0, :]
                T = min(len(x), len(x_hat))
                mse = float(np.mean((x[:T] - x_hat[:T]) ** 2))
            else:
                mse = None

            best[E_pi_y]["min_fa"] = fa
            best[E_pi_y]["mse"] = mse
            best[E_pi_y]["accuracy"] = accuracy
            best[E_pi_y]["complexity"] = complexity

            best[E_pi_y]["lambda_x"] = np.load(folder / "lambda_x.npy") if (folder / "lambda_x.npy").exists() else None
            best[E_pi_y]["lambda_y"] = np.load(folder / "lambda_y.npy") if (folder / "lambda_y.npy").exists() else None
            best[E_pi_y]["theta"] = np.load(folder / "theta.npy") if (folder / "theta.npy").exists() else None

            best[E_pi_y]["cov_lambda_x"] = np.load(folder / "cov_lambda_x.npy") if (folder / "cov_lambda_x.npy").exists() else None
            best[E_pi_y]["cov_lambda_y"] = np.load(folder / "cov_lambda_y.npy") if (folder / "cov_lambda_y.npy").exists() else None
            best[E_pi_y]["cov_theta"] = np.load(folder / "cov_theta.npy") if (folder / "cov_theta.npy").exists() else None

            best[E_pi_y]["kappa_x"] = float(kappa_x)
            best[E_pi_y]["theta_interval"] = theta_interval
            best[E_pi_y]["theta_beta"] = theta_beta
            best[E_pi_y]["lambda_beta"] = lambda_beta
            best[E_pi_y]["folder"] = folder

    return best


def compute_best_by_kx(
    parent_dir: Path,
    kx_values: Sequence[int],
    E_pi_y_constraints: Sequence[float],
) -> Dict[int, Dict[float, Dict[str, Any]]]:
    return {kx: compute_best_for_kx(parent_dir, kx, E_pi_y_constraints) for kx in kx_values}


# ==============================
# core uncertainty plotting
# ==============================
def _plot_uncertainty_panel(
    ax,
    best_dict: Dict[float, Dict[str, Any]],
    E_pi_y_constraints: Sequence[float],
    E_pi_x_fix: float,
    key_mean: str,
    key_cov: str,
    z: float,
    T_plot: Optional[int],
    ylabel: str,
    *,
    dim: Optional[int] = None,          # NEW: choose channel; None = old 1D behavior
    force_xlim_0_10000: bool = False,   # NEW: optional convenience
    legend_ncol: int = 3,
    legend_loc: str = "best",
    tail_K: Optional[int] = None,
    # --- font sizes ---
    xlabel_fs: int = 14,
    ylabel_fs: int = 14,
    legend_fs: int = 12,
    xtick_fs: int = 12,
    ytick_fs: int = 12,
    legend_title_fs: int = 12,
):
    """
    If dim is None:
        Uses legacy behavior: to_1d_timeseries(mean), to_1d_variance(cov)
    If dim is an int:
        Uses channel-wise behavior:
          - mean: (T,) or (T,D) -> y = mean or mean[:,dim]
          - cov:  (T,), (T,D), or (T,D,D) -> v = cov, cov[:,dim], or cov[:,dim,dim]
        (If cov missing/invalid: plot mean only, no band.)
    """

    def _as_np(a):
        return None if a is None else np.asarray(a)

    def _clip_T_1d(y, v, T_plot_):
        if T_plot_ is None:
            T_ = len(y)
        else:
            T_ = int(min(len(y), int(T_plot_)))
        y = y[:T_]
        if v is not None:
            v = v[:T_]
        return y, v, T_

    def _tail_slice_1d(y, v, K):
        if K is None:
            return y, v
        K = int(K)
        if K <= 0:
            return y, v
        y2 = y[-K:]
        v2 = None if v is None else v[-K:]
        return y2, v2

    def _safe_std_from_var(v):
        v = np.asarray(v, dtype=float)
        v = np.where(np.isfinite(v), v, np.nan)
        v = np.maximum(v, 0.0)
        return np.sqrt(v)

    def _extract_channel_mean(theta, d: int):
        if theta is None:
            return None
        theta = np.asarray(theta)
        if theta.ndim == 1:
            return theta if d == 0 else None
        if theta.ndim == 2:
            return theta[:, d] if d < theta.shape[1] else None
        return None

    def _extract_channel_var(cov, d: int, T: int):
        if cov is None:
            return None
        cov = np.asarray(cov)

        if cov.ndim == 1:
            return cov[:T] if d == 0 else None
        if cov.ndim == 2:  # (T,D) diagonal variance
            if d >= cov.shape[1]:
                return None
            return cov[:min(T, cov.shape[0]), d]
        if cov.ndim == 3:  # (T,D,D)
            if d >= cov.shape[1] or d >= cov.shape[2]:
                return None
            return cov[:min(T, cov.shape[0]), d, d]
        return None

    for E in E_pi_y_constraints:
        mean = best_dict.get(E, {}).get(key_mean, None)
        cov = best_dict.get(E, {}).get(key_cov, None)
        if mean is None:
            continue  # mean is required

        ratio = float(E) / float(E_pi_x_fix)

        # --------- choose behavior ---------
        if dim is None:
            # legacy (crush to 1D)
            if cov is None:
                continue  # keep legacy semantics: require cov too
            y = to_1d_timeseries(mean)
            v = to_1d_variance(cov)
            y, v, _T = clip_T(y, v, T_plot=T_plot)
            y, v = _tail_slice(y, v, tail_K)
            t = np.arange(len(y))
            std = _safe_std_from_var(v)

            line, = ax.plot(t, y, label=rf'$E_{{\pi_y}}/E_{{\pi_x}} = {ratio:g}$')
            ax.fill_between(t, y - z * std, y + z * std, color=line.get_color(), alpha=0.15)

        else:
            # channel-wise
            d = int(dim)
            mean_np = _as_np(mean)
            cov_np = _as_np(cov)

            y_full = _extract_channel_mean(mean_np, d)
            if y_full is None:
                continue
            y_full = y_full.astype(float, copy=False)
            T_full = len(y_full)

            y, _, T_ = _clip_T_1d(y_full, None, T_plot)

            v_full = _extract_channel_var(cov_np, d, T_full) if cov_np is not None else None
            v = None if v_full is None else np.asarray(v_full, dtype=float)[:T_]

            # tail slice; NOTE: reindex to 0..K-1 (matches your current helper style)
            y, v = _tail_slice_1d(y, v, tail_K)
            t = np.arange(len(y))

            line, = ax.plot(t, y, label=rf'$E_{{\pi_y}}/E_{{\pi_x}} = {ratio:g}$')

            if v is not None:
                std = _safe_std_from_var(v)
                ax.fill_between(t, y - z * std, y + z * std, color=line.get_color(), alpha=0.15)

    ax.grid(True)

    _apply_axes_style(
        ax,
        xlabel=None,
        ylabel=ylabel,
        xlabel_fs=xlabel_fs,
        ylabel_fs=ylabel_fs,
        xtick_fs=xtick_fs,
        ytick_fs=ytick_fs,
    )

    _legend(
        ax,
        loc=legend_loc,
        ncol=legend_ncol,
        legend_fs=legend_fs,
        legend_title=None,
        legend_title_fs=legend_title_fs,
    )

    # x-limits
    if force_xlim_0_10000:
        ax.set_xlim(0, 10000)
    else:
        if ax.lines:
            ax.set_xlim(0, len(ax.lines[0].get_xdata()) - 1)


# ==============================
# Public plotting API (SAVE ONLY)
# ==============================
def save_lambda_with_uncertainty_per_kx(
    best_by_kx: Dict[int, Dict[float, Dict[str, Any]]],
    kx_values: Sequence[int],
    E_pi_y_constraints: Sequence[float],
    E_pi_x_fix: float,
    which: str = "y",
    z: float = 2.0,
    T_plot: Optional[int] = None,
    figsize: Tuple[int, int] = (22, 4),
    legend_ncol: int = 3,
    ylim=None,
    out_dir: Optional[Path] = None,
    filename_prefix: Optional[str] = None,
    dpi: int = 200,
    zoom_K: Optional[int] = None,
    zoom_pad_frac: float = 0.08,
    zoom_use_uncertainty: bool = True,
    # ---------- font sizes ----------
    xlabel_fs: int = 14,
    ylabel_fs: int = 14,
    legend_fs: int = 12,
    xtick_fs: int = 12,
    ytick_fs: int = 12,
    legend_titleitle_fs: int = 12,  # kept for backwards typos
    legend_title_fs: Optional[int] = None,
) -> Dict[int, Path]:
    """
    Updated to avoid any 1D conversions.

    - Auto-detects dimensionality of lambda_{which}:
        * (T,)   -> D=1
        * (T,D)  -> D=D
    - Calls _plot_uncertainty_panel(..., dim=d) so it uses channel-wise plotting
      (no to_1d_timeseries / to_1d_variance).
    - Produces D separate figures per kx:
        base_kx{kx}_d0.pdf, ..., base_kx{kx}_d{D-1}.pdf
      and (optional) zoom versions:
        base_kx{kx}_d0_zoom.pdf, ...
    - Keeps your legend formatting (via _plot_uncertainty_panel).
    """
    if which not in {"x", "y"}:
        raise ValueError("which must be 'x' or 'y'")

    legend_title_fs = legend_title_fs if legend_title_fs is not None else 12

    out = _reports_dir(out_dir)
    saved: Dict[int, Path] = {}

    key_mean = f"lambda_{which}"
    key_cov = f"cov_lambda_{which}"
    base = (f"lambda_{which}" if filename_prefix is None else filename_prefix)

    def _as_np(a):
        return None if a is None else np.asarray(a)

    def _infer_D(best_dict: Dict[float, Dict[str, Any]]) -> int:
        for E in E_pi_y_constraints:
            lam = _as_np(best_dict.get(E, {}).get(key_mean, None))
            if lam is None:
                continue
            if lam.ndim == 1:
                return 1
            if lam.ndim == 2 and lam.shape[1] > 0:
                return int(lam.shape[1])
        return 1

    def _compute_ylim_from_tail(best_dict: Dict[float, Dict[str, Any]], d: int, K: int):
        """
        Replaces the old to_1d_* ylim logic with channel-wise ylim logic.
        Uses:
          y = lambda[:, d] (or lambda[:] if 1D)
          v from cov (T,), (T,D), or (T,D,D) -> diag variance for channel d
        then tail-slices by K (reindexed) and aggregates y +/- z*std if enabled.
        """
        ys = []
        for E in E_pi_y_constraints:
            rec = best_dict.get(E, {})
            mean = _as_np(rec.get(key_mean, None))
            cov = _as_np(rec.get(key_cov, None))
            if mean is None:
                continue

            # mean channel
            if mean.ndim == 1:
                if d != 0:
                    continue
                y_full = mean
            elif mean.ndim == 2:
                if d >= mean.shape[1]:
                    continue
                y_full = mean[:, d]
            else:
                continue

            y_full = y_full.astype(float, copy=False)

            # clip T
            if T_plot is None:
                T_ = len(y_full)
            else:
                T_ = int(min(len(y_full), int(T_plot)))

            y_full = y_full[:T_]

            # variance channel (optional)
            v = None
            if cov is not None:
                if cov.ndim == 1:
                    if d == 0:
                        v = cov[:T_]
                elif cov.ndim == 2:
                    if d < cov.shape[1]:
                        v = cov[:min(T_, cov.shape[0]), d]
                elif cov.ndim == 3:
                    if d < cov.shape[1] and d < cov.shape[2]:
                        v = cov[:min(T_, cov.shape[0]), d, d]

            # tail slice
            K2 = int(min(K, T_))
            if K2 <= 0:
                continue
            y = y_full[-K2:]
            if v is not None:
                v = np.asarray(v, dtype=float)[-K2:]

            if zoom_use_uncertainty and (v is not None):
                std = _safe_std_from_var(v)
                ys.append(y - z * std)
                ys.append(y + z * std)
            else:
                ys.append(y)

        if len(ys) == 0:
            return None

        y_all = np.concatenate([np.asarray(a).ravel() for a in ys])
        y_all = y_all[np.isfinite(y_all)]
        if y_all.size == 0:
            return None

        lo = float(np.min(y_all))
        hi = float(np.max(y_all))
        if hi == lo:
            eps = 1e-6 if lo == 0 else abs(lo) * 1e-3
            lo, hi = lo - eps, hi + eps
        pad = (hi - lo) * float(zoom_pad_frac)
        return (lo - pad, hi + pad)

    for kx in kx_values:
        best_dict = best_by_kx[kx]
        D = _infer_D(best_dict)

        representative_path = None

        # -------- full plots (one per dim; NO 1D conversions) --------
        for d in range(D):
            fig, ax = plt.subplots(figsize=figsize)

            _plot_uncertainty_panel(
                ax=ax,
                best_dict=best_dict,
                E_pi_y_constraints=E_pi_y_constraints,
                E_pi_x_fix=E_pi_x_fix,
                key_mean=key_mean,
                key_cov=key_cov,
                z=z,
                T_plot=T_plot,
                ylabel=rf'$\lambda_{{{which},{d}}}$' if D > 1 else rf'$\lambda_{which}$',
                dim=d,  # <-- critical: channel-wise, no crushing
                legend_ncol=legend_ncol,
                tail_K=None,
                legend_loc="best",
                xlabel_fs=xlabel_fs,
                ylabel_fs=ylabel_fs,
                legend_fs=legend_fs,
                xtick_fs=xtick_fs,
                ytick_fs=ytick_fs,
                legend_title_fs=legend_title_fs,
            )

            if ylim is not None:
                ax.set_ylim(*ylim)

            _apply_axes_style(
                ax,
                xlabel="Time step",
                ylabel=None,
                xlabel_fs=xlabel_fs,
                ylabel_fs=ylabel_fs,
                xtick_fs=xtick_fs,
                ytick_fs=ytick_fs,
            )

            fig.tight_layout()
            filepath = out / f"{base}_kx{kx}_d{d}.pdf"
            p = _save_close(fig, filepath, dpi=dpi)
            if representative_path is None:
                representative_path = p

        if representative_path is not None:
            saved[kx] = representative_path

        # -------- zoom plots (one per dim; NO 1D conversions) --------
        if zoom_K is not None:
            zoom_K_int = int(zoom_K)
            if zoom_K_int <= 0:
                raise ValueError("zoom_K must be a positive integer.")

            for d in range(D):
                figz, axz = plt.subplots(figsize=figsize)

                _plot_uncertainty_panel(
                    ax=axz,
                    best_dict=best_dict,
                    E_pi_y_constraints=E_pi_y_constraints,
                    E_pi_x_fix=E_pi_x_fix,
                    key_mean=key_mean,
                    key_cov=key_cov,
                    z=z,
                    T_plot=T_plot,
                    ylabel=rf'$\lambda_{{{which},{d}}}$' if D > 1 else rf'$\lambda_{which}$',
                    dim=d,  # <-- critical
                    legend_ncol=legend_ncol,
                    tail_K=zoom_K_int,
                    legend_loc="lower left",
                    xlabel_fs=xlabel_fs,
                    ylabel_fs=ylabel_fs,
                    legend_fs=legend_fs,
                    xtick_fs=xtick_fs,
                    ytick_fs=ytick_fs,
                    legend_title_fs=legend_title_fs,
                )

                # Recompute zoom y-lims channel-wise (replacement for old to_1d_* logic)
                if ylim is not None:
                    axz.set_ylim(*ylim)
                else:
                    ylims = _compute_ylim_from_tail(best_dict, d=d, K=zoom_K_int)
                    if ylims is not None:
                        axz.set_ylim(*ylims)

                _apply_axes_style(
                    axz,
                    xlabel=f"Last {zoom_K_int} steps (reindexed)",
                    ylabel=None,
                    xlabel_fs=xlabel_fs,
                    ylabel_fs=ylabel_fs,
                    xtick_fs=xtick_fs,
                    ytick_fs=ytick_fs,
                )

                figz.tight_layout()
                _save_close(figz, out / f"{base}_kx{kx}_d{d}_zoom.pdf", dpi=dpi)

    return saved


def save_theta_with_uncertainty_per_kx(
    best_by_kx: Dict[int, Dict[float, Dict[str, Any]]],
    kx_values: Sequence[int],
    E_pi_y_constraints: Sequence[float],
    E_pi_x_fix: float,
    z: float = 2.0,
    T_plot: Optional[int] = None,
    figsize: Tuple[int, int] = (22, 4),
    legend_ncol: int = 3,
    ylim: Optional[Tuple[float, float]] = None,
    out_dir: Optional[Path] = None,
    filename_prefix: str = "theta",
    dpi: int = 200,
    zoom_K: Optional[int] = None,
    zoom_pad_frac: float = 0.08,
    zoom_use_uncertainty: bool = True,
    # ---------- font sizes ----------
    xlabel_fs: int = 14,
    ylabel_fs: int = 14,
    legend_fs: int = 12,
    xtick_fs: int = 12,
    ytick_fs: int = 12,
    legend_title_fs: int = 12,  # kept for signature compatibility
) -> Dict[int, Path]:
    """
    Updated to call _plot_uncertainty_panel correctly and avoid any 1D conversions.

    Assumes you have updated _plot_uncertainty_panel to support `dim=...` (channel-wise)
    without using to_1d_timeseries/to_1d_variance.

    Behaviour:
      - Auto-detects theta dimensionality:
          * theta shape (T,)   -> D = 1
          * theta shape (T, D) -> D = D
      - Produces D separate figures per kx:
          theta_kx{kx}_d0.pdf, ..., theta_kx{kx}_d{D-1}.pdf
        and (optional) zoom versions:
          theta_kx{kx}_d0_zoom.pdf, ...
      - Keeps your legend format via _plot_uncertainty_panel:
      - Forces x-axis to [0, 10000] for all plots by telling the panel.
    """
    out = _reports_dir(out_dir)
    saved: Dict[int, Path] = {}

    key_mean = "theta"
    key_cov = "cov_theta"

    def _as_np(a):
        return None if a is None else np.asarray(a)

    def _infer_D(best_dict: Dict[float, Dict[str, Any]]) -> int:
        for E in E_pi_y_constraints:
            th = _as_np(best_dict.get(E, {}).get(key_mean, None))
            if th is None:
                continue
            if th.ndim == 1:
                return 1
            if th.ndim == 2 and th.shape[1] > 0:
                return int(th.shape[1])
        return 1

    def _compute_ylim_from_tail(best_dict: Dict[float, Dict[str, Any]], d: int, K: int):
        """
        Channel-wise replacement for your old zoom ylim logic (no 1D conversions).
        Uses:
          y = theta[:, d] (or theta[:] if 1D)
          v from cov (T,), (T,D), or (T,D,D) -> diag variance for channel d
        then tail-slices by K (reindexed) and aggregates y +/- z*std if enabled.
        """
        ys = []
        for E in E_pi_y_constraints:
            rec = best_dict.get(E, {})
            mean = _as_np(rec.get(key_mean, None))
            cov = _as_np(rec.get(key_cov, None))
            if mean is None:
                continue

            # mean channel
            if mean.ndim == 1:
                if d != 0:
                    continue
                y_full = mean
            elif mean.ndim == 2:
                if d >= mean.shape[1]:
                    continue
                y_full = mean[:, d]
            else:
                continue

            y_full = y_full.astype(float, copy=False)

            # clip T
            if T_plot is None:
                T_ = len(y_full)
            else:
                T_ = int(min(len(y_full), int(T_plot)))
            y_full = y_full[:T_]

            # variance channel (optional)
            v = None
            if cov is not None:
                if cov.ndim == 1:
                    if d == 0:
                        v = cov[:T_]
                elif cov.ndim == 2:
                    if d < cov.shape[1]:
                        v = cov[:min(T_, cov.shape[0]), d]
                elif cov.ndim == 3:
                    if d < cov.shape[1] and d < cov.shape[2]:
                        v = cov[:min(T_, cov.shape[0]), d, d]

            # tail slice
            K2 = int(min(K, T_))
            if K2 <= 0:
                continue
            y = y_full[-K2:]
            if v is not None:
                v = np.asarray(v, dtype=float)[-K2:]

            if zoom_use_uncertainty and (v is not None):
                std = _safe_std_from_var(v)
                ys.append(y - z * std)
                ys.append(y + z * std)
            else:
                ys.append(y)

        if len(ys) == 0:
            return None

        y_all = np.concatenate([np.asarray(a).ravel() for a in ys])
        y_all = y_all[np.isfinite(y_all)]
        if y_all.size == 0:
            return None

        lo = float(np.min(y_all))
        hi = float(np.max(y_all))
        if hi == lo:
            eps = 1e-6 if lo == 0 else abs(lo) * 1e-3
            lo, hi = lo - eps, hi + eps
        pad = (hi - lo) * float(zoom_pad_frac)
        return (lo - pad, hi + pad)

    for kx in kx_values:
        best_dict = best_by_kx[kx]
        D = _infer_D(best_dict)
        representative_path = None

        # -------- FULL plots (one per dim; NO 1D conversions) --------
        for d in range(D):
            fig, ax = plt.subplots(figsize=figsize)

            _plot_uncertainty_panel(
                ax=ax,
                best_dict=best_dict,
                E_pi_y_constraints=E_pi_y_constraints,
                E_pi_x_fix=E_pi_x_fix,
                key_mean=key_mean,
                key_cov=key_cov,
                z=z,
                T_plot=T_plot,
                ylabel=rf"$\theta_{{{d}}}$" if D > 1 else r"$\theta$",
                dim=d,  # <-- critical: channel-wise in the panel
                force_xlim_0_10000=True,  # <-- critical: enforce x-axis range
                legend_ncol=legend_ncol,
                tail_K=None,
                legend_loc="best",
                xlabel_fs=xlabel_fs,
                ylabel_fs=ylabel_fs,
                legend_fs=legend_fs,
                xtick_fs=xtick_fs,
                ytick_fs=ytick_fs,
                legend_title_fs=legend_title_fs,
            )

            # (panel already applied axes style; we only enforce ylim if requested)
            if ylim is not None:
                ax.set_ylim(*ylim)

            # Ensure x-label is present (your panel uses xlabel=None inside _apply_axes_style)
            _apply_axes_style(
                ax,
                xlabel="Time step",
                ylabel=None,
                xlabel_fs=xlabel_fs,
                ylabel_fs=ylabel_fs,
                xtick_fs=xtick_fs,
                ytick_fs=ytick_fs,
            )

            fig.tight_layout()
            filepath = out / f"{filename_prefix}_kx{kx}_d{d}.pdf"
            p = _save_close(fig, filepath, dpi=dpi)
            if representative_path is None:
                representative_path = p

        if representative_path is not None:
            saved[kx] = representative_path

        # -------- ZOOM plots (one per dim; NO 1D conversions) --------
        if zoom_K is not None:
            zoom_K_int = int(zoom_K)
            if zoom_K_int <= 0:
                raise ValueError("zoom_K must be a positive integer.")

            for d in range(D):
                figz, axz = plt.subplots(figsize=figsize)

                _plot_uncertainty_panel(
                    ax=axz,
                    best_dict=best_dict,
                    E_pi_y_constraints=E_pi_y_constraints,
                    E_pi_x_fix=E_pi_x_fix,
                    key_mean=key_mean,
                    key_cov=key_cov,
                    z=z,
                    T_plot=T_plot,
                    ylabel=rf"$\theta_{{{d}}}$" if D > 1 else r"$\theta$",
                    dim=d,
                    force_xlim_0_10000=True,
                    legend_ncol=legend_ncol,
                    tail_K=zoom_K_int,
                    legend_loc="lower left",
                    xlabel_fs=xlabel_fs,
                    ylabel_fs=ylabel_fs,
                    legend_fs=legend_fs,
                    xtick_fs=xtick_fs,
                    ytick_fs=ytick_fs,
                    legend_title_fs=legend_title_fs,
                )

                if ylim is not None:
                    axz.set_ylim(*ylim)
                else:
                    ylims = _compute_ylim_from_tail(best_dict, d=d, K=zoom_K_int)
                    if ylims is not None:
                        axz.set_ylim(*ylims)

                _apply_axes_style(
                    axz,
                    xlabel=f"Last {zoom_K_int} steps (reindexed)",
                    ylabel=None,
                    xlabel_fs=xlabel_fs,
                    ylabel_fs=ylabel_fs,
                    xtick_fs=xtick_fs,
                    ytick_fs=ytick_fs,
                )

                figz.tight_layout()
                _save_close(figz, out / f"{filename_prefix}_kx{kx}_d{d}_zoom.pdf", dpi=dpi)

    return saved



def save_mse_vs_ratio(
    best_by_kx: Dict[int, Dict[float, Dict[str, Any]]],
    kx_values: Sequence[int],
    E_pi_y_constraints: Sequence[float],
    E_pi_x_fix: float,
    out_dir: Optional[Path] = None,
    filename: str = "mse_vs_ratio.pdf",
    dpi: int = 200,
    # ---------- font sizes ----------
    xlabel_fs: int = 14,
    ylabel_fs: int = 14,
    legend_fs: int = 12,
    xtick_fs: int = 12,
    ytick_fs: int = 12,
    legend_title_fs: int = 12,
    # ---------- NEW: y-limits ----------
    ylim: Optional[Tuple[float, float]] = None,
) -> Path:
    out = _reports_dir(out_dir)

    positions = np.arange(len(E_pi_y_constraints))
    x_labels = [f"{(E / E_pi_x_fix):g}" for E in E_pi_y_constraints]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    markers = {2: "o", 3: "s", 4: "D"}
    linestyles = {2: "-", 3: "--", 4: ":"}

    for kx in kx_values:
        mse_vals: List[float] = []
        for E in E_pi_y_constraints:
            mse = best_by_kx.get(kx, {}).get(E, {}).get("mse", None)
            mse_vals.append(float(mse) if mse is not None and np.isfinite(mse) else np.nan)

        ax.plot(
            positions,
            mse_vals,
            marker=markers.get(kx, "o"),
            linestyle=linestyles.get(kx, "-"),
            label=fr"$k_x = {kx}$",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=xtick_fs)
    ax.grid(True, axis="y")

    _apply_axes_style(
        ax,
        xlabel=r"$E_{\pi_y} / E_{\pi_x}$",
        ylabel=r"$\mathrm{MSE}(x,\hat{x})$",
        xlabel_fs=xlabel_fs,
        ylabel_fs=ylabel_fs,
        xtick_fs=xtick_fs,
        ytick_fs=ytick_fs,
    )

    _legend(
        ax,
        loc="best",
        legend_fs=legend_fs,
        legend_title=None,
        legend_title_fs=legend_title_fs,
    )

    # ---------- NEW ----------
    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.tight_layout()

    filename = _slug_filename(filename)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return _save_close(fig, out / filename, dpi=dpi)


def save_fa_decomposition(
    best_by_kx: Dict[int, Dict[float, Dict[str, Any]]],
    kx_values: Sequence[int],
    E_pi_y_constraints: Sequence[float],
    E_pi_x_fix: float,
    out_dir: Optional[Path] = None,
    filename: str = "fa_decomposition.pdf",
    dpi: int = 200,
    stored_accuracy_is_negative: bool = False,
    # ---------- font sizes ----------
    xlabel_fs: int = 14,
    ylabel_fs: int = 14,
    legend_fs: int = 12,
    xtick_fs: int = 12,
    ytick_fs: int = 12,
    legend_title_fs: int = 12,
    # ---------- NEW ----------
    ylim: Optional[Tuple[float, float]] = None,
) -> Path:
    out = _reports_dir(out_dir)

    positions = np.arange(len(E_pi_y_constraints))
    x_labels = [f"{(E / E_pi_x_fix):g}" for E in E_pi_y_constraints]

    fig, ax = plt.subplots(figsize=(14, 4.8))

    linestyles = {2: "-", 3: "--", 4: ":"}

    # color encodes metric (same color across kx)
    metric_style = {
        "fa":   {"marker": "o", "label": "FA", "color": "#1f77b4"},
        "acc":  {"marker": "^", "label": "Acc.", "color": "#ff7f0e"},
        "comp": {"marker": "s", "label": "Comp.", "color": "#2ca02c"},
    }

    for kx in kx_values:
        ls = linestyles.get(kx, "-")

        fa_vals: List[float] = []
        acc_vals: List[float] = []
        comp_vals: List[float] = []

        for E in E_pi_y_constraints:
            d = best_by_kx.get(kx, {}).get(E, {})

            fa = d.get("min_fa", None)
            acc = d.get("accuracy", None)
            comp = d.get("complexity", None)

            fa_vals.append(float(fa) if fa is not None and np.isfinite(fa) else np.nan)

            if acc is not None and np.isfinite(acc):
                a = float(acc)
                if stored_accuracy_is_negative:
                    a = -a
                acc_vals.append(a)
            else:
                acc_vals.append(np.nan)

            comp_vals.append(float(comp) if comp is not None and np.isfinite(comp) else np.nan)

        ax.plot(
            positions, fa_vals,
            linestyle=ls, marker=metric_style["fa"]["marker"],
            color=metric_style["fa"]["color"],
            label=f"FA (kx={kx})",
        )
        ax.plot(
            positions, acc_vals,
            linestyle=ls, marker=metric_style["acc"]["marker"],
            color=metric_style["acc"]["color"],
            label=f"Accuracy (kx={kx})",
        )
        ax.plot(
            positions, comp_vals,
            linestyle=ls, marker=metric_style["comp"]["marker"],
            color=metric_style["comp"]["color"],
            label=f"Complexity (kx={kx})",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=xtick_fs)
    ax.grid(True, axis="y")

    _apply_axes_style(
        ax,
        xlabel=r"$E_{\pi_y} / E_{\pi_x}$",
        ylabel="Value",
        xlabel_fs=xlabel_fs,
        ylabel_fs=ylabel_fs,
        xtick_fs=xtick_fs,
        ytick_fs=ytick_fs,
    )

    # ---------- NEW ----------
    if ylim is not None:
        ax.set_ylim(*ylim)

    # ---------- SINGLE LEGEND (top-left) ----------
    # Build compact handles that explain *encodings*:
    # - marker+color => metric
    # - linestyle => kx
    metric_handles = [
        Line2D(
            [0], [0],
            marker=metric_style["fa"]["marker"],
            color=metric_style["fa"]["color"],
            linestyle="None",
            label=metric_style["fa"]["label"],
        ),
        Line2D(
            [0], [0],
            marker=metric_style["acc"]["marker"],
            color=metric_style["acc"]["color"],
            linestyle="None",
            label=metric_style["acc"]["label"],
        ),
        Line2D(
            [0], [0],
            marker=metric_style["comp"]["marker"],
            color=metric_style["comp"]["color"],
            linestyle="None",
            label=metric_style["comp"]["label"],
        ),
    ]

    kx_handles = [
        Line2D(
            [0], [0],
            linestyle=linestyles.get(kx, "-"),
            color="k",
            label=fr"$k_x = {kx}$",
        )
        for kx in kx_values
    ]

    # Optional: add section headers inside ONE legend (still a single legend object)
    combined_handles = metric_handles + kx_handles

    _legend(
        ax,
        handles=combined_handles,
        loc="upper left",
        legend_fs=legend_fs,
        legend_title=None,          # keep everything inside one box
        legend_title_fs=legend_title_fs,
    )

    fig.tight_layout()

    filename = _slug_filename(filename)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return _save_close(fig, out / filename, dpi=dpi)


def save_kappa_best(
    best_by_kx: Dict[int, Dict[float, Dict[str, Any]]],
    kx_values: Sequence[int],
    E_pi_y_constraints: Sequence[float],
    E_pi_x_fix: float,
    out_dir: Optional[Path] = None,
    filename: str = "kappa_best.pdf",
    dpi: int = 200,
    # ---------- font sizes ----------
    xlabel_fs: int = 14,
    ylabel_fs: int = 14,
    legend_fs: int = 12,
    xtick_fs: int = 12,
    ytick_fs: int = 12,
    legend_title_fs: int = 12,
) -> Path:
    out = _reports_dir(out_dir)

    positions = np.arange(len(E_pi_y_constraints))
    x_labels = [f"{(E / E_pi_x_fix):g}" for E in E_pi_y_constraints]

    fig, ax = plt.subplots(figsize=(14, 4))

    markers = {2: "o", 3: "s", 4: "D"}
    linestyles = {2: "-", 3: "--", 4: ":"}

    for kx in kx_values:
        best_vals: List[float] = []
        for E in E_pi_y_constraints:
            b = best_by_kx.get(kx, {}).get(E, {}).get("kappa_x", None)
            best_vals.append(float(b) if b is not None and np.isfinite(b) else np.nan)

        ax.plot(
            positions,
            best_vals,
            marker=markers.get(kx, "o"),
            linestyle=linestyles.get(kx, "-"),
            label=fr"$k_x = {kx}$",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=xtick_fs)
    ax.grid(True, axis="y")

    _apply_axes_style(
        ax,
        xlabel=r"$E_{\pi_y} / E_{\pi_x}$",
        ylabel=r"$\kappa_x$",
        xlabel_fs=xlabel_fs,
        ylabel_fs=ylabel_fs,
        xtick_fs=xtick_fs,
        ytick_fs=ytick_fs,
    )
    _legend(ax, loc="best", legend_fs=legend_fs, legend_title=None, legend_title_fs=legend_title_fs)

    fig.tight_layout()

    filename = _slug_filename(filename)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return _save_close(fig, out / filename, dpi=dpi)


def save_theta_interval_best(
    best_by_kx: Dict[int, Dict[float, Dict[str, Any]]],
    kx_values: Sequence[int],
    E_pi_y_constraints: Sequence[float],
    E_pi_x_fix: float,
    out_dir: Optional[Path] = None,
    filename: str = "theta_interval_best.pdf",
    dpi: int = 200,
    # ---------- font sizes ----------
    xlabel_fs: int = 14,
    ylabel_fs: int = 14,
    legend_fs: int = 12,
    xtick_fs: int = 12,
    ytick_fs: int = 12,
    legend_title_fs: int = 12,
) -> Path:
    out = _reports_dir(out_dir)

    positions = np.arange(len(E_pi_y_constraints))
    x_labels = [f"{(E / E_pi_x_fix):g}" for E in E_pi_y_constraints]

    fig, ax = plt.subplots(figsize=(14, 4))

    markers = {2: "o", 3: "s", 4: "D"}
    linestyles = {2: "-", 3: "--", 4: ":"}

    for kx in kx_values:
        vals: List[float] = []
        for E in E_pi_y_constraints:
            ti = best_by_kx.get(kx, {}).get(E, {}).get("theta_interval", None)
            if ti is None:
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(ti))
                except (TypeError, ValueError):
                    vals.append(np.nan)

        ax.plot(
            positions,
            vals,
            marker=markers.get(kx, "o"),
            linestyle=linestyles.get(kx, "-"),
            label=fr"$k_x = {kx}$",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=xtick_fs)
    ax.grid(True, axis="y")

    _apply_axes_style(
        ax,
        xlabel=r"$E_{\pi_y} / E_{\pi_x}$",
        ylabel=r"$\theta_{\mathrm{interval}}$",
        xlabel_fs=xlabel_fs,
        ylabel_fs=ylabel_fs,
        xtick_fs=xtick_fs,
        ytick_fs=ytick_fs,
    )
    _legend(ax, loc="best", legend_fs=legend_fs, legend_title=None, legend_title_fs=legend_title_fs)

    fig.tight_layout()

    filename = _slug_filename(filename)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return _save_close(fig, out / filename, dpi=dpi)


def save_theta_beta_best(
    best_by_kx: Dict[int, Dict[float, Dict[str, Any]]],
    kx_values: Sequence[int],
    E_pi_y_constraints: Sequence[float],
    E_pi_x_fix: float,
    out_dir: Optional[Path] = None,
    filename: str = "theta_beta_best.pdf",
    dpi: int = 200,
    # ---------- font sizes ----------
    xlabel_fs: int = 14,
    ylabel_fs: int = 14,
    legend_fs: int = 12,
    xtick_fs: int = 12,
    ytick_fs: int = 12,
    legend_title_fs: int = 12,
) -> Path:
    out = _reports_dir(out_dir)

    positions = np.arange(len(E_pi_y_constraints))
    x_labels = [f"{(E / E_pi_x_fix):g}" for E in E_pi_y_constraints]

    fig, ax = plt.subplots(figsize=(14, 4))

    markers = {2: "o", 3: "s", 4: "D"}
    linestyles = {2: "-", 3: "--", 4: ":"}

    for kx in kx_values:
        vals: List[float] = []
        for E in E_pi_y_constraints:
            tb = best_by_kx.get(kx, {}).get(E, {}).get("theta_beta", None)
            if tb is None:
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(tb))
                except (TypeError, ValueError):
                    vals.append(np.nan)

        ax.plot(
            positions,
            vals,
            marker=markers.get(kx, "o"),
            linestyle=linestyles.get(kx, "-"),
            label=fr"$k_x = {kx}$",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=xtick_fs)
    ax.grid(True, axis="y")

    _apply_axes_style(
        ax,
        xlabel=r"$E_{\pi_y} / E_{\pi_x}$",
        ylabel=r"$\theta_{\beta}$",
        xlabel_fs=xlabel_fs,
        ylabel_fs=ylabel_fs,
        xtick_fs=xtick_fs,
        ytick_fs=ytick_fs,
    )
    _legend(ax, loc="best", legend_fs=legend_fs, legend_title=None, legend_title_fs=legend_title_fs)

    fig.tight_layout()

    filename = _slug_filename(filename)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return _save_close(fig, out / filename, dpi=dpi)


def save_lambda_beta_best(
    best_by_kx: Dict[int, Dict[float, Dict[str, Any]]],
    kx_values: Sequence[int],
    E_pi_y_constraints: Sequence[float],
    E_pi_x_fix: float,
    out_dir: Optional[Path] = None,
    filename: str = "lambda_beta_best.pdf",
    dpi: int = 200,
    # ---------- font sizes ----------
    xlabel_fs: int = 14,
    ylabel_fs: int = 14,
    legend_fs: int = 12,
    xtick_fs: int = 12,
    ytick_fs: int = 12,
    legend_title_fs: int = 12,
) -> Path:
    out = _reports_dir(out_dir)

    positions = np.arange(len(E_pi_y_constraints))
    x_labels = [f"{(E / E_pi_x_fix):g}" for E in E_pi_y_constraints]

    fig, ax = plt.subplots(figsize=(14, 4))

    markers = {2: "o", 3: "s", 4: "D"}
    linestyles = {2: "-", 3: "--", 4: ":"}

    for kx in kx_values:
        vals: List[float] = []
        for E in E_pi_y_constraints:
            lb = best_by_kx.get(kx, {}).get(E, {}).get("lambda_beta", None)
            if lb is None:
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(lb))
                except (TypeError, ValueError):
                    vals.append(np.nan)

        ax.plot(
            positions,
            vals,
            marker=markers.get(kx, "o"),
            linestyle=linestyles.get(kx, "-"),
            label=fr"$k_x = {kx}$",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=xtick_fs)
    ax.grid(True, axis="y")

    _apply_axes_style(
        ax,
        xlabel=r"$E_{\pi_y} / E_{\pi_x}$",
        ylabel=r"$\lambda_{\beta}$",
        xlabel_fs=xlabel_fs,
        ylabel_fs=ylabel_fs,
        xtick_fs=xtick_fs,
        ytick_fs=ytick_fs,
    )
    _legend(ax, loc="best", legend_fs=legend_fs, legend_title=None, legend_title_fs=legend_title_fs)

    fig.tight_layout()

    filename = _slug_filename(filename)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return _save_close(fig, out / filename, dpi=dpi)
