import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import json
from functions.plotting import delta_method_exp, bpa  # (kept as-is, not used here but preserved)
from functions import collect_and_sort
class PlotDispatcher:
    # -------------------------
    # Constructor & IO
    # -------------------------
    def __init__(self,data_dir):

        self.figsize = (30, 9)
        self.dpi = 100
        self.legend_size = 22
        self.tick_size = 22

        self.base_colors = colormaps['tab20'].colors
        self.color_cycle = self.base_colors[::2]  # even indices

        self.data_dir = data_dir
        self.output_dir = os.path.join(data_dir, "plots")
        os.makedirs(self.output_dir, exist_ok=True)

        # Load snapshot
        snap_fp = os.path.join(self.data_dir, 'snapshot.json')
        with open(snap_fp, 'r') as f:
            self.snapshot = json.load(f)

        # Load .npy files as attributes
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".npy"):
                base_name = os.path.splitext(file_name)[0]
                data = np.load(os.path.join(self.data_dir, file_name))
                setattr(self, base_name, data)

        # Convert context lengths to boundary indices (as before)
        self.y_context_lengths = np.cumsum(self.y_context_lengths) - 1

    # -------------------------
    # Public entry
    # -------------------------
    def dispatch(self):
        self.plot_pi(key='x')
        self.plot_pi(key='y')
        self.plot_vfe()
        self.plot_accuracy_complexity_tradeoff()
        self.plot_lambda(key='x')
        self.plot_lambda(key='y')
        self.plot_sensations_predictions()
        self.plot_state_estimation()
        self.plot_theta()
        self.plot_white_noise(key='x')
        self.plot_white_noise(key='y')
        self.plot_colored_noise(key='x')
        self.plot_colored_noise(key='y')
        self.plot_sigma_schedule(key='x')
        self.plot_sigma_schedule(key='y')

        # Keep jitter plotting hooks unchanged / optional
        # self.plot_jitters(self.ozaki_jitters, jitter_name='ozaki_jitters')
        # self.plot_jitters(self.hessian_jitters, jitter_name='posterior_hessian_jitters')

    # -------------------------
    # Helpers (DECLUTTER!)
    # -------------------------
    def _fig_ax(self):
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111)
        return fig, ax

    def _apply_common(
        self, ax,
        legend_loc='upper left',
        show_legend=True,
        xlim_left=0,
        ylim_bottom=None,
        xticksize=None,
        yticksize=None,
        grid=True
    ):
        if grid:
            ax.grid(True)
        if xlim_left is not None:
            ax.set_xlim(left=xlim_left)
        if ylim_bottom is not None:
            ax.set_ylim(bottom=ylim_bottom)

        ax.tick_params(axis='x', labelsize=xticksize or self.tick_size)
        ax.tick_params(axis='y', labelsize=yticksize or self.tick_size)
        if show_legend:
            ax.legend(loc=legend_loc, prop={'size': self.legend_size})

    def _save(self, fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, name), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _colors(self, n):
        return [self.color_cycle[i % len(self.color_cycle)] for i in range(n)]

    def _plot_ci(self, ax, x, mean, std, color, alpha=0.3):
        ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color)

    def _vlines_context(self, ax, indices, color='k', linestyle='--', linewidth=1):
        for idx in range(len(indices) - 1):
            ax.axvline(x=indices[idx], color=color, linestyle=linestyle, linewidth=linewidth)

    def _free_action(self, vfe):
        cumsum = []
        running = vfe[0]
        for i, v in enumerate(vfe):
            if i == 0:
                cumsum.append(running)
            else:
                running += v
                cumsum.append(running)
        return cumsum

    # -------------------------
    # Plots
    # -------------------------
    def plot_vfe(self):
        # VFE
        fig, ax = self._fig_ax()
        ax.plot(self.vfe, c='#a65628', label='VFE')
        self._vlines_context(ax, self.y_context_lengths)
        ax.set_ylim(bottom=0)
        self._apply_common(ax, legend_loc='upper left')
        self._save(fig, 'vfe.pdf')

        # Free Action
        fa = self._free_action(self.vfe)
        fig, ax = self._fig_ax()
        ax.plot(fa, c='#a65628', label='Free Action')
        self._vlines_context(ax, self.y_context_lengths)
        ax.set_ylim(bottom=0)
        self._apply_common(ax, legend_loc='upper left')
        self._save(fig, 'fa.pdf')

    def plot_accuracy_complexity_tradeoff(self):
        """
        Plot how much of the VFE comes from Accuracy vs Complexity over time.
        Uses the fractions:
            phi_C = Complexity / (Accuracy + Complexity)
            phi_A = Accuracy / (Accuracy + Complexity)
        both in [0, 1].
        """

        acc = np.asarray(self.accuracy, dtype=float)
        comp = np.asarray(self.complexity, dtype=float)

        # VFE per time step
        vfe = acc + comp

        # small epsilon to avoid division by zero
        eps = 1e-12
        denom = vfe + eps

        frac_complexity = comp / denom
        frac_accuracy = acc / denom  # = 1 - frac_complexity, but we compute explicitly for robustness

        # --- Plot ---
        fig, ax = self._fig_ax()

        ax.plot(
            frac_accuracy,
            label=r'$\phi_A(t) = \frac{A_t}{A_t + C_t}$',
            linewidth=2
        )

        ax.plot(
            frac_complexity,
            label=r'$\phi_C(t) = \frac{C_t}{A_t + C_t}$',
            linewidth=2,
            linestyle='--'
        )

        # same context markers as before
        self._vlines_context(ax, self.y_context_lengths)

        ax.set_ylim(0.0, 1.0)

        # Optional: nicer y-ticks
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        self._apply_common(ax, legend_loc='upper right')
        self._save(fig, 'accuracy_complexity_tradeoff.pdf')

    def plot_lambda(self, key):
        if key == 'x':
            Lambda = self.lambda_x
            Cov_lambda = self.cov_lambda_x
        elif key == 'y':
            Lambda = self.lambda_y
            Cov_lambda = self.cov_lambda_y
        else:
            raise ValueError("key must be 'x' or 'y'.")

        T, d = Lambda.shape
        time = np.arange(T)
        names = [f"$\\lambda_{{{key}}}^{{{i}}}$" for i in range(d)]
        colors = self._colors(d)

        # diag std per dim
        STD = np.sqrt(np.maximum(np.stack([Cov_lambda[:, i, i] for i in range(d)], axis=0), 0))

        fig, ax = self._fig_ax()
        for i in range(d):
            self._plot_ci(ax, time, Lambda[:, i], STD[i], color=colors[i], alpha=0.3)
            ax.plot(time, Lambda[:, i], color=colors[i], label=names[i])

        self._apply_common(ax, legend_loc='upper left', ylim_bottom=None)
        self._save(fig, f'lambda_{key}.pdf')

    def plot_pi(self, key):
        # E[exp(lambda)] via delta method + CI
        if key == 'x':
            Lambda = self.lambda_x
            Cov_lambda = self.cov_lambda_x
            pi_schedule = 1 / (self.x_sigma_schedule ** 2)
        elif key == 'y':
            Lambda = self.lambda_y
            Cov_lambda = self.cov_lambda_y
            pi_schedule = 1 / (self.y_sigma_schedule ** 2)
        else:
            raise ValueError("key must be 'x' or 'y'.")

        T, d = Lambda.shape
        time = np.arange(T)
        names = [f"$\\pi_{{{key}}}^{{{i}}}$" for i in range(d)]
        colors = self._colors(d)

        pi_eta_all = []
        pi_std_all = []
        for t in range(T):
            mu_t = Lambda[t]
            cov_t = Cov_lambda[t]
            pi_eta_t, pi_cov_t = delta_method_exp.compute(mu_t, cov_t)
            pi_eta_all.append(pi_eta_t)
            pi_std_all.append(np.sqrt(np.maximum(np.diag(pi_cov_t), 1e-12)))
        pi_eta_all = np.asarray(pi_eta_all)
        pi_std_all = np.asarray(pi_std_all)

        fig, ax = self._fig_ax()
        for i in range(d):
            self._plot_ci(ax, time, pi_eta_all[:, i], pi_std_all[:, i], color=colors[i], alpha=0.3)
            ax.plot(time, pi_eta_all[:, i], color=colors[i], label=names[i])

        # Ground-truth precision (kept as before)
        # ax.plot(pi_schedule, color='k', linestyle='--', linewidth=2, label='Ground truth')

        #ax.set_ylim(bottom=0)
        self._apply_common(ax, legend_loc='upper left')
        self._save(fig, f'pi_{key}.pdf')

    def plot_sigma_schedule(self, key):
        if key == 'y':
            sigma_schedule = self.y_sigma_schedule
            name = 'y_sigma_schedule.pdf'
        elif key == 'x':
            sigma_schedule = self.x_sigma_schedule
            name = 'x_sigma_schedule.pdf'
        else:
            raise ValueError("key must be 'x' or 'y'.")

        # sigma schedule
        fig, ax = self._fig_ax()
        ax.plot(sigma_schedule, c='#a65628', label=f'{key} sigma schedule')
        ax.set_ylim(bottom=0)
        self._apply_common(ax, legend_loc='upper left')
        self._save(fig, name)

        # free action (same filename as your original code)
        fa = self._free_action(self.vfe)
        fig, ax = self._fig_ax()
        ax.plot(fa, c='#a65628', label='Free Action')
        ax.set_ylim(bottom=0)
        self._apply_common(ax, legend_loc='upper left')
        self._save(fig, 'fa.pdf')

    def plot_colored_noise(self, key):
        if key == 'y':
            noise = self.y_colored_noise
            name = 'y_colored_noise.pdf'
        elif key == 'x':
            noise = self.x_colored_noise
            name = 'x_colored_noise.pdf'
        else:
            raise ValueError("key must be 'x' or 'y'.")

        d = noise.shape[0]
        names = [f"$cn_{{{i}}}$" for i in range(d)]
        colors = self._colors(d)

        fig, ax = self._fig_ax()
        for i in range(d):
            ax.plot(noise[i, :], color=colors[i], label=names[i])

        self._apply_common(ax, legend_loc='lower left')
        self._save(fig, name)

    def plot_white_noise(self, key):
        if key == 'y':
            noise = self.y_white_noise
            name = 'y_white_noise.pdf'
        elif key == 'x':
            noise = self.x_white_noise
            name = 'x_white_noise.pdf'
        else:
            raise ValueError("key must be 'x' or 'y'.")

        d = noise.shape[0]
        names = [f"$wn_{{{i}}}$" for i in range(d)]
        colors = self._colors(d)

        fig, ax = self._fig_ax()
        for i in range(d):
            ax.plot(noise[i, :], color=colors[i], label=names[i])

        self._apply_common(ax, legend_loc='lower left')
        self._save(fig, name)

    def plot_sensations_predictions(self):
        ky, dy = self.gen_sensations.shape[1], self.gen_sensations.shape[2]

        titles = []
        t = 'y'
        for i in range(ky):
            titles.append(t)
            t = t + '`'

        if ky == 1:
            gen_sensations = np.squeeze(self.gen_sensations, axis=1)
            gen_predictions = np.squeeze(self.gen_predictions, axis=1)
            for j in range(dy):
                fig, ax = self._fig_ax()
                ax.plot(gen_sensations[:, j], c='#009E73', label=f'{titles[0]}[{j}]')
                ax.plot(gen_predictions[:, j], c='#CC79A7', alpha=0.80, label=f'{titles[0]}hat[{j}]')
                #ax.set_ylim(-0.5, 5.5)
                self._apply_common(ax, legend_loc='upper left')
                self._save(fig, f'{titles[0]}[{j}].pdf')
        else:
            for i in range(ky):
                for j in range(dy):
                    fig, ax = self._fig_ax()
                    ax.plot(self.gen_sensations[:, i, j], c='#009E73', label=f'{titles[i]}[{j}]')
                    ax.plot(self.gen_predictions[:, i, j], c='#CC79A7', alpha=0.80, label=f'{titles[i]}hat[{j}]')
                    #ax.set_ylim(-0.5, 5.5)
                    self._apply_common(ax, legend_loc='upper left')
                    self._save(fig, f'{titles[i]}[{j}].pdf')

    # def plot_state_estimation(self):
    #     gp_name = self.snapshot['gp']['name']
    #     kx, dx = self.gen_x_estimates.shape[1], self.gen_x_estimates.shape[2]
    #     dy = self.y.shape[1]
    #     dt = self.snapshot['gp']['dt']
    #     colors = ['#ff7f00', '#984ea3', '#377eb8']
    #
    #     is_lorenz_3d = (gp_name == 'lorenz' and dx == 3)
    #
    #     # =========================
    #     # 1) Observations y
    #     # =========================
    #     if is_lorenz_3d and dy == 3:
    #         # 3D plot for y
    #         fig = plt.figure()
    #         fig.set_size_inches(10, 7)
    #         ax = fig.add_subplot(111, projection='3d')
    #
    #         ax.plot(
    #             self.y[:, 0],
    #             self.y[:, 1],
    #             self.y[:, 2],
    #             label='y'
    #         )
    #
    #         self._apply_common(ax, legend_loc='upper left', xlim_left=None)
    #         ax.set_xlabel('y[0]')
    #         ax.set_ylabel('y[1]')
    #         ax.set_zlabel('y[2]')
    #
    #         self._save(fig, 'y.pdf')
    #     else:
    #         # Original 2D behaviour
    #         fig, ax = self._fig_ax()
    #         for idx in range(dy):
    #             ax.plot(self.y[:, idx], c=colors[idx % len(colors)], label=f'y[{idx}]')
    #         self._apply_common(ax, legend_loc='upper left')
    #         self._save(fig, 'y.pdf')
    #
    #     # =========================
    #     # 2) Noisy states x_noisy
    #     # =========================
    #     if is_lorenz_3d:
    #         fig = plt.figure()
    #         fig.set_size_inches(10, 7)
    #         ax = fig.add_subplot(111, projection='3d')
    #
    #         ax.plot(
    #             self.x_noisy[:, 0],
    #             self.x_noisy[:, 1],
    #             self.x_noisy[:, 2],
    #             label='x noisy'
    #         )
    #
    #         self._apply_common(ax, legend_loc='upper left', xlim_left=None)
    #         ax.set_xlabel('x[0]')
    #         ax.set_ylabel('x[1]')
    #         ax.set_zlabel('x[2]')
    #
    #         self._save(fig, 'x_noisy.pdf')
    #     else:
    #         fig, ax = self._fig_ax()
    #         for idx in range(dx):
    #             ax.plot(self.x_noisy[:, idx], colors[idx % len(colors)], label=f'x[{idx}]')
    #         self._apply_common(ax, legend_loc='upper left')
    #         self._save(fig, 'x_noisy.pdf')
    #
    #     # =========================
    #     # 3) Build true generalized coordinates from x_clean
    #     # =========================
    #     true_gen_x = [self.x_clean]
    #     for _ in range(kx - 1):
    #         true_gen_x.append(np.diff(true_gen_x[-1], axis=0, prepend=0) / dt)
    #     true_gen_x = np.array(true_gen_x)
    #
    #     # Titles: x, x`, x``, ...
    #     titles = []
    #     t = 'x'
    #     for i in range(kx):
    #         titles.append(t)
    #         t = t + '`'
    #
    #     # =========================
    #     # 4) Per generalized coordinate
    #     # =========================
    #     for coord in range(kx):
    #         # -------- true --------
    #         if is_lorenz_3d:
    #             fig = plt.figure()
    #             fig.set_size_inches(10, 7)
    #             ax = fig.add_subplot(111, projection='3d')
    #
    #             ax.plot(
    #                 true_gen_x[coord][:, 0],
    #                 true_gen_x[coord][:, 1],
    #                 true_gen_x[coord][:, 2],
    #                 label=f'{titles[coord]} (true)'
    #             )
    #
    #             self._apply_common(ax, legend_loc='upper left', xlim_left=None)
    #             ax.set_xlabel(f'{titles[coord]}[0]')
    #             ax.set_ylabel(f'{titles[coord]}[1]')
    #             ax.set_zlabel(f'{titles[coord]}[2]')
    #
    #             self._save(fig, f'{titles[coord]}.pdf')
    #         else:
    #             fig, ax = self._fig_ax()
    #             for idx in range(dx):
    #                 ax.plot(
    #                     true_gen_x[coord][:, idx],
    #                     c=colors[idx % len(colors)],
    #                     label=f'{titles[coord]}[{idx}]'
    #                 )
    #             self._apply_common(ax, legend_loc='upper left')
    #             self._save(fig, f'{titles[coord]}.pdf')
    #
    #         # -------- estimate --------
    #         est = self.gen_x_estimates[:, coord, :]
    #
    #         if is_lorenz_3d:
    #             fig = plt.figure()
    #             fig.set_size_inches(10, 7)
    #             ax = fig.add_subplot(111, projection='3d')
    #
    #             ax.plot(
    #                 est[:, 0],
    #                 est[:, 1],
    #                 est[:, 2],
    #                 label=f'{titles[coord]}hat (est)'
    #             )
    #
    #             self._apply_common(ax, legend_loc='upper left', xlim_left=None)
    #             ax.set_xlabel(f'{titles[coord]}hat[0]')
    #             ax.set_ylabel(f'{titles[coord]}hat[1]')
    #             ax.set_zlabel(f'{titles[coord]}hat[2]')
    #
    #             self._save(fig, f'{titles[coord]}hat.pdf')
    #         else:
    #             fig, ax = self._fig_ax()
    #             for idx in range(dx):
    #                 ax.plot(
    #                     est[:, idx],
    #                     c=colors[idx % len(colors)],
    #                     label=f'{titles[coord]}hat[{idx}]'
    #                 )
    #             self._apply_common(ax, legend_loc='upper left')
    #             self._save(fig, f'{titles[coord]}hat.pdf')

    def plot_state_estimation(self):
        gp_name = self.snapshot['gp']['name']
        kx, dx = self.gen_x_estimates.shape[1], self.gen_x_estimates.shape[2]
        dy = self.y.shape[1]
        dt = self.snapshot['gp']['dt']
        colors = ['#ff7f00', '#984ea3', '#377eb8']

        # y
        fig, ax = self._fig_ax()
        for idx in range(dy):
            ax.plot(self.y[:, idx], c=colors[idx], label=f'y[{idx}]')
        #ax.set_ylim(-0.5, 5.5)
        self._apply_common(ax, legend_loc='upper left')
        self._save(fig, 'y.pdf')

        # x noisy
        fig, ax = self._fig_ax()
        for idx in range(dx):
            ax.plot(self.x_noisy[:, idx], colors[idx], label=f'x[{idx}]')
        #ax.set_ylim(-2, 6)
        self._apply_common(ax, legend_loc='upper left')
        self._save(fig, 'x_noisy.pdf')

        # build true generalized coordinates from x via finite differences. The true states are x_noisy (NOT x_clean)
        true_gen_x = [self.x_noisy]
        for _ in range(kx - 1):
            true_gen_x.append(np.diff(true_gen_x[-1], axis=0, prepend=0) / dt)
        true_gen_x = np.array(true_gen_x)

        # per generalized coordinate
        titles = []
        t = 'x'
        for i in range(kx):
            titles.append(t)
            t = t + '`'

        for coord in range(kx):
            # true
            fig, ax = self._fig_ax()
            for idx in range(dx):
                ax.plot(true_gen_x[coord][:, idx], c=colors[idx], label=f'{titles[coord]}[{idx}]')
            self._apply_common(ax, legend_loc='upper left')
            self._save(fig, f'{titles[coord]}.pdf')

            # estimate
            fig, ax = self._fig_ax()
            est = self.gen_x_estimates[:, coord, :]
            for idx in range(dx):
                ax.plot(est[:, idx], c=colors[idx], label=f'{titles[coord]}hat[{idx}]')
            self._apply_common(ax, legend_loc='upper left')
            self._save(fig, f'{titles[coord]}hat.pdf')

    def plot_theta(self):
        z_norm = 1.64485  # ~ 90% CI
        num_params = self.theta.shape[1]
        time = np.arange(self.theta.shape[0])
        colors = self._colors(num_params)
        names = [f"$\\theta_{{{i}}}$" for i in range(num_params)]
        cov_theta = self.cov_theta

        STD = np.sqrt(np.maximum(np.stack([cov_theta[:, i, i] for i in range(num_params)], axis=0), 0))

        for i in range(num_params):
            fig, ax = self._fig_ax()
            self._plot_ci(ax, time, self.theta[:, i], z_norm * STD[i], color=colors[i], alpha=0.3)
            ax.plot(time, self.theta[:, i], color=colors[i], label=names[i])
            #ax.set_ylim(-0.7, 1.6)
            self._apply_common(ax, legend_loc='upper left')
            self._save(fig, f'theta_{i}.pdf')

    def plot_zeta(self):
        precision_forgetting_warm_up = self.snapshot['precision_forgetting']['warm_up']

        # weighted prediction error
        fig, ax = self._fig_ax()
        ax.plot(self.weighted_error, c='r', label='Weighted Prediction Error')
        ax.axvline(x=precision_forgetting_warm_up, color='black', linestyle='--', linewidth=2, label='Warm-up')
        ax.set_ylim(bottom=0)
        self._apply_common(ax, legend_loc='upper left')
        self._save(fig, 'weighted_prediction_error.pdf')

        # zeta
        fig, ax = self._fig_ax()
        ax.plot(self.zeta, c='b', label='Volatility [0,1]')
        ax.axvline(x=precision_forgetting_warm_up, color='black', linestyle='--', linewidth=2, label='Warm-up')
        ax.set_ylim(bottom=0)
        self._apply_common(ax, legend_loc='upper left')
        self._save(fig, 'zeta.pdf')

    def plot_jitters(self, jitters, jitter_name):
        initial_jitter = self.snapshot['optimizer']['jitter']
        steps = ['D', 'E', 'M_x', 'M_y']
        hessian_names = ['x_cov', 'theta_cov', 'lambda_x_cov', 'lambda_y_cov']

        for n in steps:
            combined_values = []
            for dictionary in jitters:
                combined_values.extend(dictionary[n])

            if jitter_name == 'posterior_hessian_jitters':
                # split tuples into 4 lists
                l1, l2, l3, l4 = zip(*combined_values)
                lists = [list(l1), list(l2), list(l3), list(l4)]

                fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
                axes = axes.flatten()
                for j, ax in enumerate(axes):
                    ax.plot(lists[j], label=f'{hessian_names[j]} ({n})')
                    ax.axhline(initial_jitter, color='red', linestyle='--', linewidth=1,
                               label=f'Init. Jitter = {initial_jitter}')
                    ax.legend(loc=1, prop={'size': self.legend_size})
                    ax.grid(True)
                    ax.set_xlim(left=0)
                    ax.tick_params(axis='both', labelsize=self.tick_size)
                fig.tight_layout()
                fig.savefig(os.path.join(self.output_dir, f'posterior_hessian_jitters({n}-step).pdf'),
                            bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            elif jitter_name == 'ozaki_jitters':
                fig, ax = self._fig_ax()
                ax.plot(combined_values)
                ax.axhline(initial_jitter, color='red', linestyle='--', linewidth=1,
                           label=f'Init. Jitter = {initial_jitter}')
                ax.legend(loc=1, prop={'size': self.legend_size})
                ax.grid(True)
                ax.set_xlim(left=0)
                self._save(fig, f'ozaki_jitters({n}-step).pdf')

# base_dir = r'C:\Users\mldaw\pycharm_projects\one-layer-PC-network\results\glv-glv\kx=3'
# sorted_by_fa, sorted_by_mse = collect_and_sort.collect(base_dir, constraint_value=None)
# print(sorted_by_fa[0]['path'])
# data_dir = os.path.join(base_dir, sorted_by_fa[0]['path'])
# os.startfile(data_dir)
# data_dir = (os.path.join(base_dir, sorted_by_fa[0]['path']))
# p = PlotDispatcher(data_dir)
# p.dispatch()

# base_dir = r'C:\Users\mldaw\pycharm_projects\one-layer-PC-network\results\2026-01-17_17-11-01-376800_NA_JNA_TNA_fd0998_C0_768489.85_4.84'
# p = PlotDispatcher(base_dir)
# p.dispatch()

# import os
# import numpy as np
# from matplotlib import cm  # For color maps
# import matplotlib.pyplot as plt
# from matplotlib import colormaps
# import pickle
# import json
# from functions.plotting import delta_method_exp, bpa
#
# class PlotDispatcher:
#     def __init__(self, data_dir):
#         self.legend_size = 22
#         self.tick_size = 22
#         self.base_colors = colormaps['tab20'].colors
#         self.color_cycle = self.base_colors[::2]
#
#
#         self.data_dir = data_dir
#         self.output_dir = os.path.join(data_dir, "plots")
#         os.makedirs(self.output_dir, exist_ok=True)
#
#         # Load the snapshot pickle file
#         file_path = os.path.join(self.data_dir, 'snapshot.json')
#         with open(file_path, 'r') as f:
#             self.snapshot = json.load(f)
#
#
#         # file_path = os.path.join(self.data_dir, 'combo_priors.pkl')
#         # with open(file_path, "rb") as f:
#         #     self.priors = pickle.load(f)
#
#         # Load .npy files and assign each as an instance variable
#         for file_name in os.listdir(self.data_dir):
#             if file_name.endswith(".npy"):
#                 base_name = os.path.splitext(file_name)[0]
#                 data = np.load(os.path.join(self.data_dir, file_name))
#                 setattr(self, base_name, data)
#
#         # np.cumsum is needed to calculate the indices
#         # -1 is necessary to have the actual indices, so we can mark the context shift points
#         self.y_context_lengths= np.cumsum(self.y_context_lengths) - 1
#
#     def dispatch(self):
#         self.plot_pi(key='x')
#         self.plot_pi(key='y')
#         # self.plot_std(key='x')
#         # self.plot_std(key='y')
#         # self.plot_bpa()
#         #self.plot_priors()
#
#         self.plot_vfe()
#         self.plot_lambda(key='x')
#         self.plot_lambda(key='y')
#         self.plot_sensations_predictions()
#         self.plot_state_estimation()
#         self.plot_theta()
#         self.plot_white_noise(key='x')
#         self.plot_white_noise(key='y')
#
#         self.plot_colored_noise(key='x')
#         self.plot_colored_noise(key='y')
#
#         self.plot_sigma_schedule(key='x')
#         self.plot_sigma_schedule(key='y')
#
#         if self.snapshot['precision_forgetting']['flag']:
#             self.plot_zeta()
#         # The saving of jitters in the main function should take place first under these exact names
#         # self.plot_jitters(self.ozaki_hitters, jitter_name='ozaki_jitters')
#         # self.plot_jitters(self.hessian_jitters, jitter_name='hessian_jitters')
#
#
#     def plot_lambda(self, key):
#         if key == 'x':
#             Lambda = self.lambda_x
#             Cov_lambda = self.cov_lambda_x
#         elif key == 'y':
#             Lambda = self.lambda_y
#             Cov_lambda = self.cov_lambda_y
#
#         # grab the standard deviations
#         time = np.arange(Lambda.shape[0])
#         # theta is a tensor with shape (N*num_params)
#         num_lambdas = Lambda.shape[1]
#         names = [f"$\\lambda_{{{key}}}^{{{i}}}$" for i in range(num_lambdas)]
#         # If more params than colors, cycle through the list
#         colors = [self.color_cycle[i % len(self.color_cycle)] for i in range(num_lambdas)]
#
#         STD = []
#         for idx in range(num_lambdas): #could be [2] too since they are squar matrices
#             STD.append(np.sqrt(Cov_lambda[:, idx, idx])) # grap
#         STD = np.array(STD)
#
#         fig = plt.figure(figsize=(16, 9))
#         ax = fig.add_subplot(111)
#
#         for idx in range(num_lambdas):
#             plt.fill_between(time, Lambda[:, idx] - STD[idx], Lambda[:, idx] + STD[idx], alpha=0.3, color=colors[idx])
#
#             ax.plot(time, Lambda[:, idx], color=colors[idx], label=names[idx])
#
#         # for idx in range(len(self.context_lengths) - 1):
#         #
#         #     ax.axvline(x=self.context_lengths[idx], color='k', linestyle='--', linewidth=1)
#
#         plt.grid(True)
#         ax.legend(loc='upper left', prop={'size': self.legend_size})
#         plt.xlim(left=0)
#         plt.xticks(fontsize=self.tick_size)
#         plt.yticks(fontsize=self.tick_size)
#         fig.savefig(os.path.join(self.output_dir, f'lambda_{key}.pdf'), bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#
#     def plot_pi(self, key):
#         """
#         Plots the approximate expected precision and confidence intervals using
#         the Delta method from posterior over log-precision (lambda).
#         """
#         if key == 'x':
#             Lambda = self.lambda_x  # shape: [T, d]
#             Cov_lambda = self.cov_lambda_x  # shape: [T, d, d]
#             pi_schedule = (1 / (self.x_sigma_schedule**2))
#
#         elif key == 'y':
#             Lambda = self.lambda_y
#             Cov_lambda = self.cov_lambda_y
#             pi_schedule = (1 / (self.y_sigma_schedule**2))
#
#         else:
#             raise ValueError(f"Invalid key '{key}'. Must be 'x' or 'y'.")
#
#         time = np.arange(Lambda.shape[0])
#         num_lambdas = Lambda.shape[1]
#         names = [f"$\\pi_{{{key}}}^{{{i}}}$" for i in range(num_lambdas)]
#         colors = [self.color_cycle[i % len(self.color_cycle)] for i in range(num_lambdas)]
#
#         # Compute expected value and STD using Delta method
#         pi_eta_all = []
#         pi_std_all = []
#
#         for t in range(Lambda.shape[0]):
#             mu_t = Lambda[t]  # shape: [d]
#             cov_t = Cov_lambda[t]  # shape: [d, d]
#             pi_eta_t, pi_cov_t = delta_method_exp.compute(mu_t, cov_t)
#
#             pi_eta_all.append(pi_eta_t)
#             pi_std_all.append(np.sqrt(np.maximum(np.diag(pi_cov_t), 1e-12)))  # prevent NaN from sqrt
#
#         pi_eta_all = np.array(pi_eta_all)  # shape: [T, d]
#         pi_std_all = np.array(pi_std_all)  # shape: [T, d]
#
#         # Plotting
#         fig = plt.figure(figsize=(16, 9))
#         ax = fig.add_subplot(111)
#
#         for idx in range(num_lambdas):
#             ax.fill_between(time,
#                             pi_eta_all[:, idx] - pi_std_all[:, idx],
#                             pi_eta_all[:, idx] + pi_std_all[:, idx],
#                             alpha=0.3,
#                             color=colors[idx])
#             ax.plot(time, pi_eta_all[:, idx], color=colors[idx], label=names[idx])
#
#         # for idx in range(len(self.context_lengths) - 1):
#         #     ax.axvline(x=self.context_lengths[idx], color='k', linestyle='--', linewidth=1)
#
#         #Optional: Ground truth precision if available
#         if key == 'y':
#             plt.plot(pi_schedule, color='k', linestyle='--', linewidth=2, label='Ground truth')
#         elif key == 'x':
#             plt.plot(pi_schedule, color='k', linestyle='--', linewidth=2, label='Ground truth')
#
#
#         plt.grid(True)
#         ax.legend(loc='upper left', prop={'size': self.legend_size})
#         plt.xlim(left=0)
#         plt.ylim(bottom=0)
#         plt.xticks(fontsize=self.tick_size)
#         plt.yticks(fontsize=self.tick_size)
#
#         fig.savefig(os.path.join(self.output_dir, f'pi_{key}.pdf'), bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#
#     def plot_sigma_schedule(self, key):
#         if key == 'y':
#             sigma_schedule = self.y_sigma_schedule
#             name = 'y_sigma_schedule.pdf'
#         elif key == 'x':
#             sigma_schedule = self.x_sigma_schedule
#             name = 'x_sigma_schedule.pdf'
#
#         fig = plt.figure(figsize=(16, 9))
#         ax = fig.add_subplot(111)
#         ax.plot(sigma_schedule, c='#a65628', label='%s sigma schedule' % key)
#         # for idx in range(len(self.y_context_lengths)-1):
#         #     ax.axvline(x=self.y_context_lengths[idx], color='k', linestyle='--', linewidth=1)
#         ax.legend(loc=2, prop={'size': self.legend_size})
#         plt.ylim(bottom=0)
#         plt.yticks(fontsize=self.tick_size)
#         plt.xlim(left=0)
#         plt.xticks(fontsize=self.tick_size)
#         plt.grid(True)
#         fig.tight_layout()
#         # Save the plot to the constructed file path
#         fig.savefig(os.path.join(self.output_dir, name), bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#
#         free_action = []
#         temp = self.vfe[0]
#         for idx in range(len(self.vfe)):
#             if idx == 0:
#                 free_action.append(temp)
#             else:
#                 temp += self.vfe[idx]
#                 free_action.append(temp)
#
#         fig = plt.figure(figsize=(16, 9))
#         ax = fig.add_subplot(111)
#         ax.plot(free_action, c='#a65628', label='Free Action')
#         ax.legend(loc=2, prop={'size': self.legend_size})
#         plt.xticks(fontsize=14)
#         plt.ylim(bottom=0)
#         plt.yticks(fontsize=self.tick_size)
#         plt.xlim(left=0)
#         plt.xticks(fontsize=self.tick_size)
#         plt.grid(True)
#         # Save the plot to the constructed file path
#         fig.savefig(os.path.join(self.output_dir, 'fa.pdf'), bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#     def plot_colored_noise(self, key):
#
#         if key == 'y':
#             noise = self.y_colored_noise
#             name = 'y_colored_noise.pdf'
#         elif key == 'x':
#             noise = self.x_colored_noise
#             name = 'x_colored_noise.pdf'
#
#
#         num_dim = noise.shape[0]
#         names = [f"$cn_{{{i}}}$" for i in range(num_dim)]
#         colors = [self.color_cycle[i % len(self.color_cycle)] for i in range(num_dim)]
#
#         fig = plt.figure(figsize=(16, 9))
#         ax = fig.add_subplot(111)
#
#         for idx in range(noise.shape[0]):
#             ax.plot(noise[idx, :], color=colors[idx], label=names[idx])
#
#         # for idx in range(len(self.y_context_lengths)-1):
#         #     ax.axvline(x=self.y_context_lengths[idx], color='k', linestyle='--', linewidth=1)
#         plt.grid(True)
#         ax.legend(loc='lower left', prop={'size': self.legend_size})
#         plt.xlim(left=0)
#         plt.xticks(fontsize=self.tick_size)
#         plt.yticks(fontsize=self.tick_size)
#         fig.savefig(os.path.join(self.output_dir, name), bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#     def plot_white_noise(self, key):
#
#         if key == 'y':
#             noise = self.y_white_noise
#             name = 'y_white_noise.pdf'
#         elif key == 'x':
#             noise = self.x_white_noise
#             name = 'x_white_noise.pdf'
#
#
#         num_dim = noise.shape[0]
#         names = [f"$wn_{{{i}}}$" for i in range(num_dim)]
#         colors = [self.color_cycle[i % len(self.color_cycle)] for i in range(num_dim)]
#
#         fig = plt.figure(figsize=(16, 9))
#         ax = fig.add_subplot(111)
#
#         for idx in range(noise.shape[0]):
#             ax.plot(noise[idx, :], color=colors[idx], label=names[idx])
#
#         # for idx in range(len(self.y_context_lengths) - 1):
#         #     ax.axvline(x=self.y_context_lengths[idx], color='k', linestyle='--', linewidth=1)
#
#         plt.grid(True)
#         ax.legend(loc='lower left', prop={'size': self.legend_size})
#         plt.xlim(left=0)
#         plt.xticks(fontsize=self.tick_size)
#         plt.yticks(fontsize=self.tick_size)
#         fig.savefig(os.path.join(self.output_dir, name), bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#
#     def plot_vfe(self):
#         fig = plt.figure(figsize=(16, 9))
#         ax = fig.add_subplot(111)
#         ax.plot(self.vfe, c='#a65628', label='VFE')
#         # Add vertical dashed lines at context boundaries
#         for idx in range(len(self.y_context_lengths)-1):
#             ax.axvline(x=self.y_context_lengths[idx], color='k', linestyle='--', linewidth=1)
#         ax.legend(loc=2, prop={'size': self.legend_size})
#         plt.ylim(bottom=0)
#         plt.yticks(fontsize=self.tick_size)
#         plt.xlim(left=0)
#         plt.xticks(fontsize=self.tick_size)
#         plt.grid(True)
#         fig.tight_layout()
#         # Save the plot to the constructed file path
#         fig.savefig(os.path.join(self.output_dir, 'vfe.pdf'), bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#
#         free_action = []
#         temp = self.vfe[0]
#         for idx in range(len(self.vfe)):
#             if idx == 0:
#                 free_action.append(temp)
#             else:
#                 temp += self.vfe[idx]
#                 free_action.append(temp)
#
#         fig = plt.figure(figsize=(16, 9))
#         ax = fig.add_subplot(111)
#         ax.plot(free_action, c='#a65628', label='Free Action')
#         for idx in range(len(self.y_context_lengths)-1):
#             ax.axvline(x=self.y_context_lengths[idx], color='k', linestyle='--', linewidth=1)
#         ax.legend(loc=2, prop={'size': self.legend_size})
#
#         plt.xticks(fontsize=14)
#         plt.ylim(bottom=0)
#         plt.yticks(fontsize=self.tick_size)
#         plt.xlim(left=0)
#
#         plt.xticks(fontsize=self.tick_size)
#         plt.grid(True)
#         # Save the plot to the constructed file path
#         fig.savefig(os.path.join(self.output_dir, 'fa.pdf'), bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#
#
#     def plot_sensations_predictions(self):
#         ky, dy = self.gen_sensations.shape[1], self.gen_sensations.shape[2]
#         t = 'y'
#         titles = [t]
#         for i in range(ky - 1):
#             t = t + '`'
#             titles.append(t)
#
#         if ky == 1:
#             counter = 1
#             gen_sensations = np.squeeze(self.gen_sensations, axis=1)
#             gen_predictions = np.squeeze(self.gen_predictions, axis=1)
#
#             for j in range(dy):
#                 fig = plt.figure(figsize=(16, 9))
#                 ax = fig.add_subplot(111)
#                 ax.plot(gen_sensations[:, j], c='#009E73', label=titles[0] + '[%d]' % j)
#                 ax.plot(gen_predictions[:, j], c='#CC79A7', alpha=0.80, label=titles[0] + 'hat' + '[%d]' % j)
#                 ax.legend(loc=2, prop={'size': self.legend_size})
#                 plt.grid()
#                 plt.xlim(left=0)
#                 plt.ylim(-0.5, 5.5)
#                 plt.xticks(fontsize=self.tick_size)
#                 plt.yticks(fontsize=self.tick_size)
#
#                 fig.savefig(os.path.join(self.output_dir, '%s' % titles[0] + '[%d]' % j + '.pdf'), bbox_inches='tight',
#                             pad_inches=0)
#                 plt.close(fig)
#
#                 counter += 1
#         else:
#             counter = 1
#             for i in range(ky):
#                 for j in range(dy):
#                     fig = plt.figure(figsize=(16, 9))
#                     ax = fig.add_subplot(111)
#                     ax.plot(self.gen_sensations[:, i, j], c='#009E73', label=titles[i] + '[%d]' % j)
#                     ax.plot(self.gen_predictions[:, i, j], c='#CC79A7', alpha=0.80, label=titles[i] + 'hat' + '[%d]' % j)
#                     ax.legend(loc=2, prop={'size': self.legend_size})
#                     plt.grid()
#                     plt.ylim(-0.5, 5.5)
#                     plt.xlim(left=0)
#                     plt.xticks(fontsize=self.tick_size)
#                     plt.yticks(fontsize=self.tick_size)
#                     fig.savefig(os.path.join(self.output_dir, '%s' % titles[i] + '[%d]' % j + '.pdf'), bbox_inches='tight',
#                                 pad_inches=0)
#                     plt.close(fig)
#
#                     counter += 1
#     def plot_state_estimation(self):
#         kx, dx = self.gen_x_estimates.shape[1], self.gen_x_estimates.shape[2]
#         dy = self.y.shape[1]
#         dt = self.snapshot['gp']['dt']
#         colors = ['#ff7f00', '#984ea3', '#377eb8']
#         true_gen_x = [self.x_clean]
#         for i in range(kx - 1):
#             true_gen_x.append(np.diff(true_gen_x[-1], axis=0, prepend=0) / dt)
#         true_gen_x = np.array(true_gen_x)
#
#         titles = ['x']
#         t = 'x'
#         for i in range(kx - 1):
#             t = t + '`'
#             titles.append(t)
#
#         fig = plt.figure(figsize=(16, 9))
#         ax = fig.add_subplot(111)
#         for idx in range(dy):
#             ax.plot(self.y[:, idx], c=colors[idx], label='y[%d]' % idx)
#             plt.grid(True)
#         ax.legend(loc=2, prop={'size': self.legend_size})
#         #ylim = plt.ylim()
#         plt.ylim(-0.5,5.5)
#         plt.xlim(left=0)
#         plt.xticks(fontsize=self.tick_size)
#         plt.yticks(fontsize=self.tick_size)
#         fig.savefig(os.path.join(self.output_dir, 'y.pdf'), bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#
#         fig = plt.figure(figsize=(16, 9))
#         ax = fig.add_subplot(111)
#         for idx in range(dx):
#             ax.plot(self.x_noisy[:, idx], colors[idx], label='x[%d]' % idx)
#             plt.grid(True)
#         ax.legend(loc=2, prop={'size': self.legend_size})
#         #plt.ylim(ylim)
#         plt.ylim(-2,6)
#         plt.xlim(left=0)
#         plt.xticks(fontsize=self.tick_size)
#         plt.yticks(fontsize=self.tick_size)
#         fig.savefig(os.path.join(self.output_dir, 'x_noisy.pdf'), bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#
#         coordinate_idx = 0
#         while (coordinate_idx < kx):
#
#             fig = plt.figure(figsize=(16, 9))
#             ax = fig.add_subplot(111)
#             for idx in range(dx):
#                 ax.plot(true_gen_x[coordinate_idx][:, idx], c=colors[idx], label=titles[coordinate_idx] + '[%d]' % idx)
#                 plt.grid(True)
#             ax.legend(loc=2, prop={'size': self.legend_size})
#             plt.xlim(left=0)
#             plt.xticks(fontsize=self.tick_size)
#             plt.yticks(fontsize=self.tick_size)
#             #ylim = plt.ylim()
#             if coordinate_idx >= 1: # velocity
#                 plt.ylim(-5.0, 5.0)
#             else:
#                 plt.ylim(-0.5, 5.5)
#             fig.savefig(os.path.join(self.output_dir, '%s.pdf' % titles[coordinate_idx]), bbox_inches='tight', pad_inches=0)
#             plt.close(fig)
#
#             fig = plt.figure(figsize=(16, 9))
#             ax = fig.add_subplot(111)
#             estimate = self.gen_x_estimates[:, coordinate_idx, :]
#             for idx in range(dx):
#                 ax.plot(estimate[:, idx], c=colors[idx], label=titles[coordinate_idx] + 'hat' + '[%d]' % idx)
#                 plt.grid(True)
#             ax.legend(loc=2, prop={'size': self.legend_size})
#
#
#             plt.xlim(left=0)
#             plt.xticks(fontsize=self.tick_size)
#             plt.yticks(fontsize=self.tick_size)
#             fig.savefig(os.path.join(self.output_dir, '%s' % titles[coordinate_idx] + 'hat' + '.pdf'), bbox_inches='tight',
#                         pad_inches=0)
#             plt.close(fig)
#
#             fig = plt.figure(figsize=(16, 9))
#             ax = fig.add_subplot(111)
#             diff = true_gen_x[coordinate_idx] - self.gen_x_estimates[:, coordinate_idx, :]
#             for idx in range(dx):
#                 ax.plot(diff[:, idx], c=colors[idx],
#                         label='[' + titles[coordinate_idx] + '-' + titles[coordinate_idx] + 'hat' + ']' + '[%d]' % idx)
#                 plt.grid(True)
#             ax.legend(loc=2, prop={'size': self.legend_size})
#             plt.xlim(left=0)
#
#             plt.xticks(fontsize=self.tick_size)
#             plt.yticks(fontsize=self.tick_size)
#             fig.savefig(os.path.join(self.output_dir,
#                                      '%s' % (titles[coordinate_idx] + '-' + titles[coordinate_idx] + 'hat') + '.pdf'),
#                         bbox_inches='tight', pad_inches=0)
#             plt.close(fig)
#
#             coordinate_idx += 1
#     def plot_theta(self):
#         z_norm = 1.64485
#         # stable and slowed down pullback point attractor as the GP
#         # GT = [0.7, 0.5, 0.3, 0.2]
#         # theta is a tensor with shape (N*num_params)
#         num_params = self.theta.shape[1]
#         time = np.arange(self.theta.shape[0])
#         colors = [self.color_cycle[i % len(self.color_cycle)] for i in range(num_params)]
#         names = [f"$\\theta_{{{i}}}$" for i in range(num_params)]
#         cov_theta = self.cov_theta
#
#         STD = []
#         for idx in range(num_params): #could be [2] too since they are squar matrices
#             temp = cov_theta[:, idx, idx]
#             STD.append(np.sqrt(temp)) # grap
#         STD = np.array(STD)
#
#         for idx in range(num_params):
#             fig = plt.figure(figsize=(16, 9))
#             ax = fig.add_subplot(111)
#             plt.fill_between(time, self.theta[:, idx] - z_norm*STD[idx], self.theta[:, idx] + z_norm*STD[idx], alpha=0.3, color=colors[idx])
#             ax.plot(time, self.theta[:, idx], color=colors[idx], label=names[idx])
#
#             #ax.axhline(y=GT[idx], color='gray', linestyle='--', linewidth=1.5, label='Ground truth= %.2f' % GT[idx])
#
#             # for j in range(len(self.context_lengths)-1):
#             #     ax.axvline(x=self.context_lengths[j], color='k', linestyle='--', linewidth=1)
#
#             plt.grid(True)
#             ax.legend(loc='upper left', prop={'size': self.legend_size})
#             plt.xlim(left=0)
#             plt.ylim(-0.7, 1.6)
#             plt.xticks(fontsize=self.tick_size)
#             plt.yticks(fontsize=self.tick_size)
#             #plt.show()
#             fig.savefig(os.path.join(self.output_dir, 'theta_%d.pdf' % idx), bbox_inches='tight', pad_inches=0)
#             plt.close(fig)
#
#     def plot_zeta(self, ):
#         # --- Plot Weighted Prediction Error ---
#         precision_forgetting_warm_up = self.snapshot['precision_forgetting']['warm_up']
#         fig1 = plt.figure(figsize=(16, 9))
#         ax1 = fig1.add_subplot(111)
#         ax1.plot(self.weighted_error, c='r', label='Weighted Prediction Error')
#         ax1.axvline(x=precision_forgetting_warm_up, color='black', linestyle='--', linewidth=2, label='Warm-up')
#         ax1.legend(loc='upper left', prop={'size': self.legend_size})
#         ax1.set_ylim(bottom=0)
#         ax1.set_xlim(left=0)
#         ax1.grid(True)
#         ax1.tick_params(axis='both', labelsize=self.tick_size)
#         fig1.tight_layout()
#         fig1.savefig(os.path.join(self.output_dir, 'weighted_prediction_error.pdf'), bbox_inches='tight', pad_inches=0)
#         plt.close(fig1)
#
#         # --- Plot Zeta (Volatility) ---
#         fig2 = plt.figure(figsize=(16, 9))
#         ax2 = fig2.add_subplot(111)
#         ax2.plot(self.zeta, c='b', label='Volatility [0,1]')
#         ax2.axvline(x=precision_forgetting_warm_up, color='black', linestyle='--', linewidth=2, label='Warm-up')
#         ax2.legend(loc='upper left', prop={'size': self.legend_size})
#
#         ax2.set_ylim(bottom=0)
#         ax2.set_xlim(left=0)
#         ax2.grid(True)
#         ax2.tick_params(axis='both', labelsize=self.tick_size)
#         fig2.tight_layout()
#         fig2.savefig(os.path.join(self.output_dir, 'zeta.pdf'), bbox_inches='tight', pad_inches=0)
#         plt.close(fig2)
#
#     def plot_jitters(self, jitters, jitter_name):
#         initial_jitter = self.snapshot['optimizer']['jitter']
#         # # First save the hessian_jitters as a json file
#         # if jitter_name == 'posterior_hessian_jitters':
#         #     with open(os.path.join(self.output_dir, "posterior_hessian_jitters.json"), "w") as f:
#         #         json.dump(temp_jitters, f)
#         #         jitters = temp_jitters[1:]
#         # elif jitter_name == 'ozaki_jitters':
#         #     with open(os.path.join(self.output_dir, "ozaki_jitters.json"), "w") as f:
#         #         json.dump(temp_jitters, f)
#         #         jitters = temp_jitters
#
#         steps = ['D', 'E', 'M_x', 'M_y']
#         hessian_names = ['x_cov', 'theta_cov', 'lambda_x_cov', 'lambda_y_cov']
#
#         for n in steps:
#             # Initialize the result list
#             combined_values = []
#
#             # Iterate through hessian_jitters[1:] and extract the 'D' values
#             for dictionary in jitters:
#                 combined_values.extend(dictionary[n])
#
#             if jitter_name == 'posterior_hessian_jitters':
#                 # Separate the elements into 4 lists
#                 l1, l2, l3, l4 = zip(*combined_values)
#                 # Convert the tuples to lists (if needed)
#                 l = [list(l1), list(l2), list(l3), list(l4)]
#
#                 # Create a figure of size (16, 9) with 4 subplots
#                 fig, axes = plt.subplots(2, 2, figsize=(16, 9))
#                 for j, ax in enumerate(axes.flatten()):
#                     ax.plot(l[j], label=hessian_names[j] + f' ({n})')
#                     ax.axhline(initial_jitter, color='red', linestyle='--', linewidth=1,
#                                label=f'Init. Jitter = {initial_jitter}')
#                     ax.legend(loc=1, prop={'size': self.legend_size})  # Set legend size and location
#                     ax.grid(True)  # Apply grid to each subplot
#                     ax.set_xlim(left=0)  # Set x-axis limit for each subplot
#                     ax.tick_params(axis='both', labelsize=self.tick_size)  # Set tick size for both axes
#
#                 fig.savefig(os.path.join(self.output_dir, 'posterior_hessian_jitters(%s-step)' % n + '.pdf'))
#                 plt.close(fig)
#
#             elif jitter_name == 'ozaki_jitters':
#                 fig = plt.figure(figsize=(16, 9))
#                 ax = fig.add_subplot(111)
#                 ax.plot(combined_values)
#                 ax.axhline(initial_jitter, color='red', linestyle='--', linewidth=1,
#                            label=f'Init. Jitter = {initial_jitter}')
#                 ax.legend(loc=1, prop={'size': self.legend_size})  # Set legend size and location
#                 ax.grid(True)  # Apply grid to each subplot
#                 ax.set_xlim(left=0)  # Set x-axis limit for each subplot
#                 fig.savefig(os.path.join(self.output_dir, 'ozaki_jitters(%s-step)' % n + '.pdf'))
#                 plt.close(fig)
#
#
# def list_immediate_subdirs(root_dir):
#     return [
#         os.path.join(root_dir, name)
#         for name in os.listdir(root_dir)
#         if os.path.isdir(os.path.join(root_dir, name))
#     ]
#
# # Example usage
# root = r"C:\Users\mldaw\pycharm_projects\one-layer-PC-network\results\2025-10-12_02-39-25-389740_OD_EM_J47814267_T40_08cb25_C40_1306909.92_0.85"
# all_subdirs = list_immediate_subdirs(root)
# total = len(all_subdirs)
# for idx, data_dir in enumerate(all_subdirs):
#     print('plotting %d/%d' % ((idx + 1), total))
#     p = PlotDispatcher(data_dir)
#     p.dispatch()


