import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

# Define the grid
eta_vals = np.linspace(-5, 5, 200)  # log-mean
Pi_vals = np.linspace(1, 1024, 200)  # precision (inverse variance in log space)
ETA_lambda, PI_lambda = np.meshgrid(eta_vals, Pi_vals)

# Compute E[Lambda] and Var[Lambda]
E_precision = np.exp(ETA_lambda + 0.5 / PI_lambda)
Var_precision = (np.exp(1 / PI_lambda) - 1) * np.exp(2 * ETA_lambda + 1 / PI_lambda)
Std_precision = np.sqrt(Var_precision)

# Contour levels
expected_levels = [0.5, 1.0, 2.0, 5.0, 10.0]
variance_levels = [0.01, 0.1, 1.0, 10.0, 100.0]
stddev_levels = [0.1, 0.3, 0.5, 1.0, 2.0]

# Colors and linestyles
colors = ['cyan', 'lime', 'yellow', 'orange', 'white']
linestyles = ['-', '--', ':', '-.', '-']

# === Plot 1: Expected Value of Precision
plt.figure(figsize=(10, 6))
cf1 = plt.contourf(ETA_lambda, PI_lambda, E_precision, levels=100, cmap='viridis')
legend_lines = []
for level, color, ls in zip(expected_levels, colors, linestyles):
    c = plt.contour(ETA_lambda, PI_lambda, E_precision, levels=[level], colors=color, linestyles=ls, linewidths=2)
    legend_lines.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=2, label=f'E[$\Pi$] = {level}'))
plt.legend(handles=legend_lines, loc='upper left', fontsize=10)
plt.colorbar(cf1, label='E[$\Pi$]')
plt.title('Expected Value of Precision E[$\Pi$]')
plt.xlabel('$\eta_\Lambda$ (mean of log-precision)')
plt.ylabel('$\Pi_\Lambda$ (precision of log-precision)')
plt.yscale('log')
plt.grid(True)
plt.savefig('precision_eta.pdf', bbox_inches='tight', pad_inches=0)

# === Plot 2: Variance of Precision
plt.figure(figsize=(10, 6))
cf2 = plt.contourf(ETA_lambda, PI_lambda, Var_precision, levels=100, cmap='plasma')
legend_lines = []
for level, color, ls in zip(variance_levels, colors, linestyles):
    c = plt.contour(ETA_lambda, PI_lambda, Var_precision, levels=[level], colors=color, linestyles=ls, linewidths=2)
    legend_lines.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=2, label=f'Var[$\Pi$] = {level}'))
plt.legend(handles=legend_lines, loc='upper left', fontsize=10)
plt.colorbar(cf2, label='Var[$\Pi$]')
plt.title('Variance of Precision Var[$\Pi$]')
plt.xlabel('$\eta_\Lambda$ (mean of log-precision)')
plt.ylabel('$\Pi_\Lambda$ (precision of log-precision)')
plt.yscale('log')
plt.grid(True)
plt.savefig('precision_var.pdf', bbox_inches='tight', pad_inches=0)



# === Plot 3: Implied Standard Deviation
plt.figure(figsize=(10, 6))
cf3 = plt.contourf(ETA_lambda, PI_lambda, Std_precision, levels=100, cmap='inferno')
legend_lines = []
for level, color, ls in zip(stddev_levels, colors[:len(stddev_levels)], linestyles[:len(stddev_levels)]):
    c = plt.contour(ETA_lambda, PI_lambda, Std_precision, levels=[level], colors=color, linestyles=ls, linewidths=2)
    legend_lines.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=2, label=f'Std ≈ {level}'))
plt.legend(handles=legend_lines, loc='upper left', fontsize=10)
plt.colorbar(cf3, label='Implied Standard Deviation (1/√E[$\Pi$])')
plt.title('Implied Standard Deviation with Styled Legend')
plt.xlabel('$\eta_\Lambda$ (mean of log-precision)')
plt.ylabel('$\Pi_\Lambda$ (precision of log-precision)')
plt.yscale('log')
plt.grid(True)
plt.savefig('precision_std.pdf', bbox_inches='tight', pad_inches=0)

# plt.tight_layout()
#
# plt.show()
