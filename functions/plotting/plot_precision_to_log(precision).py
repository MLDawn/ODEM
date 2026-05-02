import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define grid in precision space
E_vals = np.linspace(0.1, 10, 200)
Var_vals = np.linspace(0.01, 100, 200)
E_grid, Var_grid = np.meshgrid(E_vals, Var_vals)

# Inverse mapping to log-normal parameters
Pi_grid = 1 / np.log(1 + Var_grid / E_grid**2)
Eta_grid = np.log(E_grid) - 0.5 * np.log(1 + Var_grid / E_grid**2)
std_log_space = np.sqrt(1 / Pi_grid)

# Define candidate contour levels and styles
levels_eta = [-4, -2, 0, 2, 4]
levels_pi = [0.1, 0.5, 1.0, 2.0, 5.0]
levels_std = [0.5, 1.0, 2.0, 4.0, 8.0]
colors = ['cyan', 'lime', 'yellow', 'orange', 'white']
linestyles = ['-', '--', ':', '-.', '-']

# Filter only valid levels within surface range
valid_levels_eta = [lvl for lvl in levels_eta if Eta_grid.min() <= lvl <= Eta_grid.max()]
valid_levels_pi = [lvl for lvl in levels_pi if Pi_grid.min() <= lvl <= Pi_grid.max()]
valid_levels_std = [lvl for lvl in levels_std if std_log_space.min() <= lvl <= std_log_space.max()]

# Plot 1: η (log-mean)
plt.figure(figsize=(10, 6))
cf1 = plt.contourf(E_grid, Var_grid, Eta_grid, levels=100, cmap='viridis')
legend_lines = []
for level, color, ls in zip(valid_levels_eta, colors, linestyles):
    c = plt.contour(E_grid, Var_grid, Eta_grid, levels=[level], colors=color, linestyles=ls, linewidths=2)
    legend_lines.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=2, label=f'$\eta$ = {level}'))
plt.legend(handles=legend_lines, loc='upper left', fontsize=10)
plt.colorbar(cf1, label='$\eta_\Lambda$')
plt.title('Mapping from Precision Space to $\eta_\Lambda$')
plt.xlabel('E[$\Pi$] (Expected Precision)')
plt.ylabel('Var[$\Pi$] (Precision Variance)')
plt.grid(True)
plt.savefig('lambda_eta.pdf', bbox_inches='tight', pad_inches=0)


# Plot 2: Π (log-precision)
plt.figure(figsize=(10, 6))
cf2 = plt.contourf(E_grid, Var_grid, Pi_grid, levels=100, cmap='plasma')
legend_lines = []
for level, color, ls in zip(valid_levels_pi, colors, linestyles):
    c = plt.contour(E_grid, Var_grid, Pi_grid, levels=[level], colors=color, linestyles=ls, linewidths=2)
    legend_lines.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=2, label=f'$Pi_\Lambda$ = {level}'))
plt.legend(handles=legend_lines, loc='upper left', fontsize=10)
plt.colorbar(cf2, label='Π (log-precision)')
plt.title('Mapping from Precision Space to Π in Log-Normal Space')
plt.xlabel('E[$\Pi$] (Expected Precision)')
plt.ylabel('Var[$\Pi$] (Precision Variance)')
plt.grid(True)
plt.savefig('lambda_precision.pdf', bbox_inches='tight', pad_inches=0)

# Plot 3: Std of log-normal prior
plt.figure(figsize=(10, 6))
cf3 = plt.contourf(E_grid, Var_grid, std_log_space, levels=100, cmap='inferno')
legend_lines = []
for level, color, ls in zip(valid_levels_std, colors, linestyles):
    c = plt.contour(E_grid, Var_grid, std_log_space, levels=[level], colors=color, linestyles=ls, linewidths=2)
    legend_lines.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=2, label=f'$std_\Lambda$ = {level}'))
plt.legend(handles=legend_lines, loc='upper left', fontsize=10)
plt.colorbar(cf3, label='$Std[log(\Pi)]$ = sqrt(1/Π)')
plt.title('Mapping from Precision Space to Std of Log-Normal Space')
plt.xlabel('E[$\Pi$] (Expected Precision)')
plt.ylabel('Var[$\Pi$] (Precision Variance)')
plt.grid(True)
plt.savefig('lambda_std.pdf', bbox_inches='tight', pad_inches=0)

plt.tight_layout()
plt.show()
