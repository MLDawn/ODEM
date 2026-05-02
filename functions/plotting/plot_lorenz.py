import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # sometimes needed for older matplotlib

def plot(x):
    # x_clean is your (1000, 3) tensor from solving Lorenz
    # shape: [time, dim] = [N, 3]

    # Move to CPU + NumPy for plotting (if it's on GPU / requires_grad)
    traj = x.detach().cpu().numpy()

    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, linewidth=0.8)

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("x₃")
    ax.set_title("Lorenz Attractor")

    plt.tight_layout()
    plt.show()
