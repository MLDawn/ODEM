from functions.initialisation import seed, initialise_sequential
from functions.saving import save_numpy_array, serializable
from functions import make_json_safe, unique_directory_name, slurm_task_indices
import os
import json
import importlib
import itertools
import time
from functions.plotting.plot import PlotDispatcher
from algorithms import ODEM

# Seed all sources of randomness to ensure reproducibility
seed.generate()

# Calls the initialise_sequential.py script to set the initial values of the experiment parameters, as specified in parameters.yaml file
parameters, noise = initialise_sequential.set("parameters.yaml")
wrapped_parameters = [v if isinstance(v, list) else [v] for v in parameters]
combinations = list(itertools.product(*wrapped_parameters))

# This takes care of the start/end index for the current slurm task
# If slurm is not being used, then this will simply make the start/end to map to all the combinations
start_index, end_index = slurm_task_indices.compute(len(combinations))

# store all the priors as well as the optimal prior
# store the failed and completed combo during triple estimation
completed_combos, failed_combos = [], []
# Parent directory for logs and results
base_log_dir = "logs"
base_results_dir = 'results'

# Loop over the calculated start_index and end_index
for i in range(start_index, end_index):
    start_time = time.time()
    # Crucial to use try/except to make sure the code will not crash
    try:
        combo = combinations[i]

        (kx, ky, gp_name,
         dt, T, f_name, g_name,
         E_theta, sigma_theta,
         E_pi_x, sigma_lambda_x,
         E_pi_y, sigma_lambda_y,
         nu_x, kappa_x,
         lambda_eta_adapt, lambda_eta_rate, lambda_eta_t_0, lambda_eta_gamma,
         lambda_interval, lambda_beta,
         theta_eta_adapt, theta_eta_rate, theta_eta_t_0, theta_eta_gamma, theta_interval, theta_beta, carry_cov,
         jitter, algorithm_name, device) = combo

        # Extract the dynamics flow and the observation model Python functions
        dynamics = importlib.import_module('functions.generative_model.dynamics')
        likelihood = importlib.import_module('functions.generative_model.likelihood')

        f = getattr(dynamics, f_name)
        g = getattr(likelihood, g_name)

        # EM interval
        EM_interval = theta_interval

        (VFE, accuracy, complexity, gen_sensations, gen_predictions, x_clean, x_noisy, y, gen_x_estimates,
         THETA, LAMBDA_X, LAMBDA_Y, COV_LAMBDA_X, COV_LAMBDA_Y, COV_THETA,
         y_white_noise, y_sigma_schedule, y_colored_noise, y_context_lengths,
         x_white_noise, x_sigma_schedule, x_colored_noise, x_context_lengths,
         p_theta_eta, p_theta_pi, free_action, mse) = (
            ODEM.start(
                kx, ky, f, g, f_name, gp_name,
                dt, T,
                E_theta, sigma_theta,
                E_pi_x, sigma_lambda_x,
                E_pi_y, sigma_lambda_y,
                nu_x, kappa_x,
                lambda_eta_adapt, lambda_eta_rate, lambda_eta_t_0, lambda_eta_gamma, lambda_beta,
                theta_eta_adapt, theta_eta_rate, theta_eta_t_0, theta_eta_gamma, EM_interval, theta_beta,
                jitter, noise,
                carry_cov, device, tqdm_disable=False))

        # Create a subfolder with the name of the algorithm (i.e., OD_E_M or OD_EM) and mae this the new base_results_dir
        # This allows submitting multiple SLURM jobs for different algorithms and making sure the results shall remain separated
        # below, for every combo in the loop, we will append to this to make sure uniqueness
        fa_str = f"{float(free_action.detach()):.2f}"
        mse_str = f"{float(mse):.2f}"

        results_dir = unique_directory_name.generate(base_path=base_results_dir, combo_index= i)
        results_dir += f"_{fa_str}_{mse_str}"

        os.makedirs(results_dir, exist_ok=True)
        assert os.path.exists(results_dir), f"Results directory was not created: {results_dir}"

        # # Save all the prior combos along with their FA and MSE as well as the prior_subset chosen
        # with open(os.path.join(results_dir, "combo_priors.pkl"), "wb") as file_obj:
        #     pickle.dump(combo_priors, file_obj)

        # Save current combo as JSON
        with open(os.path.join(results_dir, "combo.json"), "w") as file_obj:
            json.dump(make_json_safe.convert(combo), file_obj, indent=4)

        # Save all of the key quantities below
        save_numpy_array.save(VFE, 'vfe', directory=results_dir)
        save_numpy_array.save(accuracy, 'accuracy', directory=results_dir)
        save_numpy_array.save(complexity, 'complexity', directory=results_dir)

        save_numpy_array.save(gen_sensations, 'gen_sensations', directory=results_dir)
        save_numpy_array.save(gen_predictions, 'gen_predictions', directory=results_dir)
        save_numpy_array.save(x_clean, 'x_clean', directory=results_dir)# deliberately named it x, so the plotting won't change
        save_numpy_array.save(x_noisy, 'x_noisy', directory=results_dir)

        save_numpy_array.save(y, 'y', directory=results_dir)
        save_numpy_array.save(gen_x_estimates, 'gen_x_estimates', directory=results_dir)
        save_numpy_array.save(THETA, 'theta', directory=results_dir)
        save_numpy_array.save(p_theta_eta, 'prior_theta_eta', directory=results_dir)
        save_numpy_array.save(p_theta_pi, 'prior_theta_pi', directory=results_dir)


        save_numpy_array.save(LAMBDA_X, 'lambda_x', directory=results_dir)
        save_numpy_array.save(LAMBDA_Y, 'lambda_y', directory=results_dir)

        save_numpy_array.save(COV_LAMBDA_X, 'cov_lambda_x', directory=results_dir)
        save_numpy_array.save(COV_LAMBDA_Y, 'cov_lambda_y', directory=results_dir)

        save_numpy_array.save(COV_THETA, 'cov_theta', directory=results_dir)


        save_numpy_array.save(y_sigma_schedule, 'y_sigma_schedule', directory=results_dir)
        y_pi_schedule = 1/(y_sigma_schedule**2)
        save_numpy_array.save(y_pi_schedule, 'y_pi_schedule', directory=results_dir)
        save_numpy_array.save(y_colored_noise, 'y_colored_noise', directory=results_dir)
        save_numpy_array.save(y_white_noise, 'y_white_noise', directory=results_dir)
        save_numpy_array.save(y_context_lengths, 'y_context_lengths', directory=results_dir)

        save_numpy_array.save(x_sigma_schedule, 'x_sigma_schedule', directory=results_dir)
        x_pi_schedule = 1/(x_sigma_schedule**2)
        save_numpy_array.save(x_pi_schedule, 'x_pi_schedule', directory=results_dir)
        save_numpy_array.save(x_white_noise, 'x_white_noise', directory=results_dir)
        save_numpy_array.save(x_colored_noise, 'x_colored_noise', directory=results_dir)
        save_numpy_array.save(x_context_lengths, 'x_context_lengths', directory=results_dir)

        # Recreate a snapshot dictionary variable for current combo
        snapshot = {
            'device': device,
            'algorithm_name': algorithm_name,
            'priors':
                {
                'theta':
                    {
                    'E_theta': E_theta,
                    'sigma_theta': sigma_theta
                    },
                'lambda':
                    {

                    'E_pi_x': E_pi_x,
                    'sigma_lambda_x': sigma_lambda_x,


                    'E_pi_y': E_pi_y,
                    'sigma_lambda_y': sigma_lambda_y

                    }
                },
            'optimizer': {
                'x': {
                    'nu': nu_x,
                    'kappa_x': kappa_x,
                },
                'lambda': {
                    'adapt': lambda_eta_adapt,
                    'eta': {
                        'rate': lambda_eta_rate,
                        't_0': lambda_eta_t_0,
                        'gamma': lambda_eta_gamma
                    },
                    'inter': lambda_interval,
                    'beta': lambda_beta
                },
                'theta': {
                    'adapt': theta_eta_adapt,
                    'eta': {
                        'rate': theta_eta_rate,
                        't_0': theta_eta_t_0,
                        'gamma': theta_eta_gamma
                    },
                    'inter': theta_interval,
                    'beta': theta_beta
                },
                'jitter': jitter
            },
            'gp': {
                'noise':{
                    'x_wn_mu': noise['x']['x_wn_mu'] ,
                    'x_wn_sigma': noise['x']['x_wn_sigma'],
                    'x_cn_kernel_size': noise['x']['x_cn_kernel_size'],
                    'x_cn_kernel_sigma': noise['x']['x_cn_kernel_sigma'],
                    'x_h_value': noise['x']['x_h_value'],
                    'x_lambda_value': noise['x']['x_lambda_value'],
                    },
                'name': gp_name,
                'dt': dt,
                'T': T
            },
            'gm': {
                'dynamics': f_name,
                'likelihood': g_name,
                'kx': kx,
                'ky': ky,
                'y_h_value': noise['y']['y_h_value'],
                'y_lambda_value': noise['y']['y_lambda_value'],
                'noise': {
                    'y_wn_mu': noise['y']['y_wn_mu'],
                    'y_wn_sigma': noise['y']['y_wn_sigma'],
                    'y_cn_kernel_size': noise['y']['y_cn_kernel_size'],
                    'y_cn_kernel_sigma': noise['y']['y_cn_kernel_sigma']
                }
            }
        }

        # add the final free action and mse to the snapshot, for current combo
        snapshot['fa'] = float(free_action.detach())
        snapshot['mse'] = float(mse)

        # Convert the snapshot dict
        snapshot_serializable = {k: serializable.serialize(v) for k, v in snapshot.items()}
        # Save to JSON file
        with open(os.path.join(results_dir, 'snapshot.json'), 'w') as f:
            json.dump(snapshot_serializable, f, indent=4)

        # For now avoid plotting to save space
        p = PlotDispatcher(data_dir=results_dir)
        p.dispatch()

        # Store the current combo as completed
        completed_combos.append(make_json_safe.convert(combo))

    except Exception as e:
        error_info = {
            "combo": make_json_safe.convert(combo),
            "combo_idx": i,
            "error": str(e)
        }
        # Store the failed combo along with the error message and index of the combo
        failed_combos.append(error_info)
        continue

log_dir = unique_directory_name.generate(base_path=base_log_dir, combo_index=None)
# Create the full log directory
os.makedirs(log_dir, exist_ok=True)
assert os.path.exists(log_dir), f"Log directory was not created: {log_dir}"

# At the very end, save the successful and failed combos
# In case parallel slurm jobs are running this script, we have made sure
# that log_dir is unique for each one of them through the unique_directory_name.py helper function
with open(os.path.join(log_dir, 'completed_combos.json'), 'w') as file_obj:
    json.dump(completed_combos, file_obj, indent=4)

with open(os.path.join(log_dir, 'failed_combos.json'), 'w') as file_obj:
    json.dump(failed_combos, file_obj, indent=4)