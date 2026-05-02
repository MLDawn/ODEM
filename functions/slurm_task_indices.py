import os
import sys


import os
import sys
import numpy as np

def compute(num_total_combos):
    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
        slurm_num_array_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

        indices = np.arange(num_total_combos)
        splits = np.array_split(indices, slurm_num_array_tasks)

        if task_id < len(splits):
            task_slice = splits[task_id]
            if len(task_slice) > 0:
                return int(task_slice[0]), int(task_slice[-1]) + 1
            else:
                return 0, 0
        else:
            return 0, 0
    else:
        return 0, num_total_combos

# def compute(num_total_combos):
#     """
#     Returns the start and end indices of the combinations assigned to this SLURM task.
#
#     Parameters:
#     - num_total_combos (int): Total number of parameter combinations to process.
#
#     Returns:
#     - (start_index, end_index): Tuple of integers indicating the slice of combos for this task.
#     """
#     if len(sys.argv) > 1:
#         task_id = int(sys.argv[1])
#         slurm_num_array_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT"))
#         combos_per_task = num_total_combos // slurm_num_array_tasks
#
#         start_index = task_id * combos_per_task
#         end_index = (
#             start_index + combos_per_task
#             if task_id != slurm_num_array_tasks - 1
#             else num_total_combos
#         )
#     else:
#         start_index, end_index = 0, num_total_combos
#
#     return start_index, end_index
