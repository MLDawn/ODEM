import os
from datetime import datetime
import uuid  # Ensure this is imported for unique suffix

def generate(base_path, combo_index):
    """
    Generate a unique and traceable folder name for experiment results.

    Structure:
    timestamp_JOBNAME_J{job_id}_T{task_id}_{uuid}_C{combo_index}

    Parameters:
    - base_path (str): Base directory where the folder should be created.
    - combo_index (int or None): Unique index of the experiment combination.

    Returns:
    - str: Full path to a uniquely named folder.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    job_id = os.environ.get('SLURM_JOB_ID', 'NA')
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', 'NA')
    job_name = os.environ.get('SLURM_JOB_NAME', 'NA')
    short_uid = uuid.uuid4().hex[:6]  # Short unique suffix

    if combo_index is not None:
        folder_name = f"{timestamp}_{job_name}_J{job_id}_T{task_id}_{short_uid}_C{combo_index}"
    else:
        folder_name = f"{timestamp}_{job_name}_J{job_id}_T{task_id}_{short_uid}"

    return os.path.join(base_path, folder_name)
