import os
import json


def collect(base_dir, constraint_value=None):
    results = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        if 'snapshot.json' not in files:
            continue

        snapshot_path = os.path.join(root, 'snapshot.json')

        try:
            with open(snapshot_path, 'r') as f:
                data = json.load(f)

            # ----- optional constraint check -----
            if constraint_value is not None:
                try:
                    E_pi_y = data['priors']['lambda']['E_pi_y']
                except KeyError:
                    continue

                if E_pi_y != constraint_value:
                    continue
            # -------------------------------------

            fa = data.get('fa', None)
            mse = data.get('mse', None)

            if fa is not None and mse is not None:
                results.append({
                    'path': root,
                    'fa': fa,
                    'mse': mse
                })

        except Exception as e:
            print(f"⚠️ Could not read {snapshot_path}: {e}")

    if not results:
        msg = "No valid snapshot.json files found."
        if constraint_value is not None:
            msg += " (constraint not met)"
        print(msg)
        return None, None

    # Sort separately
    sorted_by_fa = sorted(results, key=lambda x: x['fa'])
    sorted_by_mse = sorted(results, key=lambda x: x['mse'])

    return sorted_by_fa, sorted_by_mse
