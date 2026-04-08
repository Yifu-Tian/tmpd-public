import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def resolve_output_root(default_root, results_dir):
    """
    Resolve benchmark output root.

    The legacy launcher default is "logs". When this value is passed (or empty),
    keep writing to the script-defined default root.
    """
    if results_dir in (None, "", "logs"):
        return default_root

    raw = str(results_dir)
    if raw.startswith("logs/") or raw.startswith("logs\\"):
        return default_root

    candidate = os.path.abspath(os.path.expanduser(raw))
    parent_name = os.path.basename(os.path.dirname(candidate))
    leaf_name = os.path.basename(candidate)
    if parent_name == "logs" and leaf_name.isdigit():
        return default_root

    return candidate


def save_dataframe_csv(df, path, index=False):
    df.to_csv(path, index=index)
    return path
