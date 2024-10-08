import subprocess
import numpy as np
import linecache

def parse_log(
    string_fields: str="Step         PotEng",
    string_end_md_steps: str="Loop time of",
    log_file: str="log.lammps",
    num_steps: int=1000,
):
    """
    Efficiently parse a large LAMMPS log file, by only reading a subset of the lines. A total
    of `num_steps` lines will be read, evenly spaced between the first line containing the
    string `string_fields` and the first line containing the string `string_end_md_steps`.
    The function will return a dictionary with the thermo fields as keys and the values as lists, e.g.
    ```
    {"Step": [0, 1, 2, ...], "PotEng": [-1.0, -1.1, -1.2, ...]}
    ```

    Args:
        string_fields (str): String that marks the beginning of the thermo fields in the log file.
            Default is "Step         PotEng".
        string_end_md_steps (str): String that marks the end of the MD steps in the log file.
            Default is "Loop time of".
        log_file (str): Path to the LAMMPS log file. Default is "log.lammps".
        num_steps (int): Number of steps to read.

    Returns:
        dict: Dictionary with thermo fields as keys and lists of values as values.
    """
    out = subprocess.run(["grep", "-wn", string_fields, log_file], capture_output=True, text=True)
    first_line, fields = int(out.stdout.split()[0].split(":")[0]), out.stdout.split()[1:]
    last_lines = [i for i in out.stdout.split() if "Loop" in i] # Can match more than one
    # Select first one after first_line
    last_line = [int(i.split(":")[0]) for i in last_lines if int(i.split(":")[0]) > first_line][0]

    x = np.linspace(first_line+1, last_line-1, num_steps, dtype=int)
    lines = []
    for i in x:
        lines.append(
            [float(n) for n in linecache.getline(log_file, i).split()]
        )
    data = {}
    for i, k in enumerate(fields):
        data[k] = [l[i] for l in lines]
    return data