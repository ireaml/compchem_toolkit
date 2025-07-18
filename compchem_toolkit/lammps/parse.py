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
    # Check if several lines with ":"
    lines_starting_md = [(index, i) for index, i in enumerate(out.stdout.split()) if ":" in i]
    if len(lines_starting_md) > 1:
        print("Several MD runs found in log file. Selecting last one.")
        #print(lines_starting_md)
    first_line, fields = (
        int(lines_starting_md[-1][1].split(":")[0]), # Index of line in file
        out.stdout.split()[lines_starting_md[-1][0]+1:]
    )
    out_last = subprocess.run(["grep", "-wn", string_end_md_steps, log_file], capture_output=True, text=True)
    index_lines = [l for l in out_last.stdout.split() if ":" in l]
    last_line = int(index_lines[-1].split(":")[0])
    print("First and Last line", first_line, last_line)
    print("Fields", fields)

    x = np.linspace(first_line+1, last_line-1, num_steps, dtype=int)
    lines = []
    for i in x:
        l = linecache.getline(log_file, i)
        if l:
            # Check it is a float
            if len(l.split()) == len(fields):
                try:
                    lines.append(
                        [float(n) for n in l.split()]
                    )
                except ValueError:
                    pass
    data = {}
    for i, k in enumerate(fields):
        data[k] = [l[i] for l in lines]
    return data