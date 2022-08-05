
def parse_acf(filename):
    data = []
    with open(filename) as f:
        raw = f.readlines()
        headers = ("x", "y", "z", "charge", "min_dist", "atomic_vol")
        raw.pop(0)
        raw.pop(0)
        while True:
            l = raw.pop(0).strip()
            if l.startswith("-"):
                break
            vals = map(float, l.split()[1:])
            data.append(dict(zip(headers, vals)))
        for l in raw:
            toks = l.strip().split(":")
            if toks[0] == "VACUUM CHARGE":
                vacuum_charge = float(toks[1])
            elif toks[0] == "VACUUM VOLUME":
                vacuum_volume = float(toks[1])
            elif toks[0] == "NUMBER OF ELECTRONS":
                nelectrons = float(toks[1])
    return data
