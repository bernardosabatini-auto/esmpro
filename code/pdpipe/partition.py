"""pdpipe.partition: pull-down (IP) paired with its own post-IP supernatant (unbound fraction).

For each reaction (condition x BR) the IP and the supernatant come from the same tube, so
    partition = log2 IP - log2 supernatant
is that reaction's bound : free ratio, up to a constant (the two are separate MS runs, each
median-normalised). A change in partition is a change in binding that is not explained by a change in
how much of the protein is free in the reaction.

pair(ip, sup)        Dataset of partitions over matched reactions and shared proteins
charge(seq)          net charge at pH 7.4 and isoelectric point from sequence (EMPIRICAL pKa set, EMBOSS)
"""
import numpy as np, pandas as pd
from .io import Dataset

PKA = dict(Nterm=8.6, Cterm=3.6, K=10.8, R=12.5, H=6.5, D=3.9, E=4.1, C=8.5, Y=10.1)   # EMBOSS iep


def _net(seq, pH):
    n = {a: seq.count(a) for a in "KRHDECY"}
    pos = sum(n[a] / (1 + 10 ** (pH - PKA[a])) for a in "KRH") + 1 / (1 + 10 ** (pH - PKA["Nterm"]))
    neg = sum(n[a] / (1 + 10 ** (PKA[a] - pH)) for a in "DECY") + 1 / (1 + 10 ** (PKA["Cterm"] - pH))
    return pos - neg


def charge(seq, pH=7.4):
    """(net charge at pH, isoelectric point) for one sequence; NaNs for a missing sequence."""
    if not isinstance(seq, str) or not seq: return np.nan, np.nan
    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _net(seq, mid) > 0: lo = mid
        else: hi = mid
    return _net(seq, pH), (lo + hi) / 2


def pair(ip, sup, name=None):
    """Match reactions by (cond, BR); keep proteins measured in both datasets. Samples carry the IP's
    injection position (the IP runs carry the drift) and both source columns."""
    Si = ip.samples.set_index(["cond", "BR"]); Ss = sup.samples.set_index(["cond", "BR"])
    keys = [k for k in Si.index if k in Ss.index]
    prot = ip.X.index.intersection(sup.X.index)
    cols, rows = {}, []
    for k in keys:
        a, b = Si.loc[k, "column"], Ss.loc[k, "column"]; c = f"{k[0]}_{k[1]}"
        cols[c] = ip.X.loc[prot, a].values - sup.X.loc[prot, b].values
        r = Si.loc[k].to_dict(); r.update(cond=k[0], BR=k[1], column=c, ip_column=a, sup_column=b); rows.append(r)
    X = pd.DataFrame(cols, index=prot)
    S = pd.DataFrame(rows)
    order = [c for c in ip.samples["column"] if c in set(S["ip_column"])]
    S = S.set_index("ip_column", drop=False).loc[order].reset_index(drop=True)
    X = X[S["column"]]
    return Dataset(name or f"{ip.name}_partition", X, S, ip.proteins.loc[prot], None,
                   dict(ip=ip.name, sup=sup.name, n_reactions=len(S), n_proteins=len(prot)))
