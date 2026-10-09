"""fs03: the HSPB1 screens side by side (FS73, FS76, GA_20 salt titration).
  1. Control pull-downs: how similar is the HSPB1 pull-down across experiments (log2 level, centred)?
  2. Ionic effects: does raising Mg2+ (FS76) resemble raising NaCl (GA_20, 0 -> 150 mM)?
Writes data/fs_data/fs03.json and reports/figures/cross/*.png."""
import os, json, warnings
import numpy as np, pandas as pd
from pdpipe import io, cross, stats as st

warnings.filterwarnings("ignore")
FIG = f"{io.ROOT}/reports/figures/cross"; os.makedirs(FIG, exist_ok=True)
d73 = io.load_proteoda(io.load_config("fs73")); d76 = io.load_proteoda(io.load_config("fs76"))
ga = pd.read_csv(f"{io.ROOT}/data/pd_data/targets.tsv", sep="\t").set_index("accession")


def ctl(ds, cond):
    v = ds.X[ds.samples.loc[ds.samples["cond"] == cond, "column"]].mean(1); return v - v.median()


lev = cross.align({"FS73 control": ctl(d73, "ctl|0"), "FS76 Mg 0.25, no Li": ctl(d76, "0.25|0"),
                   "FS76 Mg 2.5, no Li": ctl(d76, "2.5|0"), "GA_20 0 mM salt": ga["mean_base"] - ga["mean_base"].median()})
J = {"levels": cross.scatter_matrix(lev, f"{FIG}/control_levels.png", "HSPB1 control pull-downs, shared proteins"), "n_levels": len(lev)}
mg = st.cond_means(d76.X, d76.samples)
mgeff = mg[[c for c in mg if c.startswith("2.5|")]].mean(1) - mg[[c for c in mg if c.startswith("0.25|")]].mean(1)
eff = cross.align({"FS76 Mg 2.5 - 0.25": mgeff, "GA_20 75 - 0 mM salt": ga["y_s75"], "GA_20 150 - 0 mM salt": ga["y_s150"]})
J["ionic"] = cross.scatter_matrix(eff, f"{FIG}/ionic_effects.png", "Mg2+ (FS76) vs NaCl (GA_20) effects")
J["n_ionic"] = len(eff)
json.dump(J, open(f"{io.ROOT}/data/fs_data/fs03.json", "w"), indent=1, default=float)
print(json.dumps(J, indent=1, default=float))
