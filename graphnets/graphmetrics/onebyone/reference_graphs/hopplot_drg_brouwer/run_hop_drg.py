import json, subprocess, sys, os
from fractions import Fraction
OB = "/home/vahid/testcopy/graphnets/graphmetrics/onebyone"
D = os.path.join(OB, "reference_graphs/hopplot_drg_brouwer")
SC = "/tmp/claude-1000/-home-vahid/35024967-977c-4994-9962-d3a35007a9c2/scratchpad/refsearch_agent3/hopjson"
os.makedirs(SC, exist_ok=True)
exp = {}
for f in ("drg_expected.json", "drg_expected_more.json"):
    for k, v in json.load(open(os.path.join(D, f))).items():
        kk = v.get("k_from_published", v.get("k"))
        exp[k] = (v["n"], kk)
allok = True
for name, (n, k) in exp.items():
    js = os.path.join(SC, f"{name}.json")
    subprocess.run([sys.executable, os.path.join(OB, "static_test_hop_plot.py"), "--graph", os.path.join(D, f"{name}.npz"), "--json", js], check=True, capture_output=True)
    st = json.load(open(js))["stat"]
    cum = [sum(k[1:h+1]) for h in range(1, 7)]
    pub_count = [n * c for c in cum]              # ordered pairs
    pub_frac = [c / (n - 1) for c in cum]
    same_count = st["hop_count"] == pub_count
    same_frac = all(a == b for a, b in zip(st["hop_frac"], pub_frac))
    allok &= same_count and same_frac
    print(f"{name:13s} n={st['n']:3d} m={st['m']:4d} ours hop_count={st['hop_count']} published n*sum(k)={pub_count} SAME={same_count}; "
          f"hop_frac[h=2] ours={st['hop_frac'][1]!r} published={Fraction(cum[1], n-1)}={pub_frac[1]!r} SAME={same_frac}")
print("ALL SAME:", allok)
