"""Parse USF free association cue-target pairs; build Palla et al. 2005's
undirected weighted word-association graph (weight = FSG(a->b) + FSG(b->a)).
Usage: python -I usf_parse.py RAWDIR"""
import sys, os, glob, collections
raw = sys.argv[1]
rows = 0; bad = 0
w = collections.defaultdict(float)
dirw = {}
words = set(); cues = set()
for f in sorted(glob.glob(os.path.join(raw, "Cue_Target_Pairs.*"))):
    for line in open(f, encoding="latin-1"):
        line = line.strip()
        if not line or line.startswith("<") or line.startswith("CUE,"):
            continue
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6:
            bad += 1; continue
        cue, tgt, fsg = p[0], p[1], p[5]
        try:
            fsg = float(fsg)
        except ValueError:
            bad += 1; continue
        rows += 1
        cues.add(cue); words.add(cue); words.add(tgt)
        dirw[(cue, tgt)] = fsg
print("rows", rows, "bad", bad, "distinct cues", len(cues), "distinct words", len(words), "directed pairs", len(dirw))
und = collections.defaultdict(float)
self_loops = 0
for (a, b), x in dirw.items():
    if a == b:
        self_loops += 1; continue
    und[tuple(sorted((a, b)))] += x
print("self loops", self_loops, "undirected links", len(und))
nodes = {v for e in und for v in e}
print("nodes with >=1 link", len(nodes))
import pickle
pickle.dump(dict(und), open(os.path.join(os.path.dirname(raw.rstrip('/')), "usf_undirected_weights.pkl"), "wb"))
