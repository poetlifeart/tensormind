"""DIMACS clique-benchmark binary graph (.clq.b) -> edge-list npz (n_total + edges, undirected, 0-indexed).

Usage: python3 -I dimacs_clq_b_to_npz.py IN.clq.b OUT.npz

Format (DIMACS asc2bin.c / bin2asc.c, T. Badics): first line = length L of the
ASCII preamble; then L bytes of preamble ('c' comment lines and 'p edge n m');
then, for row i = 0 .. n-1, (i + 8) // 8 bytes holding the lower triangle of the
adjacency matrix: column j of row i is bit (7 - j % 8) of byte j // 8 (most
significant bit first).  Checks made: every decoded edge has j < i (no bit set
on or above the diagonal), the edge count equals the 'p' line, the file is
consumed exactly.  The whole graph is kept (no component is dropped).
"""
import re
import sys

import numpy as np

raw = open(sys.argv[1], 'rb').read()
nl = raw.index(b'\n')
L = int(raw[:nl])
pre = raw[nl + 1:nl + 1 + L].decode('ascii')
n, m = map(int, re.search(r'^p\s+\w+\s+(\d+)\s+(\d+)', pre, re.M).groups())
body = raw[nl + 1 + L:]
pos, edges, bad = 0, [], 0
for i in range(n):
    nb = (i + 8) // 8
    row = body[pos:pos + nb]
    pos += nb
    for j in range(nb * 8):
        if row[j >> 3] & (1 << (7 - (j & 7))):
            if j < i:
                edges.append((j, i))
            else:
                bad += 1
assert pos == len(body), f"file not consumed exactly: used {pos} of {len(body)} bytes"
assert bad == 0, f"{bad} bits set on/above the diagonal"
assert len(edges) == m, f"decoded {len(edges)} edges, p line says {m}"
deg = np.bincount(np.array(edges).ravel(), minlength=n)
print(f"{sys.argv[1]}: n={n} m={m} (p line {n} {m}); isolates={int((deg == 0).sum())}; "
      f"min/max degree {deg.min()}/{deg.max()}; density {2 * m / (n * (n - 1)):.4f}")
np.savez(sys.argv[2], n_total=n, edges=np.array(edges, dtype=np.int64))
print("wrote", sys.argv[2])
