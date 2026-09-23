#!/usr/bin/env python3
"""
EXPRESSIVENESS OF THE CHROMATIC TENSOR GRAMMAR
==============================================

Two questions, two parts of this file.  The brain is NOT the target here.

  PART 1  What region of graph-property space can the grammar reach?  Vary the
          seed and the chromatic number and measure what comes out.

  PART 2  Where do real graphs that are concurrently computational -- circuit
          netlists, dependency graphs, dataflow -- sit in that same space?

THE GRAMMAR, stated once
------------------------
Exactly the construction of construction/bilateral/construct.py, with the
seed and the number of colour classes left free:

    0. a seed S on n_s vertices carrying a proper chi-colouring
    1. G_1 = S
    2. G_{k+1} = G_k (x) S              tensor (Kronecker) product, deduped
    3. after each step, add alpha_k * n chromatically legal triangle closures:
       pick v with probability proportional to deg(v)^beta, take two of its
       neighbours u,w, add {u,w} if colour(u) != colour(w) and block(u)==block(w)
    4. colour and block labels ride on the leading base-n_s digit of the index

Step 3 calls construct.py's close_triangles DIRECTLY -- it is not
reimplemented here -- so the closure rule cannot drift from the deposited one.
Steps 2 and 4 are reimplemented because construct.py's grow() hard-codes
SEED_N = 20; the code below mirrors it line for line and is marked GRAMMAR.

Chi is set by the seed's colouring and is preserved by the product (Weichsel):
every product edge joins fibres whose seed vertices are adjacent, and every
seed edge is cross-colour, so chi(G_k) <= chi(S) for all k without a check.
Closure is filtered to be cross-colour, so the bound survives.

RUNNING IT
----------
Designed for a multi-day run.  Results are appended to results/*.jsonl one
line per graph, flushed immediately, and every invocation SKIPS configurations
already present.  Kill it and restart it freely.

    python3 expressiveness.py --part 1                 # the grammar sweep
    python3 expressiveness.py --part 2                 # comparison graphs
    python3 expressiveness.py --part 1 --max-nodes 5000    # a faster pass
    python3 expressiveness.py --summary                # read what exists so far
"""
import argparse, itertools, json, os, sys, time, hashlib, urllib.request, gzip, io
import numpy as np
import networkx as nx
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
GRAPHS = os.path.join(HERE, 'graphs')
os.makedirs(RESULTS, exist_ok=True); os.makedirs(GRAPHS, exist_ok=True)

# ---- the deposited closure rule, imported rather than copied ----------------
_B = os.path.join(HERE, '..', 'construction', 'bilateral')
_spec = importlib.util.spec_from_file_location("cc", os.path.join(_B, 'construct.py'))
cc = importlib.util.module_from_spec(_spec); sys.modules["cc"] = cc; _spec.loader.exec_module(cc)
close_triangles = cc.close_triangles          # GRAMMAR step 3, verbatim

# ---- the LOCAL construction's operators, also imported rather than copied ---
_L = os.path.join(HERE, '..', 'construction', 'local')
_lspec = importlib.util.spec_from_file_location(
    "lr", os.path.join(_L, 'emergence_localrepair_standalone.py'))
lr = importlib.util.module_from_spec(_lspec); sys.modules["lr"] = lr
_lspec.loader.exec_module(lr)
prune_local = lr.triangle_prune_local        # prune to a budget, keep top triangle score
add_chromatic = lr.add_chromatic_to_budget   # add back chromatically legal closures
edge_budget = lr.edge_budget                 # n * sqrt((n-1)/2)
tensor_lr = lr.tensor_product


# =============================================================================
# SEEDS
# =============================================================================

def chromatic_number_exact(G, cap=8):
    """Exact chi by brute force.  Only for tiny seeds (n <= 12)."""
    n = G.number_of_nodes()
    if n == 0: return 0
    nodes = sorted(G.nodes()); idx = {v: i for i, v in enumerate(nodes)}
    adj = [[idx[w] for w in G[v]] for v in nodes]
    for k in range(1, cap + 1):
        col = [-1] * n
        def go(i):
            if i == n: return True
            used = {col[w] for w in adj[i] if col[w] >= 0}
            top = min(k, (max(col[:i]) + 2) if i else 1)   # symmetry breaking
            for c in range(top):
                if c not in used:
                    col[i] = c
                    if go(i + 1): return True
                    col[i] = -1
            return False
        if go(0): return k
    return None


def proper_colouring_ok(colours, edges):
    return all(colours[u] != colours[v] for u, v in edges)


def make_seed(kind, n_s, chi, rng, balance='balanced', nblocks=1):
    """Return (edges, colours, chi_true) or None.

    chi is the number of colour CLASSES used to build the seed; it is an upper
    bound on the seed's chromatic number, not the chromatic number itself.  A
    balanced class assignment always yields a proper chi-colouring, but the
    graph is often colourable with fewer, so chi_true is MEASURED exactly and
    the sweep labels every configuration by chi_true.  Requesting chi=6 and
    receiving a 3-chromatic seed is common and is not an error; it simply means
    that configuration contributes a chi=3 data point.

    Every seed must also be connected and, when chi_true >= 3, non-bipartite,
    so that odd cycles survive every tensor power.
    """
    if n_s < chi: return None
    if balance == 'balanced':
        sizes = [n_s // chi + (1 if i < n_s % chi else 0) for i in range(chi)]
    else:
        # one class about twice the others, as the paper's (6,6,8) seed does
        base = max(1, n_s // (chi + 1))
        sizes = [base] * (chi - 1) + [n_s - base * (chi - 1)]
        if sizes[-1] < 1: return None
    colours = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)]).astype(np.int32)

    if kind == 'complete_multipartite':
        edges = [(u, v) for u in range(n_s) for v in range(u + 1, n_s)
                 if colours[u] != colours[v]]
    elif kind == 'random_sparse':
        target = int(round(1.6 * n_s))
        edges = _random_legal(n_s, colours, target, rng)
    elif kind == 'random_medium':
        target = int(round(3.0 * n_s))
        edges = _random_legal(n_s, colours, target, rng)
    elif kind == 'random_dense':
        allp = [(u, v) for u in range(n_s) for v in range(u + 1, n_s) if colours[u] != colours[v]]
        target = int(round(0.6 * len(allp)))
        edges = _random_legal(n_s, colours, target, rng)
    elif kind == 'clique_chain':
        # A chain of K_chi cliques.  Each clique takes ONE vertex from EACH
        # colour class, so every pair inside it is cross-colour and the whole
        # clique survives the legality filter.  An earlier version built the
        # cliques on CONSECUTIVE indices, which share a colour under a balanced
        # assignment, so the filter deleted most clique edges and no K_chi ever
        # survived (max clique came out 2-3, never chi).
        byclass = [np.where(colours == c)[0] for c in range(chi)]
        reps = min(len(x) for x in byclass)
        if reps == 0: return None
        edges = []
        for r in range(reps):
            vs = [int(byclass[c][r]) for c in range(chi)]
            edges += [(min(a, b), max(a, b))
                      for ii, a in enumerate(vs) for b in vs[ii + 1:]]
        # link consecutive cliques through one cross-colour pair each
        for r in range(reps - 1):
            a, b = int(byclass[0][r]), int(byclass[1][r + 1])
            edges.append((min(a, b), max(a, b)))
        # attach any leftover vertices legally
        used = {v for e in edges for v in e}
        for v in range(n_s):
            if v in used: continue
            cand = [w for w in used if colours[w] != colours[v]]
            if cand: edges.append((min(v, cand[0]), max(v, cand[0])))
    elif kind == 'star_of_cliques':
        # one hub joined to everything legal, plus a light random rest
        edges = [(0, v) for v in range(1, n_s) if colours[0] != colours[v]]
        edges += [e for e in _random_legal(n_s, colours, int(1.5 * n_s), rng,
                                           avoid=set(edges)) if e not in set(edges)]
    else:
        return None

    edges = sorted({(min(u, v), max(u, v)) for u, v in edges})
    if not edges: return None
    if not proper_colouring_ok(colours, edges): return None
    G = nx.Graph(edges); G.add_nodes_from(range(n_s))
    if not nx.is_connected(G): return None
    chi_true = chromatic_number_exact(G)          # exact; ~8 ms at n_s = 20
    if chi_true is None: return None
    if chi_true >= 3 and nx.is_bipartite(G): return None
    # recolour with exactly chi_true classes so the labels the grammar carries
    # match the measured chromatic number
    cols = nx.coloring.greedy_color(G, strategy='DSATUR')
    if max(cols.values()) + 1 != chi_true:
        return None                                # DSATUR missed the optimum
    colours = np.array([cols[v] for v in range(n_s)], dtype=np.int32)
    if not proper_colouring_ok(colours, edges): return None

    # BLOCK LABELS.  Blocks partition the seed independently of colour, and
    # closure is confined inside a block, so they are what produce modular
    # organisation.  Assign contiguous blocks, then require that every block
    # contains at least two colours -- otherwise closure inside that block can
    # never fire, for the same reason chi=2 is inert globally.
    if nblocks <= 1:
        blocks = np.zeros(n_s, np.int32)
    else:
        if n_s < 2 * nblocks: return None
        bsz = [n_s // nblocks + (1 if i < n_s % nblocks else 0) for i in range(nblocks)]
        blocks = np.concatenate([np.full(z, i) for i, z in enumerate(bsz)]).astype(np.int32)
        for b in range(nblocks):
            if len(set(colours[blocks == b].tolist())) < 2: return None
        # the seed must still be connected ACROSS blocks, else the product is
        # a disjoint union of block-worlds
        if not nx.is_connected(nx.Graph([e for e in edges])): return None
    return edges, colours, chi_true, blocks


def _random_legal(n_s, colours, target, rng, avoid=None):
    allp = [(u, v) for u in range(n_s) for v in range(u + 1, n_s) if colours[u] != colours[v]]
    if avoid: allp = [p for p in allp if p not in avoid]
    if not allp: return []
    target = min(target, len(allp))
    pick = rng.choice(len(allp), size=target, replace=False)
    return sorted(allp[i] for i in pick)


# =============================================================================
# THE GRAMMAR
# =============================================================================

def seed_invariants(seed_edges, n_s, steps):
    """What the seed fixes BEFORE any closure, so closure's contribution is
    separable in the analysis.

    Two hard constraints on the grammar's expressiveness, both verified:

      DENSITY.  m_{k+1} = 2 m_k m_s and n_{k+1} = n_k n_s, so the pure product
      has m_k = 2^{k-1} m_s^k and mean degree exactly (2 m_s / n_s)^k.  Mean
      degree is therefore the SEED's mean degree raised to the step count;
      alpha can only add on top of it.  A sweep that reports the density range
      it reaches is reporting the range of seed densities, raised to a power.

      SPECTRUM.  A(S^{(x)k}) = A(S)^{(x)k}, so the eigenvalues of the pure
      product are exactly the k-fold products of the seed's eigenvalues
      (verified numerically to 6e-15).  Before closure the grammar has no
      spectral freedom at all beyond the seed's own spectrum, so every
      spectral quantity -- lambda_max, spectral gap, Estrada, graph energy --
      is a deterministic function of n_s numbers.
    """
    m_s = len(seed_edges)
    G = nx.Graph(seed_edges); G.add_nodes_from(range(n_s))
    ev = np.sort(np.linalg.eigvalsh(nx.to_numpy_array(G, nodelist=range(n_s))))[::-1]
    return dict(seed_m=m_s, seed_meandeg=2 * m_s / n_s,
                seed_density=2 * m_s / (n_s * (n_s - 1)),
                seed_eigenvalues=[round(float(x), 6) for x in ev],
                seed_lambda_max=float(ev[0]),
                product_m=int(2 ** (steps - 1) * m_s ** steps),
                product_meandeg=float((2 * m_s / n_s) ** steps),
                product_lambda_max=float(ev[0] ** steps))


def grow(seed_edges, seed_colours, alphas, beta, n_steps, rng,
         max_edges=3_000_000, seed_blocks=None):
    """GRAMMAR steps 1-4.  Mirrors construct.py's grow() with a free seed size.

    construct.py hard-codes SEED_N = 20; the only changes here are that S is a
    parameter, blocks are a single class (the seed carries no block labels),
    and growth aborts if the edge count would exceed max_edges.
    """
    S = len(seed_colours)
    su = np.array([a for a, b in seed_edges] + [b for a, b in seed_edges], np.int64)
    sv = np.array([b for a, b in seed_edges] + [a for a, b in seed_edges], np.int64)
    u = np.array([a for a, b in seed_edges], np.int64)
    v = np.array([b for a, b in seed_edges], np.int64)
    n = S
    colour = seed_colours.copy()
    block = (np.zeros(n, np.int32) if seed_blocks is None
             else np.asarray(seed_blocks, np.int32).copy())
    trace = []
    before = len(u)
    u, v = close_triangles(u, v, n, colour, block, int(alphas[0] * n), rng, beta)
    trace.append(dict(step=1, n=n, product=before, closure=len(u) - before, total=len(u)))
    for step in range(2, n_steps + 1):
        if len(u) * 2 * len(seed_edges) > max_edges:
            return None, None, None, None, trace, 'aborted_too_many_edges'
        u = (u[:, None] * S + su[None, :]).ravel()
        v = (v[:, None] * S + sv[None, :]).ravel()
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        keys = np.unique(lo * (S ** step) + hi)
        u, v = keys // (S ** step), keys % (S ** step)
        n = S ** step
        digit = (np.arange(n) // (S ** (step - 1))) % S
        # colour AND block ride the leading base-S digit, exactly as in
        # construct.py: an edge inside a block at step k is still inside that
        # block at step k+1, because expansion appends digits on the right
        colour = seed_colours[digit]
        block = (np.zeros(n, np.int32) if seed_blocks is None
                 else np.asarray(seed_blocks, np.int32)[digit])
        before = len(u)
        u, v = close_triangles(u, v, n, colour, block, int(alphas[step - 1] * n), rng, beta)
        trace.append(dict(step=step, n=n, product=before, closure=len(u) - before, total=len(u)))
    assert (colour[u] == colour[v]).sum() == 0, "CHROMATIC VIOLATION"
    return u.astype(np.int32), v.astype(np.int32), colour, block, trace, 'ok'


# =============================================================================
# MEASUREMENT -- identical for generated and real graphs
# =============================================================================

def measure(G, label, extra=None, apl_sources=192, rng_seed=11,
            max_null_edges=400_000):
    """Topology of an undirected simple graph.  APL/diameter are sampled on the
    giant component when the graph is large, and the sampling is recorded."""
    t0 = time.time()
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = np.array([d for _, d in G.degree()], float)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    gc = G.subgraph(comps[0])
    exact = gc.number_of_nodes() <= 1500
    if exact:
        apl = nx.average_shortest_path_length(gc); diam = nx.diameter(gc); sampled = False
    else:
        rng = np.random.default_rng(rng_seed)
        srcs = rng.choice(sorted(gc.nodes()), size=min(apl_sources, gc.number_of_nodes()),
                          replace=False)
        tot = cnt = 0; ecc = 0
        for s in srcs:
            d = nx.single_source_shortest_path_length(gc, int(s))
            tot += sum(d.values()); cnt += len(d) - 1; ecc = max(ecc, max(d.values()))
        apl = tot / max(cnt, 1); diam = ecc; sampled = True
    from networkx.algorithms.community import louvain_communities, modularity
    comm = louvain_communities(G, seed=7)
    # degree-preserving null for the small-world ratios
    null_ok = m <= max_null_edges
    try:
        if not null_ok: raise RuntimeError('graph too large for a null ensemble')
        H = G.copy()
        nswap = min(10 * m, 2_000_000)      # 10*m is unaffordable above ~300k edges
        nx.double_edge_swap(H, nswap=nswap, max_tries=20 * nswap, seed=rng_seed)
        hgc = H.subgraph(max(nx.connected_components(H), key=len))
        # the exact/sampled choice must be made for H on H's OWN giant
        # component: swapping does not preserve component structure, and
        # deciding it from G's gc can compare a 432-node apl against a
        # 1728-node apl_r
        exact_h = hgc.number_of_nodes() <= 1500
        if exact_h:
            apl_r = nx.average_shortest_path_length(hgc)
        else:
            rng = np.random.default_rng(rng_seed)
            srcs = rng.choice(sorted(hgc.nodes()), size=min(apl_sources, hgc.number_of_nodes()),
                              replace=False)
            tot = cnt = 0
            for s in srcs:
                d = nx.single_source_shortest_path_length(hgc, int(s))
                tot += sum(d.values()); cnt += len(d) - 1
            apl_r = tot / max(cnt, 1)
        clust_r = nx.average_clustering(H)
    except Exception:
        apl_r = clust_r = float('nan')
    clust = nx.average_clustering(G)
    out = dict(label=label, n=n, m=m, density=nx.density(G),
               meandeg=float(deg.mean()), degsd=float(deg.std(ddof=1)),
               degcv=float(deg.std(ddof=1) / deg.mean()) if deg.mean() else None,
               dmin=int(deg.min()), dmax=int(deg.max()),
               n_components=len(comps), giant_frac=len(comps[0]) / n,
               clustering=clust, transitivity=nx.transitivity(G),
               modularity=modularity(G, comm), n_communities=len(comm),
               apl=apl, apl_sampled=sampled,
               # An exact diameter and a sampled lower bound must not share a
               # field name: Part 1's large graphs report max eccentricity over
               # 192 sources, Part 2's small graphs report nx.diameter.  Exactly
               # one of the next two keys is populated in any given row.
               diameter_exact=(int(diam) if not sampled else None),
               diameter_lower_bound=(int(diam) if sampled else None),
               diameter=int(diam),   # convenience; check apl_sampled first
               assortativity=nx.degree_assortativity_coefficient(G),
               null_computed=bool(null_ok and apl_r == apl_r and apl_r > 0),
               null_gc_frac=(hgc.number_of_nodes() / n) if null_ok and apl_r == apl_r else None,
               L_over_Lrand=(apl / apl_r) if (apl_r == apl_r and apl_r > 0) else None,
               C_over_Crand=(clust / clust_r) if clust_r == clust_r and clust_r else None,
               secs=round(time.time() - t0, 1))
    if out['L_over_Lrand'] and out['C_over_Crand']:
        out['sigma_dp'] = out['C_over_Crand'] / out['L_over_Lrand']
    if extra: out.update(extra)
    return out


def _clean(o):
    """json.dumps emits bare NaN, which is not valid JSON -- strict readers
    (JS, Go, R jsonlite, pandas) reject the whole file.  Assortativity is NaN
    for every regular graph, and a pure tensor power of a regular seed is
    regular, so this would hit most of the sweep."""
    if isinstance(o, float):
        return None if (o != o or o in (float('inf'), float('-inf'))) else o
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(x) for x in o]
    return o


def append(fname, rec):
    with open(os.path.join(RESULTS, fname), 'a') as f:
        f.write(json.dumps(_clean(rec), allow_nan=False) + '\n'); f.flush()


def done_keys(fname):
    p = os.path.join(RESULTS, fname)
    if not os.path.exists(p): return set()
    out = set()
    for line in open(p):
        try:
            r = json.loads(line)
            # an 'error' row is NOT done: a transient failure must not be frozen
            # in for every future restart of a job meant to run for days
            if r.get('status') != 'error': out.add(r.get('key'))
        except Exception: pass
    return out


# =============================================================================
# PART 1 -- the grammar sweep
# =============================================================================

def part1(args):
    FN = 'part1_grammar.jsonl'
    seen = done_keys(FN)
    print(f"PART 1: grammar sweep.  {len(seen)} configurations already done.", flush=True)
    print("""
  AXES:  chi (measured, 3-6)          seed size n_s (6-20)
         seed family (6)              seed rng (3)
         colour balance (2)           blocks (1, 2, 4)
         closure budget alpha (4)     degree bias beta (4)
         closure SCHEDULE (4): uniform / front / back / last_only
                               -- same total budget, different shape.  An edge
                               closed early is tensor-multiplied by every later
                               step, so shape matters as much as size.

  The full grid is ~170,000 cells, far more than can be run.  Configurations
  are visited in a SEEDED SHUFFLE, so any prefix of the output is an unbiased
  random sample of the whole space -- which is what mapping a reachable region
  needs, unlike an exhaustive pass that would finish one corner and none of the
  others.  Stop it whenever; what exists is a valid sample.
""", flush=True)
    KINDS = ['random_sparse', 'random_medium', 'random_dense',
             'complete_multipartite', 'clique_chain', 'star_of_cliques']
    # chi = 2 is excluded.  A 2-colourable seed gives a bipartite product: no
    # odd cycles, no triangles, clustering identically zero -- and closure
    # cannot fire at all, since both neighbours of a wedge centre lie outside
    # its colour class and with one other class they always match.  That is a
    # tautology about bipartite graphs rather than a property of this grammar,
    # and two causal phases is not a useful substrate in any case.  The 117
    # chi=2 rows already measured stay in results/ for the record; nothing
    # further is spent on them.
    CHIS = [3, 4, 5, 6]
    NS = [6, 7, 8, 10, 12, 14, 16, 20]
    ALPHAS = [('none', 0.0), ('light', 1.0), ('medium', 4.0), ('heavy', 15.0)]
    BETAS = [0.0, 0.75, 1.5, 3.0]          # degree bias in the closure sampler

    # HOW the closure budget is spread over the steps, not just how much of it
    # there is.  This matters more than the total: an edge closed at step 1 is
    # tensor-multiplied by every later step, so the same budget spent early or
    # late gives very different graphs.  (Spending it early is what saturated
    # the density budget in the 7-vertex seed-swap experiment.)
    SCHEDULES = ['uniform', 'front', 'back', 'last_only']

    # colour class sizes.  'balanced' splits n_s as evenly as chi allows;
    # 'skewed' makes one class about twice the others, as the paper's own
    # 20-vertex seed does with (6,6,8).
    BALANCE = ['balanced', 'skewed']

    # block structure.  construct.py's seed carries FOUR blocks and closure is
    # restricted to stay inside one; with a single block that filter is
    # vacuous.  Blocks are what produce modular / bilateral organisation, so
    # this is a first-class axis, not a detail.
    NBLOCKS = [1, 2, 4]

    SEEDS = [1, 2, 3]
    n_done = 0
    # Iterate in a DETERMINISTIC SHUFFLE rather than nested-loop order.  With
    # chi outermost the entire chi=2 block -- which is degenerate, since
    # closure cannot fire with two colours -- runs before any chi=3 result
    # appears.  For a job meant to deliver results as it goes, early output
    # must span the axes.  The shuffle is seeded, so the order is reproducible
    # and the resume logic is unaffected.
    grid = list(itertools.product(CHIS, NS, KINDS, SEEDS, BALANCE, NBLOCKS))
    np.random.default_rng(20260923).shuffle(grid)
    for chi, n_s, kind, seed_rng, balance, nblocks in grid:
        rng0 = np.random.default_rng(1000 * chi + 10 * n_s + seed_rng)
        made = make_seed(kind, n_s, chi, rng0, balance, nblocks)
        if made is None: continue
        s_edges, s_cols, chi_true, s_blocks = made    # chi_true MEASURED, exact
        # pick the number of steps so the parent lands under the node cap
        # The step count must respect the EDGE budget as well as the node
        # budget.  The pure product has m_k = 2^{k-1} m_s^k, which grows far
        # faster than n_k = n_s^k, so a small dense seed can blow past a
        # million edges while still looking small in vertices.
        m_s = len(s_edges)
        steps = 1
        while (n_s ** (steps + 1)) <= args.max_nodes and \
              (2 ** steps * m_s ** (steps + 1)) <= args.max_edges:
            steps += 1
        if steps < 2: continue
        # With two colour classes closure can NEVER fire: both neighbours of a
        # wedge centre lie outside the centre's class, so with only one other
        # class they always share a colour and every candidate is rejected.
        # alpha and beta are therefore inert and all eight (alpha, beta) cells
        # are bit-identical.  Run one and record the reason.
        # a zero closure budget makes the schedule and beta irrelevant, so do
        # not run four identical copies of it
        combos = [(('none', 0.0), 0.0, 'uniform')] + [
            (al, b, sc) for al, b, sc in itertools.product(ALPHAS[1:], BETAS, SCHEDULES)]
        for (aname, a), beta, sched in combos:
            # key is labelled by the MEASURED chi, not the requested one
            key = (f"chi{chi_true}_n{n_s}_{kind}_s{seed_rng}_st{steps}"
                   f"_{aname}_b{beta}_{sched}_{balance}_bl{nblocks}")
            if key in seen: continue
            # spread the closure budget over the steps.  The TOTAL is held
            # equal across schedules so only the shape differs.
            if sched == 'uniform':     w = [1.0] * steps
            elif sched == 'front':     w = [2.0 ** (steps - i) for i in range(steps)]
            elif sched == 'back':      w = [2.0 ** i for i in range(steps)]
            else:                      w = [0.0] * (steps - 1) + [float(steps)]
            w = np.array(w); w = w / w.sum() * steps
            alphas = [float(a * x) for x in w]
            t0 = time.time()
            try:
                rng = np.random.default_rng(99)
                u, v, colour, block, trace, status = grow(
                    s_edges, s_cols, alphas, beta, steps, rng, args.max_edges,
                    seed_blocks=s_blocks)
                if status != 'ok':
                    append(FN, dict(key=key, status=status, chi_requested=chi,
                                    n_seed=n_s, seed_kind=kind, balance=balance,
                    nblocks=nblocks, schedule=sched, alphas=alphas,
                    seed_block_sizes=[int(x) for x in np.bincount(s_blocks)], seed_rng=seed_rng,
                                    steps=steps, alpha=a, beta=beta, trace=trace))
                    print(f"  [skip] {key}: {status}", flush=True); continue
                N = n_s ** steps
                G = nx.Graph(); G.add_nodes_from(range(N))
                G.add_edges_from(zip(u.tolist(), v.tolist()))
                inv = seed_invariants(s_edges, n_s, steps)
                rec = measure(G, key, extra=dict(
                    key=key, status='ok', part=1,
                    chi=chi_true, chi_requested=chi,   # chi is the MEASURED value
                    n_seed=n_s, seed_kind=kind, balance=balance,
                    nblocks=nblocks, schedule=sched, alphas=alphas,
                    seed_block_sizes=[int(x) for x in np.bincount(s_blocks)], seed_rng=seed_rng, steps=steps,
                    alpha=a, alpha_name=aname, beta=beta,
                    monochromatic=int((colour[u] == colour[v]).sum()),
                    colour_class_sizes=[int(x) for x in np.bincount(colour)],
                    closure_frac=float(sum(t['closure'] for t in trace) / max(len(u), 1)),
                    closure_inert=(chi_true == 2),
                    trace=trace, **inv))
                append(FN, rec); n_done += 1
                print(f"  {key:56s} n={rec['n']:>7,} m={rec['m']:>9,} "
                      f"<k>={rec['meandeg']:7.2f} C={rec['clustering']:.4f} "
                      f"Q={rec['modularity']:.4f} L={rec['apl']:6.3f} d={rec['diameter']:>3d} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
            except Exception as e:
                append(FN, dict(key=key, status='error', error=repr(e)[:300]))
                print(f"  [ERROR] {key}: {repr(e)[:120]}", flush=True)
    print(f"\nPART 1 complete this pass: {n_done} new configurations.", flush=True)


# =============================================================================
# PART 2 -- real graphs that are concurrently computational
# =============================================================================
#
# Selection rule: the graph's edges must express a CONSTRAINT ON SIMULTANEITY
# in the system it describes -- two adjacent elements cannot act in the same
# phase, or one must precede the other.  That is what a proper colouring bounds,
# so these are the graphs for which chromatic causal depth is meaningful.
#
#   circuit netlists       gates compute concurrently; a wire is a dependency
#   dependency graphs      a package must build after what it requires
#   dataflow / scheduling  classical precedence graphs
#   metabolic networks     reactions proceed concurrently, sharing metabolites
#
# Counter-examples deliberately included as controls: social and road networks,
# where an edge expresses no simultaneity constraint at all.

SOURCES = [
 # (name, category, url, parser)
 # --- edges express a CONSTRAINT ON SIMULTANEITY -----------------------------
 ("circuit_hamrle1", "circuit-simulation",
  "https://sparse.tamu.edu/MM/Hamrle/Hamrle1.tar.gz", "mm_tar"),
 ("circuit_oscil_dcop", "circuit-simulation",
  "https://sparse.tamu.edu/MM/Sandia/oscil_dcop_01.tar.gz", "mm_tar"),
 ("chem_process_b_dyn", "chemical-process-simulation",
  "https://sparse.tamu.edu/MM/Grund/b_dyn.tar.gz", "mm_tar"),
 # nrvis's bio-celegans is the C. elegans METABOLIC network of Duch & Arenas
 # (453 nodes, 2,025 edges), NOT the 297-node neural connectome.  Two earlier
 # versions of this list got the label wrong in two different ways: first by
 # calling it E. coli, then by calling it neural.  It is C. elegans metabolism.
 ("celegans_metabolic", "metabolic",
  "https://nrvis.com/download/data/bio/bio-celegans.zip", "zip_edges"),
 # --- CONTROLS: edges express no simultaneity constraint ----------------------
 ("p2p_gnutella08", "CONTROL-peer-to-peer",
  "https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz", "edges_gz"),
 ("ca_netscience", "CONTROL-collaboration",
  "https://nrvis.com/download/data/ca/ca-netscience.zip", "zip_edges"),
]
# NOTE ON SELECTION.  The first four are graphs whose edges constrain what may
# happen at the same time: in a circuit netlist two gates joined by a net
# cannot be driven independently in the same phase; in a process-simulation
# matrix a nonzero couples two variables that must be solved together; in a
# connectome an edge is a channel whose endpoints exchange signal.  Those are
# the graphs for which a proper colouring means something.
# The last two are deliberate CONTROLS: a peer-to-peer overlay and a
# co-authorship network, where an edge expresses no constraint on simultaneity
# whatsoever.  If the grammar's outputs sit closer to the controls than to the
# concurrency graphs, that is a negative result and must be reported as one.
#
# A previous version of this list named a C. elegans file "ecoli_metabolic"
# and called a co-authorship network "collaboration-control" as though it were
# a concurrency graph.  Both were wrong; the names now match the data.


def _download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    print(f"    downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, 'wb') as f:
        f.write(r.read())
    return dest


def _load_mm_tar(path):
    """SuiteSparse ships MatrixMarket inside a .tar.gz.  Read the .mtx, take
    the sparsity pattern as an undirected simple graph."""
    import tarfile
    with tarfile.open(path, 'r:gz') as t:
        mem = [m for m in t.getmembers() if m.name.endswith('.mtx')
               and '_coord' not in m.name and '_b.mtx' not in m.name]
        if not mem: raise RuntimeError('no .mtx in archive')
        f = io.TextIOWrapper(t.extractfile(sorted(mem, key=lambda m: -m.size)[0]),
                             errors='ignore')
        G = nx.Graph(); header = None
        for line in f:
            if line.startswith('%'): continue
            p = line.split()
            if header is None:
                header = p                      # rows cols nnz
                continue
            if len(p) < 2: continue
            a, b = int(p[0]), int(p[1])
            if a != b: G.add_edge(a, b)
    return G


def _load_edges_txt(path, comment_chars='%#'):
    G = nx.Graph()
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in comment_chars: continue
            p = line.replace(',', ' ').split()
            if len(p) < 2: continue
            try: a, b = int(float(p[0])), int(float(p[1]))
            except ValueError: continue
            if a != b: G.add_edge(a, b)
    return G


def local_graphs():
    """Graphs already on disk that belong in the comparison."""
    out = []
    bud = os.path.join(HERE, '..', 'Budapest', 'budapest_connectome.gml')
    if os.path.exists(bud):
        G = nx.Graph(nx.convert_node_labels_to_integers(nx.read_gml(bud)))
        out.append(("budapest_connectome", "connectome", G))
    q = os.path.join(_B, 'graphs', 'cnew_coarse.npz')
    if os.path.exists(q):
        d = np.load(q); G = nx.Graph(); G.add_nodes_from(range(int(d['n_total'])))
        G.add_edges_from(map(tuple, d['edges']))
        out.append(("ours_8k_quotient", "constructed", G))
    return out


def part2(args):
    FN = 'part2_realworld.jsonl'
    seen = done_keys(FN)
    print(f"PART 2: comparison graphs.  {len(seen)} already done.\n", flush=True)
    for name, cat, G in local_graphs():
        if name in seen: continue
        try:
            G = nx.Graph(G); G.remove_edges_from(nx.selfloop_edges(G))
            rec = measure(G, name, extra=dict(key=name, part=2, category=cat, source='local'))
            rec['chi_greedy'] = max(nx.coloring.greedy_color(G, strategy='DSATUR').values()) + 1
            append(FN, rec)
        except Exception as e:
            append(FN, dict(key=name, status='error', category=cat, error=repr(e)[:300]))
            print(f"  [ERROR] {name}: {repr(e)[:160]}", flush=True); continue
        print(f"  {name:28s} [{cat:22s}] n={rec['n']:>7,} m={rec['m']:>9,} "
              f"C={rec['clustering']:.4f} Q={rec['modularity']:.4f} "
              f"L={rec['apl']:6.3f} chi_greedy={rec['chi_greedy']}", flush=True)
    for name, cat, url, kind in SOURCES:
        if name in seen: continue
        try:
            ext = {'zip_edges': '.zip', 'gml': '.gml',
                   'mm_tar': '.tar.gz', 'edges_gz': '.txt.gz'}.get(kind, '.txt')
            dest = os.path.join(GRAPHS, name + ext)
            _download(url, dest)
            if kind == 'gml':
                G = nx.Graph(nx.convert_node_labels_to_integers(nx.read_gml(dest)))
            elif kind == 'mm_tar':
                G = _load_mm_tar(dest)
            elif kind == 'edges_gz':
                G = _load_edges_txt(dest)
            elif kind == 'zip_edges':
                import zipfile
                with zipfile.ZipFile(dest) as z:
                    cand = [x for x in z.namelist() if x.endswith(('.edges', '.mtx', '.txt'))]
                    if not cand: raise RuntimeError('no edge file in zip')
                    z.extract(cand[0], GRAPHS)
                    G = _load_edges_txt(os.path.join(GRAPHS, cand[0]))
            else:
                G = _load_edges_txt(dest)
            G = nx.Graph(G); G.remove_edges_from(nx.selfloop_edges(G))
            if G.number_of_nodes() == 0: raise RuntimeError('empty graph')
            rec = measure(G, name, extra=dict(key=name, part=2, category=cat,
                                              source=url))
            rec['chi_greedy'] = max(nx.coloring.greedy_color(G, strategy='DSATUR').values()) + 1
            append(FN, rec)
            print(f"  {name:28s} [{cat:22s}] n={rec['n']:>7,} m={rec['m']:>9,} "
                  f"C={rec['clustering']:.4f} Q={rec['modularity']:.4f} "
                  f"L={rec['apl']:6.3f} chi_greedy={rec['chi_greedy']}", flush=True)
        except Exception as e:
            append(FN, dict(key=name, status='error', category=cat, error=repr(e)[:300]))
            print(f"  [ERROR] {name}: {repr(e)[:160]}", flush=True)


# =============================================================================
# PART 3 -- THE PRUNE-REPAIR GRAMMAR, SMALL SEEDS, SAMPLED NOT GRIDDED
# =============================================================================
#
# A DIFFERENT GRAMMAR from Part 1, and the difference is the point.
#
#   Part 1 (construct.py, the 8k line):   tensor -> ADD legal closures
#   Part 3 (the local line):              tensor -> PRUNE hard -> ADD back
#                                                 -> repair low degree
#
# Pruning is what the 8k grammar does not have, and it is what lets a DENSE
# seed work at all.  The product's mean degree is <k>_seed ** steps, so a seed
# much denser than about 0.2 saturates its own density budget from the product
# alone and leaves no room for closure -- measured directly when the 7-vertex
# seed (density 0.43) was put through the 8k pipeline and had to have closure
# confined to the final step.  Pruning removes that constraint by throwing the
# product's edges away and rebuilding to a chosen budget.
#
# So this cycle asks: with pruning available, do SMALL seeds reach a region the
# 8k grammar cannot?
#
# The operators are imported from emergence_localrepair_standalone.py, not
# reimplemented: triangle_prune_local, add_chromatic_to_budget, edge_budget.
#
# SAMPLED, NOT FACTORIAL.  The parameters are drawn at random from ranges
# rather than crossed in a grid.  A grid over eight axes is mostly waste when
# the goal is to map a region: random draws cover the interior, and every draw
# is an independent sample, so any prefix of the output is usable.

def part3(args):
    FN = 'part3_prune.jsonl'
    seen = done_keys(FN)
    print(f"PART 3: prune-repair grammar, small seeds, SAMPLED.  {len(seen)} done.\n", flush=True)
    print("  tensor -> prune to prune_frac * budget -> add back to add_frac * budget")
    print("  budget = n * sqrt((n-1)/2)   [the local construction's definition]")
    print("  parameters are DRAWN, not crossed.  Ctrl-C whenever; the sample stands.\n", flush=True)
    rng_master = np.random.default_rng(args.sample_seed)
    n_done = 0
    while True:
        n_s   = int(rng_master.choice([5, 6, 7, 8, 9, 10]))          # SMALL seeds
        chi   = int(rng_master.choice([3, 4, 5]))
        kind  = str(rng_master.choice(['random_sparse', 'random_medium',
                                       'random_dense', 'complete_multipartite',
                                       'clique_chain', 'star_of_cliques']))
        sdrng = int(rng_master.integers(1, 9))
        prune = float(rng_master.choice([0.05, 0.10, 0.15, 0.25, 0.40, 0.60]))
        add   = float(prune + rng_master.choice([0.0, 0.05, 0.08, 0.15, 0.30]))
        steps = int(rng_master.choice([3, 4, 5]))
        if n_s ** steps > args.max_nodes: continue
        key = f"p3_n{n_s}_chi{chi}_{kind}_s{sdrng}_st{steps}_pr{prune}_ad{round(add,3)}"
        if key in seen: continue
        made = make_seed(kind, n_s, chi, np.random.default_rng(97 * sdrng + n_s))
        if made is None: continue
        s_edges, s_cols, chi_true, _ = made
        t0 = time.time()
        try:
            n = n_s
            edges = np.array(s_edges, dtype=np.int64)
            cols = s_cols.copy()
            trace = []
            for k in range(2, steps + 1):
                n, edges = tensor_lr(n, edges, n_s, np.array(s_edges, dtype=np.int64))
                cols = project_cols(cols, n_s, s_cols, k, n_s)
                b = edge_budget(n)
                m_prod = len(edges)
                edges = prune_local(n, edges, cols, int(round(b * prune)))
                m_pruned = len(edges)
                edges = add_chromatic(n, edges, cols, int(round(b * add)))
                trace.append(dict(step=k, n=n, product=m_prod, after_prune=m_pruned,
                                  after_add=len(edges), budget=b))
                if len(edges) > args.max_edges:
                    raise RuntimeError('too many edges after add-back')
            mono = int((cols[edges[:, 0]] == cols[edges[:, 1]]).sum())
            G = nx.Graph(); G.add_nodes_from(range(n))
            G.add_edges_from((int(a), int(b_)) for a, b_ in edges)
            rec = measure(G, key, extra=dict(
                key=key, status='ok', part=3, grammar='prune_repair',
                chi=chi_true, chi_requested=chi, n_seed=n_s, seed_kind=kind,
                seed_rng=sdrng, steps=steps, prune_frac=prune, add_frac=add,
                monochromatic=mono, trace=trace,
                **seed_invariants(s_edges, n_s, steps)))
            append(FN, rec); n_done += 1
            print(f"  {key:54s} n={rec['n']:>7,} m={rec['m']:>9,} "
                  f"C={rec['clustering']:.4f} Q={rec['modularity']:.4f} "
                  f"L={rec['apl']:6.3f} mono={mono} [{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:
            append(FN, dict(key=key, status='error', error=repr(e)[:300]))
            print(f"  [skip] {key}: {repr(e)[:90]}", flush=True)


def project_cols(cols, n_s, s_cols, step, S):
    """Colour labels ride the LEADING base-S digit, as in both grammars."""
    n = S ** step
    digit = (np.arange(n) // (S ** (step - 1))) % S
    return s_cols[digit]


# =============================================================================
def selftest():
    """Checks that must pass before any sweep is trusted."""
    print("1. exact chromatic number on graphs with known chi")
    T = [("C5", nx.cycle_graph(5), 3), ("C6", nx.cycle_graph(6), 2),
         ("Petersen", nx.petersen_graph(), 3), ("K5", nx.complete_graph(5), 5),
         ("K_{3,3}", nx.complete_bipartite_graph(3, 3), 2),
         ("Grotzsch", nx.mycielski_graph(4), 4), ("C7", nx.cycle_graph(7), 3)]
    bad = 0
    for nm, G, want in T:
        got = chromatic_number_exact(G)
        if got != want: bad += 1
        print(f"     {nm:10s} want {want} got {got} {'ok' if got==want else 'FAIL'}")
    print("\n2. every seed is labelled by its MEASURED chromatic number")
    mis = tot = 0
    for kind in ['random_sparse', 'random_medium', 'random_dense',
                 'complete_multipartite', 'clique_chain', 'star_of_cliques']:
        for chi in (2, 3, 4, 5, 6):
            for n_s in (6, 8, 10, 12, 16, 20):
                r = make_seed(kind, n_s, chi, np.random.default_rng(7))
                if r is None: continue
                e, c, ct, bl = r; tot += 1
                G = nx.Graph(e); G.add_nodes_from(range(n_s))
                if chromatic_number_exact(G) != ct or max(c) + 1 != ct: mis += 1
    print(f"     {tot} seeds built, {mis} with a label that does not match measurement")
    print("\n3. the product law <k> = <k>_seed ** steps")
    r = make_seed('random_sparse', 8, 3, np.random.default_rng(7))
    e, c, ct, bl = r
    inv = seed_invariants(e, 8, 3)
    u, v, col, blk, tr, st = grow(e, c, [0, 0, 0], 1.5, 3, np.random.default_rng(1))
    got = 2 * len(u) / 8 ** 3
    print(f"     predicted {inv['product_meandeg']:.4f}   measured {got:.4f}   "
          f"{'ok' if abs(got-inv['product_meandeg'])<1e-6 else 'FAIL'}")
    print(f"     predicted m {inv['product_m']:,}   measured m {len(u):,}   "
          f"{'ok' if len(u)==inv['product_m'] else 'FAIL'}")
    print(f"     monochromatic edges {int((col[u]==col[v]).sum())}")
    print(f"\n{'ALL CHECKS PASS' if bad==0 and mis==0 else 'SOME CHECKS FAILED'}")


def summary(args):
    """Expressiveness is the REACHABLE REGION, not the marginal range of each
    property.  Two properties can each span widely while being perfectly
    correlated, in which case the grammar has one degree of freedom, not two.
    So this reports marginal spans, the correlation structure, and the
    effective dimension of the reachable set."""
    import itertools as it
    for fn in ('part1_grammar.jsonl', 'part2_realworld.jsonl'):
        p_ = os.path.join(RESULTS, fn)
        if not os.path.exists(p_): print(f"{fn}: not started"); continue
        rows = [json.loads(l) for l in open(p_)]
        ok = [r for r in rows if 'clustering' in r]
        print(f"\n{'='*74}\n{fn}: {len(rows)} lines, {len(ok)} measured graphs\n{'='*74}")
        if not ok: continue
        KEYS = ['density', 'clustering', 'transitivity', 'modularity', 'apl',
                'degcv', 'assortativity']
        print("MARGINAL SPANS")
        for k in KEYS:
            v = [r[k] for r in ok if r.get(k) is not None and r[k] == r[k]]
            if v: print(f"   {k:16s} min {min(v):9.4f}  max {max(v):9.4f}  span {max(v)-min(v):9.4f}")
        good = [r for r in ok if all(r.get(k) is not None and r[k] == r[k]
                                     for k in KEYS)]
        M = np.array([[r[k] for k in KEYS] for r in good], float)
        if len(good) < len(ok):
            print(f"\n   ({len(ok)-len(good)} of {len(ok)} rows excluded from the joint "
                  f"analysis: a property is missing or NaN.  Assortativity is NaN for\n"
                  f"    regular graphs, which includes every pure tensor power of a "
                  f"regular seed.)")
        if len(M) >= 5 and np.all(M.std(0) > 1e-12):
            Z = (M - M.mean(0)) / (M.std(0) + 1e-12)
            C = np.corrcoef(Z.T)
            print("\nCORRELATION between properties (|r| > 0.9 means one knob, not two)")
            for a, b in it.combinations(range(len(KEYS)), 2):
                if abs(C[a, b]) > 0.9:
                    print(f"   {KEYS[a]:14s} vs {KEYS[b]:14s}  r = {C[a,b]:+.3f}  <-- collinear")
            ev = np.sort(np.linalg.eigvalsh(C))[::-1]
            ev = ev / ev.sum()
            cum = np.cumsum(ev)
            d90 = int(np.searchsorted(cum, 0.90) + 1)
            print(f"\nEFFECTIVE DIMENSION of the reachable set")
            print(f"   variance explained: " + "  ".join(f"{x:.2f}" for x in ev))
            print(f"   {d90} of {len(KEYS)} components carry 90% of the variance")
        if fn.startswith('part1'):
            print("""
REFERENCE POINTS.  Part 1 measures PARENTS, so the right comparison is the
paper's 8,000-vertex parent, NOT the 1,015-node Budapest quotient.  Comparing
a parent's clustering against a quotient's is a category error.

    the paper's 8k PARENT    n=8,000  m=477,584  C=0.2143  Q=0.6594  L=3.4252
    its 1,015-node QUOTIENT  n=1,015  m= 64,760  C=0.6119  Q=0.5554  L=2.1872
    Budapest (a quotient)    n=1,015  m= 70,654  C=0.6696  Q=0.5574  L=2.2190

A sweep row is comparable to the first line.  Reaching the second or third
would require coarsening the parent, which this sweep does not do.
""")
            print("WHAT THE SEED ALREADY DETERMINES (density and spectrum are not free)")
            hav = [r for r in ok if r.get('product_meandeg') is not None]
            if hav:
                rat = [r['meandeg'] / r['product_meandeg'] for r in hav
                       if r['product_meandeg'] > 0]
                print(f"   measured <k> / product <k>:  min {min(rat):.3f}  "
                      f"max {max(rat):.3f}   (1.0 = closure added nothing)")
                bychi = {}
                for r in hav: bychi.setdefault(r.get('chi'), []).append(r)
                print("   by MEASURED chi:")
                for c in sorted(x for x in bychi if x is not None):
                    g = bychi[c]
                    print(f"     chi={c}: {len(g):4d} graphs   "
                          f"C {min(x['clustering'] for x in g):.3f}-{max(x['clustering'] for x in g):.3f}   "
                          f"Q {min(x['modularity'] for x in g):.3f}-{max(x['modularity'] for x in g):.3f}")
        else:
            print("\nBY CATEGORY (controls are marked CONTROL and should differ)")
            for r in sorted(ok, key=lambda x: str(x.get('category'))):
                print(f"   {str(r.get('category')):30s} {r['label'][:22]:22s} "
                      f"n={r['n']:>7,} C={r['clustering']:.4f} Q={r['modularity']:.4f} "
                      f"L={r['apl']:6.3f} chi_greedy={r.get('chi_greedy','?')}")
    print("""
NOTE ON SIZE.  Part 1 parents span roughly 256 to 16,807 vertices -- a 66x
range -- because the step count is chosen to fit under BOTH the node and the
edge budget, and the edge budget depends on the seed.  That confounds
size with grammar for any property that scales with n -- APL above all.  Use
L_over_Lrand rather than apl when comparing across seed sizes, and compare
Part 1 against Part 2 only at comparable n.
""")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--part', type=int, choices=[1, 2, 3])
    ap.add_argument('--sample-seed', type=int, default=4242,
                    help='part 3 draws parameters at random; this seeds the draw')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--selftest', action='store_true',
                    help='verify chi, the grammar and the seed families, then exit')
    ap.add_argument('--max-nodes', type=int, default=20000)
    ap.add_argument('--max-edges', type=int, default=3_000_000)
    a = ap.parse_args()
    if a.selftest: selftest(); raise SystemExit(0)
    if a.summary: summary(a)
    elif a.part == 1: part1(a)
    elif a.part == 2: part2(a)
    elif a.part == 3: part3(a)
    else: ap.print_help()
