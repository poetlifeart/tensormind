#!/usr/bin/env python3
"""
emergence_localrepair_standalone.py
-----------------------------------
ONE self-contained file that builds the LOCAL-REPAIR graph end to end and
verifies it. No imports from any other project file. It:

  1. builds the FINE graph  (16,807-node parent, tensor -> delete -> add, 4 steps)
  2. coarsens to the END graph (1,015-node quotient, Louvain + density-merge)
  3. VERIFIES both against the recorded numbers (parent edges + 7 quotient stats)
  4. saves: localrepair_parent.npz, localrepair_quotient.npz,
            localrepair_quotient_audit.npz  (colorless layers, ready for the audit)

RECIPE
  seed        = modular (G2, degrees 2-3)
  steps       = 4                       (parent = 7^5 = 16,807 nodes)
  prune-frac  = 0.10                    (delete to 0.10 * gmean, keep top-triangle edges)
  add-frac    = 0.18                    (add chromatic triangle-closing edges to 0.18 * gmean)
  repair      = LOCAL 2-hop             (attach low-degree nodes inside their own
                                         neighbourhood -> keeps modularity + a low tail)
  resolution  = 40, target 1015 nodes   (coarsening)

DETERMINISTIC given: python 3.13.5, numpy 2.3.4, networkx 3.5 (pin networkx).
Run:  python3 emergence_localrepair_standalone.py
"""
import math, time, os
from collections import defaultdict
import numpy as np, networkx as nx
from networkx.algorithms.community import louvain_communities, modularity

HERE = os.path.dirname(os.path.abspath(__file__))
BUDAPEST = os.path.join(HERE,
                        '..', '..', 'Budapest', 'budapest_connectome.gml')
# original (pre-tensormind): '/home/vahid/Desktop/finalthree/budapest_connectome.gml'
COLORS = np.array([0,0,1,1,2,2,2], dtype=np.int32)
SEED   = np.array([[0,2],[0,5],[1,3],[1,4],[1,6],[2,5],[2,6],[3,4],[3,6]], dtype=np.int64)
PRUNE_FRAC, ADD_FRAC, RES, TARGET = 0.10, 0.18, 40, 1015

# expected record (from reproduce_champions.py, verified) -- what a correct build must give
EXPECT = dict(parent_edges=277319, n=1015, density=0.130, std=68, dmin=3, dmax=502,
              ks=0.111, Q=0.52)

# ---------------- construction primitives (inlined) ----------------
def tensor_product(n_m, edges_m, n_s, edges_s):
    n = n_m*n_s; ne_m, ne_s = len(edges_m), len(edges_s)
    if ne_m==0 or ne_s==0: return n, np.empty((0,2),dtype=np.int64)
    um=np.repeat(edges_m[:,0],ne_s); vm=np.repeat(edges_m[:,1],ne_s)
    us=np.tile(edges_s[:,0],ne_m);   vs=np.tile(edges_s[:,1],ne_m)
    x1=um*n_s+us; y1=vm*n_s+vs; x2=um*n_s+vs; y2=vm*n_s+us
    e1=np.stack([np.minimum(x1,y1),np.maximum(x1,y1)],axis=1)
    e2=np.stack([np.minimum(x2,y2),np.maximum(x2,y2)],axis=1); e2=e2[e2[:,0]!=e2[:,1]]
    return n, np.unique(np.vstack([e1,e2]),axis=0)

def project_colors(cm, n_s):
    c=np.empty(len(cm)*n_s,dtype=np.int32)
    for i in range(len(cm)): c[i*n_s:(i+1)*n_s]=cm[i]
    return c

def edge_budget(n): return int(round(n*math.sqrt((n-1)/2)))

def triangle_prune_local(n, edges, colors, target):
    adj=defaultdict(set)
    for u,v in edges: u,v=int(u),int(v); adj[u].add(v); adj[v].add(u)
    scores=np.array([len(adj[int(u)]&adj[int(v)])+(len(adj[int(u)])+len(adj[int(v)]))*1e-7 for u,v in edges])
    idx=np.argpartition(scores,-target)[-target:]; kept=edges[idx]
    edge_set=set(); ak=defaultdict(set)
    for u,v in kept: u,v=int(u),int(v); edge_set.add((min(u,v),max(u,v))); ak[u].add(v); ak[v].add(u)
    deg=np.array([len(ak[i]) for i in range(n)],dtype=np.int32)
    by_color={c:np.where(colors==c)[0] for c in range(3)}; rng=np.random.default_rng(42)
    for node in np.where(deg<3)[0]:
        node=int(node); mc=int(colors[node])
        # LOCAL: attach to a 2-hop cross-colour neighbour (closes a triangle, stays in-community)
        two=set()
        for w in ak[node]: two|=ak[w]
        two-=ak[node]; two.discard(node)
        loc=[v for v in two if colors[v]!=mc and (min(node,v),max(node,v)) not in edge_set]
        rng.shuffle(loc)
        for v in loc:
            if deg[node]>=3: break
            e=(min(node,v),max(node,v)); edge_set.add(e); ak[node].add(v); ak[v].add(node); deg[node]+=1; deg[v]+=1
        # fallback to random cross-colour only if 2-hop ran out (rare)
        pool=np.concatenate([by_color[c] for c in range(3) if c!=mc]); att=0
        while deg[node]<3 and att<150:
            v=int(pool[rng.integers(len(pool))]); e=(min(node,v),max(node,v))
            if e not in edge_set: edge_set.add(e); ak[node].add(v); ak[v].add(node); deg[node]+=1; deg[v]+=1
            att+=1
    G=nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(edge_set)
    comps=sorted(nx.connected_components(G),key=len,reverse=True)
    if len(comps)>1:
        main=comps[0]
        for comp in comps[1:]:
            cl=list(comp); rng.shuffle(cl)
            for u in cl:
                cand=[v for v in main if colors[v]!=colors[u]][:200]; rng.shuffle(cand); done=False
                for v in cand[:20]:
                    e=(min(u,v),max(u,v))
                    if e not in edge_set: edge_set.add(e); done=True; break
                if done: main=main|comp; break
    return np.array(sorted(edge_set),dtype=np.int64)

def add_chromatic_to_budget(n, edges, colors, target):
    need=target-len(edges)
    if need<=0: return edges
    rng=np.random.default_rng(42); adj=defaultdict(set); eset=set()
    for u,v in edges: u,v=int(u),int(v); adj[u].add(v); adj[v].add(u); eset.add((min(u,v),max(u,v)))
    cand={}; nodes=list(range(n)); rng.shuffle(nodes)
    for v in nodes:
        nb=list(adj[v])
        if len(nb)<2: continue
        s=nb if len(nb)<=60 else list(rng.choice(nb,60,replace=False))
        for i in range(len(s)):
            for j in range(i+1,len(s)):
                u,w=int(s[i]),int(s[j])
                if colors[u]==colors[w]: continue
                e=(min(u,w),max(u,w))
                if e in eset: continue
                cand[e]=cand.get(e,0)+1
        if len(cand)>need*5: break
    for e in sorted(cand,key=lambda e:-cand[e])[:need]: eset.add(e)
    return np.array(sorted(eset),dtype=np.int64)

def build_fine():
    n=7; edges=SEED.copy(); c=COLORS.copy()
    for _ in range(4):
        n,edges=tensor_product(n,edges,7,SEED); c=project_colors(c,7); b=edge_budget(n)
        edges=triangle_prune_local(n,edges,c,int(round(b*PRUNE_FRAC)))
        edges=add_chromatic_to_budget(n,edges,c,int(round(b*ADD_FRAC)))
    return n, edges

def coarsen(n, edges):
    G=nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from((int(u),int(v)) for u,v in edges)
    cm=[set(x) for x in louvain_communities(G,resolution=RES,seed=42)]; n2c={nd:ci for ci,x in enumerate(cm) for nd in x}
    W=defaultdict(int)
    for u,v in G.edges():
        a,b=n2c[u],n2c[v]
        if a!=b: W[(min(a,b),max(a,b))]+=1
    nbr=defaultdict(dict)
    for (a,b),w in W.items(): nbr[a][b]=w; nbr[b][a]=w
    size=[len(x) for x in cm]; alive=set(range(len(cm))); mi={}
    while len(alive)>TARGET and len(alive)>2:
        a=min(alive,key=lambda i:size[i])
        if not nbr[a]: alive.discard(a); continue
        b=max(nbr[a],key=lambda x:nbr[a][x]/(size[a]*size[x]))
        for c,w in list(nbr[a].items()):
            del nbr[c][a]
            if c==b: continue
            if c in nbr[b]: nbr[b][c]+=w; nbr[c][b]+=w
            else: nbr[b][c]=w; nbr[c][b]=w
        size[b]+=size[a]; nbr[a]={}; alive.discard(a); mi[a]=b
    def root(x):
        while x in mi: x=mi[x]
        return x
    q={}; fid={}; k=0
    for nd in range(n):
        r=root(n2c[nd])
        if r not in fid: fid[r]=k; k+=1
        q[nd]=fid[r]
    Gq=nx.Graph(); Gq.add_nodes_from(range(k))
    for u,v in G.edges():
        a,b=q[u],q[v]
        if a!=b: Gq.add_edge(a,b)
    return Gq

def main():
    t0=time.time()
    print("Building FINE graph (16,807-node parent)...")
    n, edges = build_fine()
    print(f"  fine graph: {n} nodes, {len(edges)} edges  [{time.time()-t0:.0f}s]")

    print("Coarsening to END graph (1,015-node quotient)...")
    Gq = coarsen(n, edges); nq = Gq.number_of_nodes()
    dg = np.array([d for _,d in Gq.degree()])
    Q  = modularity(Gq, louvain_communities(Gq, seed=42))
    dens = 2*Gq.number_of_edges()/(nq*(nq-1))

    # KS vs Budapest
    B=nx.read_gml(BUDAPEST); DB=np.array([d for _,d in B.degree()])
    g=np.arange(0,max(dg.max(),DB.max())+1)
    ks=float(np.max(np.abs(np.searchsorted(np.sort(dg),g,side='right')/len(dg)
                          -np.searchsorted(np.sort(DB),g,side='right')/len(DB))))
    print(f"  end graph: {nq} nodes, {Gq.number_of_edges()} edges, density {dens:.3f}, "
          f"mean {dg.mean():.0f}, std {dg.std():.0f}, min {dg.min()}, max {dg.max()}, KS {ks:.3f}, Q {Q:.3f}")

    # save the three artefacts
    np.savez(os.path.join(HERE, 'localrepair_parent.npz'),   n_total=n,  edges=edges)
    np.savez(os.path.join(HERE, 'localrepair_quotient.npz'), n_total=nq, edges=np.array(sorted(Gq.edges()),dtype=np.int64))
    np.savez(os.path.join(HERE, 'localrepair_quotient_audit.npz'), n_total=nq,
             edges=np.array(sorted(Gq.edges()),dtype=np.int64), n1=nq, n2=0, n3=0)  # colorless -> generic null
    print("  saved localrepair_parent.npz / localrepair_quotient.npz / localrepair_quotient_audit.npz")

    # VERIFY
    def close(a,b,t): return abs(a-b)<=t
    checks=[('fine parent edges', len(edges)==EXPECT['parent_edges'], f"{len(edges)} vs {EXPECT['parent_edges']}"),
            ('end n',        nq==EXPECT['n'],                   f"{nq} vs {EXPECT['n']}"),
            ('end density',  close(dens,EXPECT['density'],0.004), f"{dens:.3f} vs {EXPECT['density']}"),
            ('end std',      close(dg.std(),EXPECT['std'],2),   f"{dg.std():.1f} vs {EXPECT['std']}"),
            ('end min',      dg.min()==EXPECT['dmin'],          f"{dg.min()} vs {EXPECT['dmin']}"),
            ('end max',      dg.max()==EXPECT['dmax'],          f"{dg.max()} vs {EXPECT['dmax']}"),
            ('end KS',       close(ks,EXPECT['ks'],0.004),      f"{ks:.3f} vs {EXPECT['ks']}"),
            ('end Q',        close(Q,EXPECT['Q'],0.02),         f"{Q:.3f} vs {EXPECT['Q']}")]
    print("\nVERIFICATION")
    for name,ok,detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:18s} {detail}")
    print("="*46)
    print("ALL PASS -- graph reproduced." if all(c[1] for c in checks) else "MISMATCH -- see above.")
    print(f"total {time.time()-t0:.0f}s")

if __name__=='__main__':
    main()
