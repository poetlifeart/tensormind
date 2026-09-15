"""Big symmetric swap: make the parent say exactly what the quotient says.

  1. every hidden pair (not a quotient edge) loses its parent edges outright
  2. every shown pair is set to a target multiplicity drawn rank-for-rank from
     Budapest's fibre counts -- trimmed if too heavy, DENSIFIED if too light
  3. whatever budget is left goes inside supernodes, where it buys clustering
All additions are cross-colour; chi=3, connectivity and non-bipartiteness verified.
"""
import numpy as np, networkx as nx, pickle, time, os
from collections import defaultdict

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, '..', '..', '..'))
def _p(name):
    """Resolve a data file: next to the script, in data/, or elsewhere in the repo."""
    for c in (_os.path.join(_HERE, name),
              _os.path.join(_HERE, 'inputs', name),
              _os.path.join(_REPO, 'graphnets', 'construction', 'global', name),
              _os.path.join(_REPO, 'graphnets', 'Budapest', name)):
        if _os.path.exists(c):
            return c
    return _os.path.join(_HERE, name)

t0=time.time(); rng=np.random.default_rng(42)
KEEP_PARENT_SIZE = os.environ.get('KEEP_SIZE','1')=='1'
d=np.load(_p('graph_brain_mild.npz'))
n1,n2,n3=int(d['n1']),int(d['n2']),int(d['n3']); off2,off3=n1,n1+n2; n=n1+n2+n3
G=nx.Graph(); G.add_nodes_from(range(n))
for mk,o_r,o_c in [('mask_12',off2,0),('mask_13',off3,0),('mask_23',off3,off2)]:
    r,c=np.where(d[mk]>0); G.add_edges_from((int(cc)+o_c,int(rr)+o_r) for rr,cc in zip(r,c))
col=np.empty(n,int); col[:n1]=0; col[n1:n1+n2]=1; col[n1+n2:]=2
E0=G.number_of_edges()
D=pickle.load(open(_p('v146_weights.pkl'),'rb')); wit=D['witness']; shown=set(map(tuple,D['v14']))
# the published partition, recomputed identically
from networkx.algorithms.community import louvain_communities



comms=louvain_communities(G,resolution=20.0,seed=42)
cl=[set(c) for c in sorted(comms,key=lambda c:-len(c))]
n2c={nd:ci for ci,c in enumerate(cl) for nd in c}
while len(cl)>1015:
    mi=min(range(len(cl)),key=lambda i:len(cl[i])); mc=cl[mi]
    ne=defaultdict(int)
    for nd in mc:
        for nb in G.neighbors(nd):
            k=n2c[nb]
            if k!=mi: ne[k]+=1
    bt=(max(ne,key=lambda c: ne[c]/(len(mc)*len(cl[c]))) if ne
        else min(((len(cl[i]),i) for i in range(len(cl)) if i!=mi))[1])
    cl[bt]|=mc
    for nd in mc: n2c[nd]=bt
    cl.pop(mi)
    for ci,c in enumerate(cl):
        for nd in c: n2c[nd]=ci
members=[sorted(c) for c in cl]
pe=defaultdict(list)
for u,v in G.edges():
    a,b=n2c[u],n2c[v]
    if a!=b: pe[(min(a,b),max(a,b))].append((u,v))
print(f"parent {E0:,}  pairs {len(pe):,}  shown {len(shown):,}  ({time.time()-t0:.0f}s)",flush=True)
adj={x:set(G[x]) for x in G}
# ---- 1. delete hidden pairs entirely
hidden=[p for p in pe if p not in shown]
gone=[e for p in hidden for e in pe[p]]
G.remove_edges_from(gone)
for u,v in gone: adj[u].discard(v); adj[v].discard(u)
print(f"deleted {len(gone):,} parent edges of {len(hidden):,} hidden pairs",flush=True)
# ---- 2. Budapest-matched targets for the shown pairs
bud=np.load(_p('bud_w.npy'))[0]
sp=sorted(shown, key=lambda p: wit[p])                       # ascending by current weight
tg=np.sort(rng.choice(bud, len(sp), replace=False))          # ascending Budapest sample
target={p:max(1,int(round(t))) for p,t in zip(sp,tg)}
trim=0; grow_need=0
for p in shown:
    cur=len(pe[p]); t=target[p]
    if cur>t:
        e=sorted(pe[p],key=lambda uv: len(adj[uv[0]]&adj[uv[1]]))[:cur-t]
        G.remove_edges_from(e)
        for u,v in e: adj[u].discard(v); adj[v].discard(u)
        trim+=len(e)
    elif cur<t: grow_need+=t-cur
print(f"trimmed {trim:,} from over-weight pairs; {grow_need:,} edges needed to densify under-weight pairs",flush=True)
# ---- 3. densify the under-weight shown pairs
added=0
for p in shown:
    t=target[p]; cur=sum(1 for u,v in pe[p] if G.has_edge(u,v))
    if cur>=t: continue
    A,B=members[p[0]],members[p[1]]
    cand=[(len(adj[u]&adj[v]),u,v) for u in A for v in B
          if col[u]!=col[v] and u!=v and not G.has_edge(u,v)]
    cand.sort(reverse=True)                      # triangle support first, as the pipeline does
    for sc,u,v in cand:
        if cur>=t: break
        G.add_edge(u,v); adj[u].add(v); adj[v].add(u); cur+=1; added+=1
print(f"densified: added {added:,} cross-colour parent edges  ({time.time()-t0:.0f}s)",flush=True)
# ---- 3b. the construction's own repair steps, before any surplus is spent
under=[x for x in G if len(adj[x])<3]
rep=0
for node in under:
    mc=col[node]
    two=set()
    for w_ in adj[node]: two |= adj[w_]
    two-=adj[node]; two.discard(node)
    loc=[x for x in two if col[x]!=mc]
    loc.sort(key=lambda x: -len(adj[node]&adj[x]))
    for v in loc:
        if len(adj[node])>=3: break
        G.add_edge(node,v); adj[node].add(v); adj[v].add(node); rep+=1
    while len(adj[node])<3:
        v=int(rng.integers(n))
        if col[v]!=mc and v!=node and not G.has_edge(node,v):
            G.add_edge(node,v); adj[node].add(v); adj[v].add(node); rep+=1
print(f"degree repair: +{rep:,} (nodes below degree 3: {len(under):,})",flush=True)
comps=sorted(nx.connected_components(G),key=len,reverse=True)
br=0
if len(comps)>1:
    main=comps[0]
    for comp in comps[1:]:
        for u in comp:
            done=False
            for v in list(main)[:400]:
                if col[u]!=col[v] and not G.has_edge(u,v):
                    G.add_edge(u,v); adj[u].add(v); adj[v].add(u); br+=1; done=True; break
            if done: break
        main=main|comp
print(f"connectivity repair: +{br}",flush=True)

# ---- 4. surplus into supernode interiors
if KEEP_PARENT_SIZE:
    surplus=E0-G.number_of_edges(); put=0
    order=sorted(range(len(members)),key=lambda ci:-len(members[ci]))
    for ci in order:
        if put>=surplus: break
        mem=members[ci]
        cand=[(len(adj[u]&adj[v]),u,v) for i,u in enumerate(mem) for v in mem[i+1:]
              if col[u]!=col[v] and not G.has_edge(u,v)]
        cand.sort(reverse=True)
        for sc,u,v in cand:
            if put>=surplus: break
            if not G.has_edge(u,v):
                G.add_edge(u,v); adj[u].add(v); adj[v].add(u); put+=1
    print(f"surplus into supernode interiors: {put:,}",flush=True)
mono=sum(1 for u,v in G.edges() if col[u]==col[v])
print(f"parent now {G.number_of_edges():,} (was {E0:,})")
print(f"VERIFY mono={mono} connected={nx.is_connected(G)} bipartite={nx.is_bipartite(G)}")
E=np.array(sorted((min(u,v),max(u,v)) for u,v in G.edges()),dtype=np.int64)
idx=np.full(n,-1,dtype=np.int64)
for c in range(3):
    w_=np.where(col==c)[0]; idx[w_]=np.arange(len(w_))
m12=np.zeros((n2,n1),np.uint8); m13=np.zeros((n3,n1),np.uint8); m23=np.zeros((n3,n2),np.uint8)
for u,v in E:
    cu,cv=col[u],col[v]
    if cu>cv: u,v=v,u; cu,cv=cv,cu
    if cu==0 and cv==1: m12[idx[v],idx[u]]=1
    elif cu==0 and cv==2: m13[idx[v],idx[u]]=1
    else: m23[idx[v],idx[u]]=1
np.savez(_p('parent_bigswap3_masks.npz'),n1=n1,n2=n2,n3=n3,mask_12=m12,mask_13=m13,mask_23=m23)
# quotient with the SAME partition and the SAME shown pairs
newW=[]; keptE=[]
for p in shown:
    w=sum(1 for u,v in pe[p] if G.has_edge(u,v))
    A,B=members[p[0]],members[p[1]]
    w=sum(1 for u in A for v in B if G.has_edge(u,v)) if w<target[p] else w
    newW.append(w); keptE.append(p)
np.save(_p('bigswap3_w.npy'),np.array(newW,float))
np.savez(_p('graph_bigswap3.npz'),n_total=1015,n1=338,n2=338,n3=339,
         edges=np.array(sorted(keptE),dtype=np.int64))
print(f"saved ({time.time()-t0:.0f}s)")
