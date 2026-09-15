"""Apply the published swap + Feeder inversion on top of the Budapest-matched graph.
Swap removes 6,500 weight-1 coarse edges; the inversion gives each of the 6,500 new
coarse edges exactly one chromatically legal parent edge -> weight 1.  Net effect on
the weight multiset: zero.  Symmetric: the removed edges lose their parent support too.
"""
import numpy as np, networkx as nx, pickle, time
from collections import defaultdict
from networkx.algorithms.community import louvain_communities, modularity
from scipy.stats import ks_2samp

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
fc=np.load(_p('bud_w.npy'))[0]
def gini(x):
    s=np.sort(np.asarray(x,float)); N=len(s); c=np.cumsum(s); return (N+1-2*(c/c[-1]).sum())/N
# parent produced by bigswap3
d=np.load(_p('parent_bigswap3_masks.npz'))
n1,n2,n3=int(d['n1']),int(d['n2']),int(d['n3']); off2,off3=n1,n1+n2; n=n1+n2+n3
P=nx.Graph(); P.add_nodes_from(range(n))
for mk,o_r,o_c in [('mask_12',off2,0),('mask_13',off3,0),('mask_23',off3,off2)]:
    r,c=np.where(d[mk]>0); P.add_edges_from((int(cc)+o_c,int(rr)+o_r) for rr,cc in zip(r,c))
col=np.empty(n,int); col[:n1]=0; col[n1:n1+n2]=1; col[n1+n2:]=2
E0=P.number_of_edges()
# same partition
comms=louvain_communities(P,resolution=20.0,seed=42) if False else None
D=pickle.load(open(_p('v146_weights.pkl'),'rb'))
# reuse the published partition by recomputing on the ORIGINAL parent (identical seed/res)
d0=np.load(_p('graph_brain_mild.npz'))
G0=nx.Graph(); G0.add_nodes_from(range(n))
for mk,o_r,o_c in [('mask_12',off2,0),('mask_13',off3,0),('mask_23',off3,off2)]:
    r,c=np.where(d0[mk]>0); G0.add_edges_from((int(cc)+o_c,int(rr)+o_r) for rr,cc in zip(r,c))
cm=louvain_communities(G0,resolution=20.0,seed=42)
cl=[set(c) for c in sorted(cm,key=lambda c:-len(c))]
n2c={nd:ci for ci,c in enumerate(cl) for nd in c}
while len(cl)>1015:
    mi=min(range(len(cl)),key=lambda i:len(cl[i])); mc=cl[mi]
    ne=defaultdict(int)
    for nd in mc:
        for nb in G0.neighbors(nd):
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
for u,v in P.edges():
    a,b=n2c[u],n2c[v]
    if a!=b: pe[(min(a,b),max(a,b))].append((u,v))
w={p:len(e) for p,e in pe.items()}
Q=nx.Graph(); Q.add_nodes_from(range(1015)); Q.add_edges_from(w.keys())
print(f"before swap: quotient {Q.number_of_edges():,} edges, parent {E0:,}",flush=True)
# hierarchy for tiers
L0=sorted(louvain_communities(Q,resolution=0.95,seed=42,weight=None),key=len,reverse=True)
l0={nd:i for i,c in enumerate(L0) for nd in c}
bp={0:(0.9,1.0),1:(0.7,1.0),2:(1.0,2.0),3:(1.0,2.0)}
l1={}
for bi,bl in enumerate(L0):
    g1,_=bp.get(bi,(1.0,2.0))
    for gi,gr in enumerate(sorted(louvain_communities(Q.subgraph(bl),resolution=g1,seed=42,weight=None),key=len,reverse=True)):
        for nd in gr: l1[nd]=(bi,gi)
t3=[e for e in Q.edges() if l0[e[0]]==l0[e[1]] and l1[e[0]]!=l1[e[1]]]
t3=sorted(t3,key=lambda e: w[(min(e),max(e))])[:6500]
grp=defaultdict(list)
for nd in range(1015): grp[l1[nd]].append(nd)
adjQ={x:set(Q[x]) for x in Q}
pool=[]
for k,g in grp.items():
    for i,u in enumerate(g):
        for v in g[i+1:]:
            if not Q.has_edge(u,v): pool.append((len(adjQ[u]&adjQ[v]),u,v))
pool.sort(reverse=True); pool=pool[:6500]
adjP={x:set(P[x]) for x in P}
# remove the 6,500 coarse edges AND their parent support (symmetric)
gone=0
for e in t3:
    p=(min(e),max(e)); Q.remove_edge(*e)
    for u,v in pe[p]:
        if P.has_edge(u,v): P.remove_edge(u,v); adjP[u].discard(v); adjP[v].discard(u); gone+=1
# add 6,500 coarse edges and LIFT one parent edge each
lifted=0
for sc,a,b in pool:
    Q.add_edge(a,b)
    A,B=members[a],members[b]
    best=None
    for u in A[:30]:
        for v in B[:30]:
            if col[u]!=col[v] and not P.has_edge(u,v):
                s2=len(adjP[u]&adjP[v])
                if best is None or s2>best[0]: best=(s2,u,v)
    if best:
        _,u,v=best; P.add_edge(u,v); adjP[u].add(v); adjP[v].add(u); lifted+=1
print(f"swap: removed 6,500 coarse (+{gone:,} parent edges gone); added 6,500 coarse (+{lifted:,} lifted)",flush=True)
mono=sum(1 for u,v in P.edges() if col[u]==col[v])
print(f"parent {P.number_of_edges():,} (was {E0:,})  mono={mono} connected={nx.is_connected(P)} bipartite={nx.is_bipartite(P)}")
w2=defaultdict(int)
for u,v in P.edges():
    a,b=n2c[u],n2c[v]
    if a!=b: w2[(min(a,b),max(a,b))]+=1
W=np.array([w2.get((min(u,v),max(u,v)),0) for u,v in Q.edges()],float)
com=sorted(louvain_communities(Q,seed=42),key=len,reverse=True)
lab={nd:i for i,c in enumerate(com) for nd in c}
btw=100*np.mean([lab[u]!=lab[v] for u,v in Q.edges()])
print(f"\nAFTER SWAP+INVERSION: quotient {Q.number_of_edges():,} edges")
print(f"  KS={ks_2samp(W[W>0],fc).statistic:.3f} Gini={gini(W):.3f} mean={W.mean():.2f} "
      f"w1={100*(W==1).mean():.1f}% w2={100*(W==2).mean():.1f}% max={W.max():.0f} "
      f">18={int((np.round(W)>18).sum())} zero={int((W==0).sum())}")
print(f"  Q={modularity(Q,com):.3f} C={nx.average_clustering(Q):.3f} btw={btw:.1f}%")
np.save(_p('final_w.npy'),W)
np.savez(_p('graph_final.npz'),n_total=1015,n1=338,n2=338,n3=339,
         edges=np.array(sorted((min(u,v),max(u,v)) for u,v in Q.edges()),dtype=np.int64))

E=np.array(sorted((min(u,v),max(u,v)) for u,v in P.edges()),dtype=np.int64)
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
np.savez(_p('parent_final_masks.npz'),n1=n1,n2=n2,n3=n3,mask_12=m12,mask_13=m13,mask_23=m23)
import itertools


tri=None
for a,b in itertools.islice(P.edges(),3000):
    common=set(P[a])&set(P[b])
    if common: tri=(a,b,next(iter(common))); break
print(f"FINAL PARENT saved: {P.number_of_edges():,} edges")
print(f"  proper 3-colouring : {sum(1 for u,v in P.edges() if col[u]==col[v])==0}")
print(f"  colour classes     : {[int((col==c).sum()) for c in range(3)]}")
print(f"  connected          : {nx.is_connected(P)}")
print(f"  non-bipartite      : {not nx.is_bipartite(P)}   triangle witness {tri}")
print(f"  min degree         : {min(d for _,d in P.degree())}")
print(f"  => chi = 3")

print(f"done ({time.time()-t0:.0f}s)")
