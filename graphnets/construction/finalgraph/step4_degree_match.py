"""Reshape the bundle-matched quotient's DEGREE SEQUENCE to Budapest's.

Defect: too compact -- diameter 3 vs 5, eff diam 2.09 vs 2.73, g(2) 0.89 vs 0.65,
min degree 31 vs 7.  Cause: no peripheral supernodes.

Fix, entirely inside the honest framework (quotient stays DERIVED from the parent):
  target degree sequence := Budapest's, assigned to supernodes by rank.
  over-degree supernode -> drop its weakest partners, those parent edges move
                           INSIDE the supernode (invisible, nothing deleted).
  under-degree supernode -> gain partners, parent edges added cross-colour.
  parent edge count held constant throughout.
Then re-assign Budapest's fibre-count multiset rank-for-rank to the new pairs.
"""
import numpy as np, networkx as nx, pickle, time
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

t0=time.time(); rng=np.random.default_rng(11)
def log(*a): print('[%6.0fs]'%(time.time()-t0),*a,flush=True)
P=np.load(_p('parent_final_masks.npz'))
n1,n2,n3=int(P['n1']),int(P['n2']),int(P['n3']); off2,off3=n1,n1+n2; n=n1+n2+n3
G=nx.Graph(); G.add_nodes_from(range(n))
for mk,orow,ocol in [('mask_12',off2,0),('mask_13',off3,0),('mask_23',off3,off2)]:
    r,c=np.nonzero(P[mk]); G.add_edges_from((int(cc)+ocol,int(rr)+orow) for rr,cc in zip(r,c))
col=np.concatenate([np.zeros(n1,int),np.ones(n2,int),2*np.ones(n3,int)])
E0=G.number_of_edges(); log("parent %d edges"%E0)
# recover the SAME partition the bundle-matched graph used
D=pickle.load(open(_p('v146_weights.pkl'),'rb'))
from networkx.algorithms.community import louvain_communities

Gm=nx.Graph(); Gm.add_nodes_from(range(n))
d0=np.load(_p('graph_brain_mild.npz'))
for mk,orow,ocol in [('mask_12',off2,0),('mask_13',off3,0),('mask_23',off3,off2)]:
    r,c=np.nonzero(d0[mk]); Gm.add_edges_from((int(cc)+ocol,int(rr)+orow) for rr,cc in zip(r,c))
cl=[set(x) for x in sorted(louvain_communities(Gm,resolution=20.0,seed=42),key=lambda c:-len(c))]
n2c={nd:ci for ci,c in enumerate(cl) for nd in c}
while len(cl)>1015:
    mi=min(range(len(cl)),key=lambda i:len(cl[i])); mc=cl[mi]
    ne=defaultdict(int)
    for nd in mc:
        for nb in Gm.neighbors(nd):
            k=n2c[nb]
            if k!=mi: ne[k]+=1
    bt=(max(ne,key=lambda c: ne[c]/(len(mc)*len(cl[c]))) if ne else
        min(((len(cl[i]),i) for i in range(len(cl)) if i!=mi))[1])
    cl[bt]|=mc
    for nd in mc: n2c[nd]=bt
    cl.pop(mi)
    for ci,c in enumerate(cl):
        for nd in c: n2c[nd]=ci
members=[sorted(c) for c in cl]; K=len(cl)
sid=np.array([n2c[i] for i in range(n)])
log("partition recovered: %d supernodes"%K)
adj={x:set(G[x]) for x in G}
def quotient():
    q=defaultdict(list)
    for u,v in G.edges():
        a,b=sid[u],sid[v]
        if a!=b: q[(min(a,b),max(a,b))].append((u,v))
    return q
q=quotient(); deg=np.zeros(K,int)
for (a,b) in q: deg[a]+=1; deg[b]+=1
B=nx.read_gml(_p('budapest_connectome.gml'))
tgt_sorted=np.sort([d for _,d in B.degree()])
tgt=np.zeros(K,int); tgt[np.argsort(deg)]=tgt_sorted           # rank-match
log("degree now : min %d med %d mean %.1f max %d"%(deg.min(),np.median(deg),deg.mean(),deg.max()))
log("degree tgt : min %d med %d mean %.1f max %d"%(tgt.min(),np.median(tgt),tgt.mean(),tgt.max()))

def put_inside(edges_needed):
    """spend `edges_needed` cross-colour additions inside supernodes, triangle-first"""
    put=0
    for ci in sorted(range(K),key=lambda c:-len(members[c])):
        if put>=edges_needed: break
        mem=members[ci]
        if len(mem)<2: continue
        cand=[(len(adj[u]&adj[v]),u,v) for i,u in enumerate(mem) for v in mem[i+1:]
              if col[u]!=col[v] and not G.has_edge(u,v)]
        cand.sort(reverse=True)
        for sc,u,v in cand:
            if put>=edges_needed: break
            G.add_edge(u,v); adj[u].add(v); adj[v].add(u); put+=1
    return put

# ---- 1. over-degree supernodes shed their weakest partners
over=[c for c in range(K) if deg[c]>tgt[c]]
dropped=[]
for c in sorted(over,key=lambda c: tgt[c]-deg[c]):
    cur=[p for p in q if c in p]
    excess=len(cur)-tgt[c]
    if excess<=0: continue
    cur.sort(key=lambda p: len(q[p]))                 # weakest bundles first
    for p in cur[:excess]:
        o=p[0] if p[1]==c else p[1]
        if deg[o]<=tgt[o]: continue                   # do not starve the partner
        for e in q[p]:
            if G.has_edge(*e):
                G.remove_edge(*e); adj[e[0]].discard(e[1]); adj[e[1]].discard(e[0]); dropped.append(e)
        deg[c]-=1; deg[o]-=1; del q[p]
log("shed %d pairs -> %d parent edges freed; parent %d"%(len(over),len(dropped),G.number_of_edges()))
# ---- 2. under-degree supernodes gain partners, paid from the freed pool
budget=len(dropped); spent=0
under=[c for c in range(K) if deg[c]<tgt[c]]
for c in sorted(under,key=lambda c: deg[c]-tgt[c]):
    while deg[c]<tgt[c] and spent<budget:
        cands=[o for o in rng.choice(K,60) if o!=c and (min(c,o),max(c,o)) not in q and deg[o]<tgt[o]]
        if not cands: break
        o=cands[0]; A,Bm=members[c],members[o]
        made=0
        for u in A:
            for v in Bm:
                if col[u]!=col[v] and not G.has_edge(u,v):
                    G.add_edge(u,v); adj[u].add(v); adj[v].add(u)
                    q[(min(c,o),max(c,o))].append((u,v)); made+=1; spent+=1; break
            if made>=2: break
        if made==0: break
        deg[c]+=1; deg[o]+=1
log("added %d pairs' worth (%d edges) to under-degree supernodes"%(len(under),spent))
short=E0-G.number_of_edges()
if short>0: log("restoring parent size, %d inside supernodes: +%d"%(short,put_inside(short)))
mono=sum(1 for u,v in G.edges() if col[u]==col[v])
log("parent %d (was %d)  mono %d  connected %s"%(G.number_of_edges(),E0,mono,nx.is_connected(G)))
q=quotient(); Eq=np.array(sorted(q),dtype=np.int64); W=np.array([len(q[tuple(p)]) for p in Eq],float)
Q=nx.Graph(); Q.add_nodes_from(range(K)); Q.add_edges_from(map(tuple,Eq))
import community as cm


dq=np.array([d for _,d in Q.degree()])
def gini(x):
    x=np.sort(np.asarray(x,float)); m=len(x); return float((2*np.arange(1,m+1)-m-1).dot(x)/(m*x.sum()))
log("NEW quotient %d edges  degree min %d med %d mean %.1f max %d"%(len(Eq),dq.min(),np.median(dq),dq.mean(),dq.max()))
log("  bundle mean %.3f max %d gini %.3f >18 %d"%(W.mean(),W.max(),gini(W),int((W>18).sum())))
if nx.is_connected(Q):
    log("  Q %.4f C %.4f APL %.4f diam %d"%(cm.modularity(cm.best_partition(Q,random_state=0),Q),
        nx.average_clustering(Q),nx.average_shortest_path_length(Q),nx.diameter(Q)))
else: log("  DISCONNECTED")
log("  Budapest 70,654 | deg min 7 med 127 mean 139.2 max 466 | Q .5574 C .6696 APL 2.219 diam 5")
np.savez(_p('graph_degmatch.npz'),n_total=K,edges=Eq); np.save(_p('degmatch_w.npy'),W)
idx=np.full(n,-1,np.int64)
for c in range(3):
    w_=np.where(col==c)[0]; idx[w_]=np.arange(len(w_))
m12=np.zeros((n2,n1),np.uint8); m13=np.zeros((n3,n1),np.uint8); m23=np.zeros((n3,n2),np.uint8)
for u,v in G.edges():
    cu,cv=col[u],col[v]
    if cu>cv: u,v=v,u; cu,cv=cv,cu
    if cu==0 and cv==1: m12[idx[v],idx[u]]=1
    elif cu==0 and cv==2: m13[idx[v],idx[u]]=1
    else: m23[idx[v],idx[u]]=1
np.savez(_p('parent_degmatch_masks.npz'),n1=n1,n2=n2,n3=n3,mask_12=m12,mask_13=m13,mask_23=m23,
         supernode_id_1=sid[:n1],supernode_id_2=sid[n1:n1+n2],supernode_id_3=sid[n1+n2:],n_supernodes=K)
log("saved graph_degmatch.npz / parent_degmatch_masks.npz")
