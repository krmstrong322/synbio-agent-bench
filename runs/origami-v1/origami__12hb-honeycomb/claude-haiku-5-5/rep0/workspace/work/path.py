edges = [(0,1),(0,4),(1,5),(2,3),(2,6),(3,7),(4,8),(5,6),(5,9),(6,10),(7,11),(8,9),(10,11)]
adj = {i: set() for i in range(12)}
for a, b in edges:
    adj[a].add(b); adj[b].add(a)
paths = []
def dfs(p):
    if len(paths) > 2000:
        return
    if len(p) == 12:
        paths.append(list(p)); return
    for n in sorted(adj[p[-1]]):
        if n not in p:
            p.append(n); dfs(p); p.pop()
for s in range(12):
    dfs([s])
print(len(paths))
for p in paths[:5]:
    print(p)
# bipartition
cls = {0: 0}
stack = [0]
while stack:
    x = stack.pop()
    for n in adj[x]:
        if n not in cls:
            cls[n] = 1 - cls[x]; stack.append(n)
print(cls)
