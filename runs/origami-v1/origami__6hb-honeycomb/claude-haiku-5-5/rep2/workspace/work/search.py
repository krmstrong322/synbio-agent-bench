import itertools
R = lambda a,b: list(range(a,b+1))
# valid half-crossover offsets: D[(x,y,dx)] where x->y adjacent and x has direction dx
D = {
 (0,1,'f'): [7,8,18,28,29,39,49,50,60,70,71,81,91,92,102,112,113,123],
 (0,1,'r'): [2,3,12,13,23,24,33,34,44,45,54,55,65,66,75,76,86,87,96,97,107,108,117,118],
 (1,2,'f'): [5,6,16,17,26,27,37,38,47,48,58,59,68,69,79,80,89,90,100,101,110,111,121,122],
 (1,2,'r'): [0,1,11,21,22,32,42,43,53,63,64,74,84,85,95,105,106,116],
 (2,3,'f'): [4,14,15,25,35,36,46,56,57,67,77,78,88,98,99,109,119,120],
 (2,3,'r'): [9,10,19,20,30,31,40,41,51,52,61,62,72,73,82,83,93,94,103,104,114,115,124,125],
 (3,4,'f'): [2,3,12,13,23,24,33,34,44,45,54,55,65,66,75,76,86,87,96,97,107,108,117,118],
 (3,4,'r'): [7,8,18,28,29,39,49,50,60,70,71,81,91,92,102,112,113,123],
 (4,5,'f'): [0,1,11,21,22,32,42,43,53,63,64,74,84,85,95,105,106,116],
 (4,5,'r'): [5,6,16,17,26,27,37,38,47,48,58,59,68,69,79,80,89,90,100,101,110,111,121,122],
 (5,0,'f'): [9,10,19,20,30,31,40,41,51,52,61,62,72,73,82,83,93,94,103,104,114,115,124,125],
 (5,0,'r'): [4,14,15,25,35,36,46,56,57,67,77,78,88,98,99,109,119,120],
}
def S(x,y,dx):
    if (x,y,dx) in D: return D[(x,y,dx)]
    dy = 'r' if dx=='f' else 'f'
    return D[(y,x,dy)]  # symmetric: y has opposite dir
# note: for x->y with x 'f', y is 'r' -> set D[(x,y,'f')]; check symmetric key use
results=[]
ring=[0,1,2,3,4,5]
for omit in range(6):
    # path: start after omitted edge
    path=[(omit+1+i)%6 for i in range(6)]
    for d0 in ['f','r']:
        dirs=[d0 if i%2==0 else ('r' if d0=='f' else 'f') for i in range(6)]
        sets=[S(path[i],path[i+1],dirs[i]) for i in range(5)]
        # junction offsets: search over all choices (small)
        best=None
        for offs in itertools.product(*sets):
            spans=[]
            ok=True
            for i in range(6):
                if i==0:
                    lo,hi=(0,offs[0]) if dirs[0]=='f' else (offs[0],125)
                    # free end side: f starts at bottom free end 0; r free end is at top? r: 5' at top free
                    if dirs[0]=='f': lo,hi=0,offs[0]
                    else: lo,hi=offs[0],125
                elif i==5:
                    if dirs[5]=='f': lo,hi=offs[4],125
                    else: lo,hi=0,offs[4]
                else:
                    if dirs[i]=='f': lo,hi=offs[i-1],offs[i]
                    else: lo,hi=offs[i],offs[i-1]
                spans.append(hi-lo)
            m=min(spans)
            if best is None or m>best[0]:
                best=(m,offs,spans)
        results.append((best[0],path,dirs,best[1],best[2]))
results.sort(reverse=True)
for r in results[:6]: print(r)
