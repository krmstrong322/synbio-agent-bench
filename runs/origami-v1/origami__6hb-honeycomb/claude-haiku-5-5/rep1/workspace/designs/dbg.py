import sys, time
sys.setrecursionlimit(10000)
src = open("designs/search_spans.py").read().split("results = []")[0]
ns = {}
exec(src, ns)
tile = ns["tile"]
t = time.time()
for S0 in (22, 25, 30, 33):
    for S5 in (8, 10, 6):
        start = [S0, 22, 22, 8, 8, S5]
        end = [155, 155, 141, 141, 127, 127]
        u = tile(start, end)
        print(start, "OK" if u else "no", round(time.time() - t, 1), flush=True)
        if u:
            print(u)
            break
