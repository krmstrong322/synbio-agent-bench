import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scadnano as sc
from geom import make_groups, make_helices

verts, edges, groups = make_groups()
helices = make_helices(groups)
design = sc.Design(helices=helices, groups=groups, strands=[])
design.write_scadnano_file(filename="designs/draft.sc")
for k, e in enumerate(edges):
    print(k, e['vi'], e['vj'], [round(x, 2) for x in e['p0']])
