import sys
sys.path.insert(0, "designs")
import scadnano as sc
from geom import *

groups = make_groups()
helices = make_helices(groups)
d = sc.Design(helices=helices, groups=groups, strands=[])
d.write_scadnano_file(filename="designs/draft.sc")
print("edges", len(EDGES), "helices", len(helices))
