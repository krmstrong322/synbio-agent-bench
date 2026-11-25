import scadnano as sc
d = sc.Design.from_scadnano_file("designs/6hb.sc")
for s in d.strands:
    if s.is_scaffold: continue
    if s.dna_length() < 30 or s.dna_length() > 44:
        print(s.dna_length(), [(x.helix, x.start, x.end, 'f' if x.forward else 'r') for x in s.domains])
