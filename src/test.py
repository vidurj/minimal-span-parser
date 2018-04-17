d = [(), ('S',), ('PP',), ('NP',), ('PRN',), ('VP',), ('ADVP',), ('SBAR', 'S'), ('ADJP',), ('QP',), ('UCP',), ('S', 'VP'), ('SBAR',), ('WHNP',), ('SINV',), ('FRAG',), ('NAC',), ('WHADVP',), ('NP', 'QP'), ('PRT',), ('S', 'PP'), ('S', 'NP'), ('NX',), ('S', 'ADJP'), ('WHPP',), ('SBAR', 'S', 'VP'), ('SBAR', 'SINV'), ('SQ',), ('NP', 'NP'), ('SBARQ',), ('SQ', 'VP'), ('CONJP',), ('ADJP', 'QP'), ('FRAG', 'NP'), ('FRAG', 'ADJP'), ('WHADJP',), ('ADJP', 'ADJP'), ('FRAG', 'PP'), ('S', 'ADVP'), ('FRAG', 'SBAR'), ('PRN', 'S'), ('PRN', 'S', 'VP'), ('INTJ',), ('X',), ('NP', 'NP', 'NP'), ('FRAG', 'S', 'VP'), ('ADVP', 'ADVP'), ('RRC',), ('VP', 'PP'), ('VP', 'VP'), ('SBAR', 'FRAG'), ('ADVP', 'ADJP'), ('LST',), ('NP', 'NP', 'QP'), ('PRN', 'SBAR'), ('VP', 'S', 'VP'), ('S', 'UCP'), ('FRAG', 'WHNP'), ('NP', 'PP'), ('NP', 'SBAR', 'S', 'VP'), ('WHNP', 'QP'), ('VP', 'FRAG', 'ADJP'), ('FRAG', 'WHADVP'), ('NP', 'ADJP'), ('VP', 'SBAR'), ('NP', 'S', 'VP'), ('X', 'PP'), ('S', 'VP', 'VP'), ('S', 'VP', 'ADVP'), ('WHNP', 'WHNP'), ('NX', 'NX'), ('FRAG', 'ADVP'), ('FRAG', 'VP'), ('VP', 'ADVP'), ('SBAR', 'WHNP'), ('FRAG', 'SBARQ'), ('PP', 'PP'), ('PRN', 'PP'), ('VP', 'NP'), ('X', 'NP'), ('PRN', 'SINV'), ('NP', 'SBAR'), ('PP', 'NP'), ('NP', 'INTJ'), ('FRAG', 'INTJ'), ('X', 'VP'), ('PRN', 'NP'), ('FRAG', 'UCP'), ('NP', 'ADVP'), ('SBAR', 'SBARQ'), ('SBAR', 'SBAR', 'S'), ('SBARQ', 'WHADVP'), ('ADVP', 'PRT'), ('UCP', 'ADJP'), ('PRN', 'FRAG', 'WHADJP'), ('FRAG', 'S'), ('S', 'S'), ('FRAG', 'S', 'ADJP'), ('INTJ', 'S'), ('ADJP', 'NP'), ('X', 'ADVP'), ('FRAG', 'WHPP'), ('NP', 'FRAG'), ('NX', 'QP'), ('NP', 'S'), ('SBAR', 'WHADVP'), ('X', 'SBARQ'), ('NP', 'PRN'), ('NX', 'S', 'VP'), ('NX', 'S'), ('UCP', 'PP'), ('RRC', 'VP'), ('ADJP', 'ADVP')]
e = []
for x in d:
    e.append(list(x))
    e[-1].sort()
d = e
d.sort(key=lambda x: - len(x))
for x in d:
    print(x)



[('``', '``'), ('DT', 'The'), ('NN', 'range'), ('IN', 'of'), ('NNS', 'expectations'), ('VBZ', 'is'), ('RB', 'so'), ('JJ', 'broad'), (',', ','), ("''", "''"), ('DT', 'a'), ('NN', 'dealer'), ('IN', 'at'), ('DT', 'another'), ('JJ', 'major'), ('NNP', 'U.K.'), ('NN', 'brokerage'), ('NN', 'firm'), ('VBD', 'said'), ('``', '``'), (',', ','), ('DT', 'the'), ('NN', 'deficit'), ('MD', 'may'), ('VB', 'have'), ('TO', 'to'), ('VB', 'be'), ('IN', 'nearer'), ('CC', 'or'), ('IN', 'above'), ('num', 'num'), ('CD', '2'), ('CD', 'billion'), ('IN', 'for'), ('PRP', 'it'), ('TO', 'to'), ('VB', 'have'), ('DT', 'any'), ('NN', 'impact'), ('IN', 'on'), ('DT', 'the'), ('NN', 'market'), ("''", "''")], \
[('``', '``'), ('DT', 'The'), ('NN', 'range'), ('IN', 'of'), ('NNS', 'expectations'), ('VBZ', 'is'), ('RB', 'so'), ('JJ', 'broad'), (',', ','), ("''", "''"), ('DT', 'a'), ('NN', 'dealer'), ('IN', 'at'), ('DT', 'another'), ('JJ', 'major'), ('NNP', 'U.K.'), ('NN', 'brokerage'), ('NN', 'firm'), ('VBD', 'said'), ('``', '``'), (',', ','), ('DT', 'the'), ('NN', 'deficit'), ('MD', 'may'), ('VB', 'have'), ('TO', 'to'), ('VB', 'be'), ('IN', 'nearer'), ('CC', 'or'), ('IN', 'above'), ('num', 'num'), ('CD', '2'), ('CD', 'billion'), ('IN', 'for'), ('PRP', 'it'), ('TO', 'to'), ('VB', 'have'), ('DT', 'any'), ('NN', 'impact'), ('IN', 'on'), ('DT', 'the'), ('NN', 'market'), ("''", "''"), ('.', '.')]


(S (NP (NNP Gold)) (VP (ADVP (RB also)) (VBD rose)) (. .))
(S (NP (NNP Gold)) (ADVP (RB also)) (VP (VBD rose)) (. .))