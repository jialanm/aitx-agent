# Rule selector comparison

Blocks from `data/eval_trace_32b_run6.json`; hand label in `scripts/compare_rule_selectors.py`.
No model was run. Hit: a needed rule attached. Miss: a needed rule withheld.
Extra: a rule attached that the answer does not rest on.

| Question | Needed | trigger | clause | scoped |
|---|---|---|---|---|
| AITX-00001 | exon-skipping, reading-frame, splicing | reading-frame | exon-skipping, mechanism-direction, reading-frame, splicing | exon-skipping, mechanism-direction, reading-frame, splicing |
| AITX-00002 | read-through | nonsense-mediated-decay, read-through | mechanism-direction, nonsense-mediated-decay, read-through | mechanism-direction, nonsense-mediated-decay, read-through |
| AITX-00003 | - | - | mechanism-direction | mechanism-direction |
| AITX-00004 | - | - | - | - |
| AITX-00005 | - | nonsense-mediated-decay, reading-frame | nonsense-mediated-decay, reading-frame | nonsense-mediated-decay, reading-frame |
| AITX-00006 | - | nonsense-mediated-decay, reading-frame | nonsense-mediated-decay, reading-frame | nonsense-mediated-decay, reading-frame |
| AITX-00007 | - | - | - | - |
| AITX-00008 | - | - | - | - |
| AITX-00009 | - | reading-frame | - | - |
| AITX-00010 | - | reading-frame | - | - |
| AITX-00011 | - | - | - | - |
| AITX-00012 | mechanism-direction | nonsense-mediated-decay, read-through | mechanism-direction, nonsense-mediated-decay, read-through | mechanism-direction, nonsense-mediated-decay, read-through |
| AITX-00013 | - | nonsense-mediated-decay, reading-frame | nonsense-mediated-decay, reading-frame | - |
| AITX-00014 | - | nonsense-mediated-decay, reading-frame | nonsense-mediated-decay, reading-frame | - |
| AITX-00015 | nonsense-mediated-decay | nonsense-mediated-decay, reading-frame | nonsense-mediated-decay, reading-frame | nonsense-mediated-decay, reading-frame |
| AITX-00016 | - | - | - | - |

| Selector | Needed rules attached | Withheld | Extra |
|---|---|---|---|
| trigger | 3 of 6 | 3 | 14 |
| clause | 6 of 6 | 0 | 15 |
| scoped | 6 of 6 | 0 | 11 |

Per rule, needed and attached:

| Rule | trigger | clause | scoped |
|---|---|---|---|
| reading-frame | 1/1 (+7) | 1/1 (+5) | 1/1 (+3) |
| splicing | 0/1 (+0) | 1/1 (+0) | 1/1 (+0) |
| exon-skipping | 0/1 (+0) | 1/1 (+0) | 1/1 (+0) |
| nonsense-mediated-decay | 1/1 (+6) | 1/1 (+6) | 1/1 (+4) |
| read-through | 1/1 (+1) | 1/1 (+1) | 1/1 (+1) |
| mechanism-direction | 0/1 (+0) | 1/1 (+3) | 1/1 (+3) |
