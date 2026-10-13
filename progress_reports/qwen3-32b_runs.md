# Qwen3-32B on the Phase 1 validator: first run and a reading of its reasoning

Model: Qwen/Qwen3-32B-FP8, the publisher's fp8 checkpoint, loaded with
transformers (kernels 0.16 for the fp8 matrix multiplies). Hardware: one
NVIDIA L40S (g6e.xlarge, 45.8 GiB). Decoding: greedy, batch size one,
thinking on in the loop and off for extraction, retries and citations, as
for the 8B. Commit d886b7c (2026-10-13). The agent code, prompts, schemas
and tools are identical to run 5 of the 8B (d5ae047); the commits between
add only the model and precision flags, the incremental write of the run
files, and the `--show-thinking` flag. So the two runs differ in the model
alone. n = 16 public questions; the 8B comparison is run 5 (deterministic
across five repeats).

## 1. Results

| Patient | Expected | 32B | 8B (run 5) | 32B s | 8B s |
|---|---|---|---|---|---|
| 00001 | Eteplirsen | Eteplirsen | None | 853 | 304 |
| 00002 | Ataluren | None | None | 507 | 148 |
| 00003 | None | None | None | 238 | 224 |
| 00004 | Pyridoxine | pyridoxine | pyridoxine | 196 | 142 |
| 00005 | No | No | No | 153 | 98 |
| 00006 | Lumasiran | lumasiran, nedosiran | lumasiran, nedosiran | 213 | 65 |
| 00007 | 18 months | Not applicable | 18 months | 400 | 63 |
| 00008 | ascorbic acid, desmopressin | ascorbic acid, desmopressin | same | 207 | 153 |
| 00009 | NCT05402384 | NCT05402384 | NCT05402384 | 137 | 64 |
| 00010 | No | No | No | 146 | 67 |
| 00011 | None | NCT07600736, NCT07227857 | NCT07227857 | 362 | 116 |
| 00012 | L-Serine | L-serine | Memantine | 221 | 64 |
| 00013 | 66 | 66 | 66 | 242 | 68 |
| 00014 | 0.1 | 10 | 10 | 344 | 60 |
| 00015 | No | No | No | 472 | 89 |
| 00016 | GRD | TBD | TBD | 174 | 78 |

| | 32B fp8 | 8B bf16 |
|---|---|---|
| Exact match | 10/16 | 9/16 |
| Gained / lost against the 8B | +00001, +00012 / -00007 | |
| Citations verbatim in their source (code check) | 18 of 37 | 32 of 38 |
| Citations good by hand (right link, relevant quote, sensible reason) | 13 of 37 | 18 of 38 |
| Agent time | 81 min (4,864 s) | 30 min (1,804 s) |
| Seconds per question, median | 230 | 84 |

The time ratio is about 2.7, which is the cost of reading four times the
weights per token at fp8 on one GPU, partly offset by thinking that ends
sooner on the easy questions.

## 2. Is the reasoning grounded in the evidence?

The question this run was meant to answer is not "does the larger model
score higher" but "does it reason from the retrieved text rather than from
memory". Every model turn of both runs was read against the tool texts the
turn had seen; each decisive step was classed as coming from evidence,
from memory, or from neither.

| Patient | 32B decisive step | 8B decisive step | 32B verdict |
|---|---|---|---|
| 00001 | recomputed the deletion as 1,743 letters from the HGVS, called it in-frame, overriding the exon line, then chose exon 51 as "the closest exon before the deletion" | read VEP's in-frame label, never did the arithmetic, think-loop | right answer, wrong reasoning |
| 00002 | turn 1 from memory was right (delins keeps the frame, nonsense, ataluren approved for nonsense DMD); turn 3 answered Ataluren; the evidence reminder sent it to the chapter, which does not name ataluren; turn 5 cited an FDA label check it never made and answered None | memory: ataluren "not FDA-approved", so None | right, then wrong; imagined tool result |
| 00003 | missense, chapter, drug map from memory | same, plus a turn lost to the exon line's wording | sound |
| 00004 | the chapter's pyridoxine sentence; memory of the disease was right (and invented a drug name in passing) | the chapter; memory of the disease was wrong and ClinVar corrected it | sound |
| 00005 | frameshift against "missense variants known to be responsive" | same | sound |
| 00006 | planned in turn 1 to fetch the FDA label, then answered after GeneReviews without doing it; noted the age, did not use it | answered after GeneReviews | stopped early despite its own plan |
| 00007 | wrong memory (DDC as dopa-responsive dystonia), OMIM, ClinVar, then "the tools don't show any approval" about GeneReviews, never called | GeneReviews first, read the EU/UK sentence | wrong memory plus imagined coverage |
| 00008 | chapter chosen by phenotype, two sentences | same | sound |
| 00009 | drug trial over the natural-history study | same | sound |
| 00010 | hemoglobin 5 against the exclusion of under 7 | same | sound; its citation gave a different reason |
| 00011 | ages checked, both pediatric KCNT1 trials listed; long deliberation on the answer format | one trial | sound on live data; the key predates the trials |
| 00012 | memory: loss of function needs an enhancer, antagonists go the wrong way; the PubMed query (gene, "nonsense", three drugs joined by OR) returned gastric-cancer abstracts, which it then said "support L-serine" | PHAROS lists memantine as approved on the target, so memantine | right by memory; evidence irrelevant |
| 00013 | exon line; answered without a tool, reminded, fetched, answered | exon line | sound |
| 00014 | exon line's 10.0% | same | key convention |
| 00015 | turn 1 derived "yes, NMD"; turn 2 derived "no" by a different muddled route | applied the last-exon rule once | right, unstable |
| 00016 | UniProt's boundary; did not know the NF1 domain abbreviations | same | key plus nomenclature |

Tallies:

| | 32B | 8B |
|---|---|---|
| Correct and reached soundly from evidence | 7 of 10 | 9 of 9 |
| Correct by memory or heuristic, evidence absent or irrelevant | 3 (00001, 00012, 00015) | 0 |
| Thinking describes a tool result that was never fetched | 2 (00002, 00007) | 0 |
| Reversed its own conclusion inside one question | 3 (00002, 00011, 00015) | 0 |
| Stopped before the tool that held the deciding fact | 2 (00006, 00007) | 2 (00006, 00012) |
| Wrong recalled facts noticed in the thinking | drug-to-exon map, DDC's disease, a drug name, a domain name | drug-to-exon map (changing between turns), AGXT's disease twice, COL1A1's disease, SLC35A2's CDG type, ataluren's mechanism |

Reading: the 32B remembers more, and more accurately, so it answers two
questions the 8B could not (00012's mechanism argument is a good one, and
it is from memory because no tool text contained it). It does not use the
retrieved text better than the 8B: on the seven questions both answered
soundly the evidence steps are the same, and on 00006 it abandoned the
label check it had itself planned. Its longer thinking is less stable, not
more careful: three reversals, two of them changing the final answer, and
in two questions it narrated tool results it had not fetched, which the 8B
never did. Scale bought knowledge, not grounding.

Interaction with the harness, recorded because it cut the other way: on
00002 the evidence reminder was right to demand a source for "Ataluren",
and the fetch it prompted returned a chapter that does not name the drug,
which the model took as evidence against its own correct answer. The fix
is the coverage note now in the GeneReviews, openFDA and PubMed schemas,
which says where a therapy absent from the United States sources is to be
looked for; the reminder stays.

## 3. Citations by hand

The code check (quote verbatim in the source text) is the first layer and
counts 32 of 38 for the 8B and 18 of 37 for the 32B. Reading all 75:

| Judgment | 8B | 32B |
|---|---|---|
| Good: right link, relevant quote, sensible reason | 18 | 13 |
| Verbatim but beside the point, or the reason is wrong | 11 | 4 |
| Verbatim under the wrong link (shared search text) | 3 | 1 |
| Refused: model declined with NOT SUPPORTED | 5 | 10 |
| Refused: quote was only a trial ID (under the 15-character minimum) | 0 | 6 |
| Refused: words not in the text | 1 | 1 |
| Rejected: text truncated before the quoted sentence | 0 | 2 |

The 32B's lower code count is mostly habit, not fabrication: it declines
the step-1 Ensembl page on questions where the consequence is beside the
point (correctly), and it quotes trial identifiers rather than sentences.
Both models attach a sentence to the wrong trial when the trial search
returns one text for several links; the per-trial text split fixes that.
The eleven and four "beside the point" items are the layer no string check
reaches and would need the hand label as a standing metric.

## 4. What changed since this run, and what to run next

Committed after this run, from what the two traces showed:

- The VEP block explains the relation between VEP's consequence label and
  the exon line, and states whole-exon loss when a deletion removes splice
  sites (fc82f5d), the exon line's frame remark says what it is about, and
  VEP's most-severe consequence is attributed to its own transcript
  (8c91fee).
- GeneReviews, openFDA and PubMed schemas state the United States coverage
  limit and point to PubMed for therapies approved elsewhere (899267e,
  50fef0b).
- A citation the model declines is left out rather than emitted as
  unsupported (1ab5556).

Next: rerun both models from the current commit, so the comparison above
has a before and after; split the trial-search text per trial; keep the
hand label of reasoning (evidence, memory, imagined) as a per-run table
beside the exact-match score.
