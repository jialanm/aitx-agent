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
| 00014 | 0.1 | 10 (correct: the percentage asked for) | 10 (correct) | 344 | 60 |
| 00015 | No | No | No | 472 | 89 |
| 00016 | GRD | TBD | TBD | 174 | 78 |

| | 32B fp8 | 8B bf16 |
|---|---|---|
| Exact match | 10/16 | 9/16 |
| Counting 00014 (10.0 percent is the percentage asked for; the key's 0.1 is the fraction) | 11/16 | 10/16 |
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
| 00014 | exon line's 10.0% | same | correct; the key writes the fraction |
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

## 5. Run 2 (2026-10-13, commit 84f8cff): the harness changes, same model

Same model, precision and hardware as run 1. The agent changed in response
to run 1's traces and the 8B reading; the model did not. Changes between
d886b7c and 84f8cff that touch behaviour:

- the prompt says a remembered fact is a hypothesis to confirm with a tool
  (rule 6), asks for a NOTES line before each tool call because the thinking
  of earlier turns is not kept, and allows several tool calls in a turn;
- an answer term that no tool text contains is named back to the model once,
  with the recommended tools not yet called; a None answer while a
  recommended tool is uncalled gets one reminder naming the tools;
- when the fifth loop turn ends in a tool call, one more generation with
  tools disabled reads the result and answers;
- the GeneReviews, openFDA and PubMed schemas state the United States
  coverage limit and point to PubMed for therapies approved elsewhere;
- the VEP block opens with a Frame line, has the splice-sites line, and no
  longer prints the most-severe consequence of another transcript as the
  patient's;
- a citation the model declines is left out instead of marked unsupported.

(A run at 47cc322 between the two was stopped after three questions and is
not reported.)

| Patient | Expected | Run 2 | Run 1 | Run 2 s | Run 1 s |
|---|---|---|---|---|---|
| 00001 | Eteplirsen | Eteplirsen | Eteplirsen | 772 | 853 |
| 00002 | Ataluren | Ataluren | None | 617 | 507 |
| 00003 | None | None | None | 483 | 238 |
| 00004 | Pyridoxine | pyridoxine | pyridoxine | 198 | 196 |
| 00005 | No | No | No | 212 | 153 |
| 00006 | Lumasiran | lumasiran | lumasiran, nedosiran | 347 | 213 |
| 00007 | 18 months | 18 months | Not applicable | 430 | 400 |
| 00008 | ascorbic acid, desmopressin | same | same | 229 | 207 |
| 00009 | NCT05402384 | NCT05402384 | NCT05402384 | 212 | 137 |
| 00010 | No | No | No | 150 | 146 |
| 00011 | None | None | NCT07600736, NCT07227857 | 314 | 362 |
| 00012 | L-Serine | Memantine | L-serine | 283 | 221 |
| 00013 | 66 | 66 | 66 | 174 | 242 |
| 00014 | 0.1 | 10 (correct: the percentage asked for) | 10 (correct) | 212 | 344 |
| 00015 | No | Yes | No | 386 | 472 |
| 00016 | GRD | TBD | TBD | 243 | 174 |

| | Run 2 (84f8cff) | Run 1 (d886b7c) | 8B run 5 |
|---|---|---|---|
| Exact match | 12/16 | 10/16 | 9/16 |
| Counting 00014 as correct | 13/16 | 11/16 | 10/16 |
| Citations emitted / marked unsupported | 24 / 6 | 37 / 19 | 38 / 6 |
| Declined citations left out | 13 | (marked) | (marked) |
| Agent time | 88 min | 81 min | 30 min |
| Reminders fired: term, None, evidence | 2, 4, 2 | 0, 0, 1 | 0, 0, 1 |

### What changed, question by question

- **00006, lumasiran (first time right for either model).** The term check
  fired on "lumasiran nedosiran"; the model fetched both FDA labels, read
  nedosiran's kidney-function restriction against the patient's chronic
  kidney disease, and chose lumasiran. This is the behaviour the check was
  built for: evidence fetched, read, and applied to the patient.
- **00002, Ataluren (first time right for either model).** Turn 3 answered
  None; the None reminder sent the model to ClinVar, which had nothing; on
  the last turn it reversed to Ataluren, reasoning that the N1C guidelines
  "would say" nonsense variants are eligible for read-through agents, a
  tool it never called, and that its earlier None "maybe" was a mistake.
  Right, from memory under reminder pressure; no tool text names ataluren.
  The term check could not act because no turn was left; the settlement now
  applies on every answer (0c85519), so the same answer would be removed
  unless the model looks the drug up, which the coverage note tells it to.
- **00007, 18 months (recovered).** GeneReviews was called first this time.
- **00011, None (matches the key).** The model put the condition text into
  the trial query, which returned nothing where "KCNT1" alone returned four
  trials in run 1; it then tried OMIM and answered None. A retrieval miss
  that happens to agree with a key the live registry has outrun; not a
  gain.
- **00015, Yes (lost).** The model computed the stop as 63 bases from the
  end of the transcript and applied "more than 55 means decay"; the rule
  concerns distance from the last exon-exon junction, and a stop in the last
  exon escapes. Misapplied from memory, as the 8B's reading of this rule
  was unstable too; the genetics reference's nonsense-mediated-decay
  section, added after this run, states the rule with its sources.
- **00012, Memantine (still wrong).** The model wrote the correct mechanism
  (loss of function needs an enhancer; a blocker is "counterintuitive") and
  then deferred to PHAROS's list of drugs with activity at the protein,
  reasoning that "the rules say to call at least one tool" and it had. Three
  changes followed: the prompt asks for a supporting result per step rather
  than a call count, the PHAROS line says the list is measured activity in
  any direction, and the reference gains a mechanism-direction rule.
- **00001, Eteplirsen (kept).** Reached again by the "closest exon before
  the deletion" heuristic; the Frame line of this commit was read in turn 1
  and then discounted in favour of VEP's inframe_deletion label, which led
  to the numbers-only block with coverage lines (cbedd91, 6282ad4).

### Reading

The two real gains, 00006 and 00002, came from the model being pushed back
to the tools after it had answered; one of them was then grounded (a label
read and applied), the other was not (a drug named from memory). The loss,
00015, and the standing miss, 00012, are both a rule applied wrongly or
set aside, which is what the genetics reference is for; whether a model
reads a rule it is offered is what run 3 (from 0c85519) tests. Reminders
are now a visible cost: four questions reached the fifth turn, against one
in run 1.

## 6. Runs 3 and 4 (2026-10-26 and 2026-10-27): grounding the two answers that came from memory

Run 3 (VM file name run4) ran at f23051e, after the numbers-only VEP block,
the genetics reference, InterPro domain matches in UniProt, the PubMed
gene-mention count, the per-step evidence wording, and the tool and system
message roles. Run 4 (VM file name run5) ran at 488fa62, after three changes
aimed at the two questions run 3 lost: openFDA's empty result now states that
it holds only drugs with a United States label and names PubMed; the None
reminder on a multiple-choice question lists the options as agents and names
the label call for each; the unverified-term reminder names the label call
for the drug the model wrote.

| Patient | Expected | Run 4 | Run 3 | Run 2 | Run 4 s |
|---|---|---|---|---|---|
| 00001 | Eteplirsen | None | None | Eteplirsen | 676 |
| 00002 | Ataluren | Ataluren | None | Ataluren | 1074 |
| 00003 | None | None | None | None | 468 |
| 00004 | Pyridoxine | Pyridoxine | Pyridoxine | pyridoxine | 291 |
| 00005 | No | No | No | No | 140 |
| 00006 | Lumasiran | lumasiran, nedosiran | lumasiran, nedosiran | lumasiran | 237 |
| 00007 | 18 months | 18 months | 18 months | 18 months | 179 |
| 00008 | ascorbic acid, desmopressin | same | same | same | 180 |
| 00009 | NCT05402384 | NCT05402384 | NCT04199000, NCT05402384 | NCT05402384 | 291 |
| 00010 | No | No | No | No | 133 |
| 00011 | None | NCT07600736, NCT07227857 | NCT07600736, NCT07227857 | None | 260 |
| 00012 | L-Serine | L-serine | L-serine | Memantine | 470 |
| 00013 | 66 | 66 | 66 | 66 | 147 |
| 00014 | 0.1 | 10 | 10 | 10 | 199 |
| 00015 | No | No | No | Yes | 296 |
| 00016 | GRD | GRD | GRD | TBD | 252 |

| | Run 4 (488fa62) | Run 3 (f23051e) | Run 2 (84f8cff) |
|---|---|---|---|
| Exact match | 12/16 | 10/16 | 12/16 |
| Counting the reviewed cases (00011, 00014) | 14/16 | 12/16 | 13/16 |
| Citations verbatim in their source | 34 of 39 | 29 of 36 | 18 of 24 |
| Agent time | 88 min | 85 min | 88 min |

**Run 4 is the best run of the project.** It is the first to hold the
knowledge gains and the grounding at once: 00012, 00015 and 00016, which run 2
answered wrongly, stay correct through the genetics reference's
mechanism-direction and decay rules and InterPro's domain boundaries, while
00002 returns to Ataluren by a different route than run 2's. In run 2 the
model named the drug from memory and no source mentioned it; in run 4 it
called the label, read openFDA's new statement that an empty reply is not
evidence of absence, and went to PubMed, so the answer rests on a retrieved
source. 00009 also returned to a single trial identifier.

Counting 00011 and 00014 as correct, which the review on 2026-10-13 settled,
the score is 14 of 16. 00011 names two recruiting KCNT1 trials that fit the
patient, where the key says None because both were registered after the
question's date; 00014 answers the percentage the question asks for, where
the key writes the same quantity as a fraction.

### The two real misses

- **00001 (None, expected Eteplirsen).** The None reminder went out with all
  five option names and an example label call, and the model ignored it:
  turns 4 and 5 answered None with no tool call, recorded twice as the
  reminder being ignored. Its thinking rules out each drug in turn, on one
  premise: "the VEP result says the deletion is 1743 bases, a multiple of
  three, so the reading frame is preserved. Exon-skipping is for out-of-frame
  deletions." Our own Frame line gives that reading first, the whole-exon
  reading second, and the model never reaches the second or the line computing
  that exon 51 completes the frame. It never called the genetics reference,
  whose splicing and exon-skipping rules resolve exactly this.
- **00006 (lumasiran, nedosiran, expected Lumasiran).** Both labels were
  fetched; nedosiran's restricts it to preserved kidney function, eGFR 30 or
  above, and the patient has chronic kidney disease with no eGFR given. The
  model named both without the condition. The report chain gets this right by
  recording nedosiran as conditional, which a one-word answer cannot express.

### What followed

Pointing at the reference from the tool text has now been ignored in five
runs, and the NOTES instruction in 28 of 28 tool-call turns, so the rules are
no longer waited for: step 1 attaches the rules a consequence calls for
(splice-site loss attaches splicing and exon skipping; a frameshift or stop
attaches the frame and decay rules), and an answer whose own wording turns on
a rule the model never read gets one notice naming the topic to call. The
trigger is a fact in the retrieved data or the model's own words, not its
judgment of what it needs.

The same reading applies to the options themselves. The drugs in 00001 were
ruled out from memory before any lookup, and the notice that named all five
with the exact call changed nothing, so step 1 now also fetches the FDA label
of every option RxNorm confirms is a drug, trimmed to the drug names and the
indications, with openFDA's coverage note kept whole when a drug has no
United States label. RxNorm decides, so a mechanism class such as "Gain of
function" or an option reading "None of the above" fetches nothing, and the
cap is five labels.

What this costs is measurable and is recorded here rather than absorbed:
prefetching the labels removes the model's opportunity to fail at choosing
the label tool, so these questions no longer contribute to the routing-error
count in the failure taxonomy. The step is therefore a flag, not a fixture.
`--no-step1-labels` is the control run, from which routing errors are read,
and the flag's value is written into each run's metadata as
`step1_option_labels`, so no two runs can be compared without it being
visible.
