"""ReAct agent loop — core orchestrator."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import asdict

from .model import ModelResponse, format_tool_result, generate, parse_response
from .normalization import normalize_answer, parse_option_json, verify_options
from .preprocessing import preprocess_input
from .prompts import (
    OPTION_EXTRACTION_INSTRUCTION,
    TOOL_PRIORITY,
    build_system_prompt,
    build_user_message,
)
from .schemas import EvidenceItem, TaskInput, TaskOutput
from .tools.base import BaseTool, EvidenceRecord
from .variant_validation import validate_variant
from .drug_names import is_drug_name, normalise_drug_names

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5
# How many option labels the harness fetches in step 1d. Five covers the
# validator's longest drug-option question; the cap keeps the prompt bounded.
MAX_OPTION_LABELS = 5
# How much of a source's retrieved text the justification step is shown.
SOURCE_TEXT_CHARS = 1500
UNSUPPORTED_JUSTIFICATION = "This record was retrieved and consulted; its returned text does not state the answer."


class Agent:
    """ReAct agent that uses tools to answer precision medicine questions."""

    def __init__(
        self,
        model,
        tokenizer,
        tools: list[BaseTool],
        schema_order_seed: int | None = None,
        show_thinking: bool = False,
        step1_labels: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = {tool.name: tool for tool in tools}
        self.tool_schemas = [tool.schema() for tool in tools]
        if schema_order_seed is not None:
            # Prompt-sensitivity control: same schemas, different listing order.
            random.Random(schema_order_seed).shuffle(self.tool_schemas)
        # Echo each model turn's thinking and text to stdout as the turn
        # ends, for watching a run live. The trace keeps them regardless.
        self.show_thinking = show_thinking
        # Step 1d on or off, off by default. Measured on 2026-10-28 (32B run
        # 5, 5b4e6f8): prefetching the option labels fixed nothing and lost
        # two questions, because the model read the attached set as the whole
        # world. AITX-00002's answer, ataluren, has no RxNorm concept and so
        # no label, and its absence from the set became "not approved";
        # AITX-00012's L-serine matched a homeopathic product listing, which
        # the model used to drop the option it had reasoned its way to in the
        # run before. A prefetch is safe for a source that answers the
        # question in full (VEP's consequence, a reference rule) and unsafe
        # for one that answers part of it. Kept behind the flag so the pass
        # is reproducible.
        self.step1_labels = step1_labels
        # Repeated-call guard scope: one conversation. The report chain sets
        # it per goal; the question answerer leaves it at 0.
        self.call_segment = 0
        self.last_trace: dict = {}

    def run(self, task_input: TaskInput) -> TaskOutput:
        """Run the agent on a single task input.

        Returns a TaskOutput with the answer and evidence. The run's working
        notes are kept on self.last_trace afterwards: every tool call with
        its arguments, timing and evidence, every point where the code
        stepped in (forced call, auto-chain, retries), and the final
        conversation as the model saw it.
        """
        self.last_trace = {
            "id": task_input.id,
            "tool_calls": [],
            "events": [],
            "messages": [],
            "validation": [],
            # Every model turn, including answers that never enter messages
            # (a turn-1 answer replaced by a forced lookup, the final answer,
            # the retries), so a run can be read end to end from the trace.
            "model_turns": [],
            "final_text": "",
            "answer": "",
        }

        # Step 1: Preprocess, then validate each variant and rewrite it onto
        # its current transcript version. Runs before the model so the prompt
        # and every later lookup use the checked form.
        context = preprocess_input(task_input)
        parsed_variants = context["parsed_variants"]
        validated = [
            validate_variant(g.transcript, g.variant_cdna)
            for g in task_input.patient.genotype
        ]
        self.last_trace["validation"] = [asdict(v) for v in validated]
        context["validated_variants"] = validated

        # Step 1b: the variant's predicted consequence (Ensembl VEP: consequence,
        # exon, coding length and frame, protein change) is needed for every
        # therapy decision, so it is fetched here for every validated variant
        # rather than left to the model to request. Its evidence is citable;
        # it does not count as the model having looked anything up.
        step1_evidence: list[EvidenceRecord] = []
        consequences = [self._predict_consequence(g.gene, v, step1_evidence) for g, v in zip(task_input.patient.genotype, validated)]
        # Step 1c: the rules that consequence calls for, attached rather than
        # left to the model to request (AITX-00001, five runs: it read VEP's
        # "1743 bases, a multiple of three" and ruled out every exon-skipping
        # drug from memory, never reading the splicing or exon-skipping rule).
        rules_text = self._rules_for(consequences, step1_evidence)

        # Step 2: For multiple choice, learn the options up front. They go
        # into the system prompt and later check the final answer. Empty
        # list means extraction was rejected: the model gets the generic
        # instruction and the answer passes through unchecked.
        options: list[str] = []
        if context["answer_format"] == "multiple_choice":
            options = self._extract_options(context["prompt"])

        # Step 1d: when the options are drug names, the labels that decide the
        # answer are fetched here too. Naming the tool was not enough: in five
        # 32B runs of AITX-00001 the model ruled out all five drugs from
        # memory, and the notice that listed every option with the exact call
        # was ignored twice. RxNorm decides which options are drugs, so a
        # mechanism class or "None of the above" fetches nothing.
        labels_text = self._option_labels(options, step1_evidence) if self.step1_labels else ""

        # Step 3: Build initial messages
        system_prompt = build_system_prompt(
            category=context["category"],
            parsed_variants=parsed_variants,
            clinical_context=context["clinical_context"],
            answer_format=context["answer_format"],
            date_submitted=context["date_submitted"],
            options=options,
            validated=validated,
            consequences=consequences,
            rules=rules_text,
            labels=labels_text,
        )
        user_message = build_user_message(context["prompt"], parsed_variants)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Step 4: ReAct loop. all_evidence is what the model itself retrieved;
        # the step-1 consequence evidence is added for citation at the end.
        all_evidence: list[EvidenceRecord] = []
        final_text = ""
        reminded = False
        tried: set[str] = set()
        term_reminded: list[str] = []  # answer terms the model was told no tool text contains
        none_reminded = False
        rule_reminded = False

        for iteration in range(MAX_ITERATIONS):
            logger.info(f"Agent iteration {iteration + 1}/{MAX_ITERATIONS}")

            response = generate(
                self.model,
                self.tokenizer,
                messages,
                tools=self.tool_schemas,
            )

            logger.debug(f"Model thinking: {response.thinking[:200]}...")
            logger.debug(f"Tool calls: {len(response.tool_calls)}")
            logger.debug(f"Text output: {response.text[:200]}...")
            self._record_turn("loop", iteration + 1, response)

            if not response.tool_calls:
                if not all_evidence and not reminded:
                    # The model answered without looking anything up. Code does
                    # not compose a query for it (run 2, 2026-10-07: a code-built
                    # GeneReviews query fetched a chapter for another gene). It
                    # says so once, names the recommended tool, and the model
                    # chooses the tool and writes the arguments.
                    reminded = True
                    self.last_trace["events"].append("evidence_reminder")
                    messages.append({"role": "assistant", "content": response.raw})
                    messages.append(self._harness_message(self._evidence_reminder(context)))
                    continue
                if not all_evidence and reminded and "reminder_ignored" not in self.last_trace["events"]:
                    # Told once, still no lookup: accept the answer, record it, no loop.
                    self.last_trace["events"].append("reminder_ignored")
                    logger.warning("Model answered without a tool call after the evidence reminder")

                # The answer turns on a general rule the model never read.
                unread = self._unread_rules(f"{response.text}\n{response.thinking}", {c["tool"] for c in self.last_trace["tool_calls"]})
                if unread and not rule_reminded and iteration < MAX_ITERATIONS - 1:
                    rule_reminded = True
                    self.last_trace["events"].append("rule_reminder:" + ",".join(unread))
                    messages.append({"role": "assistant", "content": response.raw})
                    messages.append(self._harness_message(self._rule_reminder(unread)))
                    continue

                # A name in the answer that no tool text contains came from
                # memory (32B run, 2026-10-13: ataluren on AITX-00002 and
                # eteplirsen on 00001 were named with no source naming them).
                # The model is told once, by term, and chooses the tool; code
                # writes no query. What is still unverified at the end is
                # removed if RxNorm knows it as a drug, kept and recorded if
                # it is a code the sources may express in other words.
                unverified = self._unverified_terms(response.text, context, options)
                if unverified and not term_reminded and iteration < MAX_ITERATIONS - 1:
                    term_reminded.extend(unverified)
                    self.last_trace["events"].append("unverified_term:" + ",".join(unverified))
                    messages.append({"role": "assistant", "content": response.raw})
                    messages.append(self._harness_message(self._unverified_term_reminder(unverified, context, tried)))
                    continue

                # A None answer while a recommended tool for the category has
                # not been consulted (32B run, 2026-10-13: AITX-00007 answered
                # "not applicable" after OMIM and ClinVar, never GeneReviews,
                # which held the answer). Code names the tools once; the model
                # decides whether to call one and with what.
                uncalled = self._uncalled_recommended(context, tried)
                if (self._is_none_answer(response.text, context) and uncalled
                        and not none_reminded and iteration < MAX_ITERATIONS - 1):
                    none_reminded = True
                    self.last_trace["events"].append("none_reminder:" + ",".join(uncalled))
                    messages.append({"role": "assistant", "content": response.raw})
                    messages.append(self._harness_message(self._none_reminder(uncalled, context, options)))
                    continue
                if none_reminded and self._is_none_answer(response.text, context):
                    self.last_trace["events"].append("none_reminder_ignored")

                # Empty-answer retry: model spent all tokens thinking, produced no text
                if not response.text and response.thinking and all_evidence:
                    logger.warning("Empty answer after think-loop, re-prompting...")
                    self.last_trace["events"].append("empty_answer_retry")
                    messages.append(self._harness_message(
                        "You already have tool results above. "
                        "Respond with ONLY the answer — no reasoning, no explanation. "
                        "Just the answer."
                    ))
                    # Generate with thinking disabled to avoid another loop
                    retry_response = generate(
                        self.model,
                        self.tokenizer,
                        messages,
                        tools=None,
                        max_new_tokens=256,
                        enable_thinking=False,
                    )
                    self._record_turn("empty_answer_retry", iteration + 1, retry_response)
                    if retry_response.text:
                        final_text = retry_response.text
                        break

                # Accept the final answer
                final_text = response.text
                break

            # Execute tool calls
            calls_before = len(self.last_trace["tool_calls"])
            for tc in response.tool_calls:
                tool = self.tools.get(tc.name)
                if tool is None:
                    # Unknown tool — inject error
                    logger.warning(f"Unknown tool: {tc.name}")
                    self.last_trace["events"].append(f"unknown_tool:{tc.name}")
                    messages.append({
                        "role": "assistant",
                        "content": response.raw,
                    })
                    messages.append(format_tool_result(
                        tc.name,
                        f"Error: Unknown tool '{tc.name}'. Available tools: {list(self.tools.keys())}",
                    ))
                    continue

                logger.info(f"Executing tool: {tc.name}({tc.arguments})")
                tried.add(tc.name)
                try:
                    summary, evidence = self._call_tool(tool, tc.name, tc.arguments, "model")
                    all_evidence.extend(evidence)
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    summary = f"Error executing {tc.name}: {e}"
                    evidence = []

                # Append assistant message (with tool call) and tool result
                messages.append({
                    "role": "assistant",
                    "content": response.raw,
                })
                messages.append(format_tool_result(tc.name, summary))

            # The lookups returned no evidence. Code does not call the next
            # tool itself (run 2, 2026-10-07: code-built chained calls fetched
            # a gene entry for a trial question and a consequence for a domain
            # question). It says so and names the next recommended tool; the
            # model decides and writes the arguments.
            turn_calls = self.last_trace["tool_calls"][calls_before:]
            # Only when the results are the adapters' own "nothing found"
            # replies: a reply with no evidence links can still be an answer,
            # such as GeneReviews' list of chapters to choose from (run 4,
            # AITX-00008: the notice steered the model away from the list).
            if not all_evidence and iteration < MAX_ITERATIONS - 1 and all(
                self._is_adapter_boilerplate(c.get("summary", "")) for c in turn_calls
            ):
                self.last_trace["events"].append("empty_result_notice")
                tried.update(tc.name for tc in response.tool_calls)
                messages.append(self._harness_message(self._empty_result_notice(context, tried)))

            # The last iteration ended in a tool call, so its result would go
            # unread (32B rerun, 2026-10-13, AITX-00001: the eteplirsen label
            # was fetched in turn 5 and never seen). One more generation with
            # tools disabled lets the model read it and answer; it cannot
            # start another lookup.
            if iteration == MAX_ITERATIONS - 1:
                self.last_trace["events"].append("final_answer_turn")
                messages.append(self._harness_message(
                    "No further tool calls are possible. Answer now from the tool results above, "
                    "in the required format, or answer None if they do not support an answer."
                ))
                last = generate(self.model, self.tokenizer, messages, tools=None)
                self._record_turn("final_answer", iteration + 1, last)
                final_text = last.text or response.text or "Unable to determine answer."
        else:
            if not final_text:
                final_text = "Unable to determine answer."

        self.last_trace["final_text"] = final_text

        # Step 5: Normalize the answer
        normalized = normalize_answer(
            final_text,
            context["answer_format"],
            context["prompt"],
            options,
        )

        # Step 6: Answer validation gate — retry once if invalid
        if not self._is_valid_answer(normalized, context["answer_format"], options):
            logger.warning(f"Invalid answer '{normalized}' for format {context['answer_format']}, retrying...")
            self.last_trace["events"].append("validation_retry")
            normalized = self._retry_for_valid_answer(
                messages, context["answer_format"], context["prompt"], options
            )

        # Step 6a: brand and salt names in a string answer become the
        # ingredient (run 3, AITX-00008: "DDAVP" for desmopressin), recorded
        # in the trace; RxNorm decides, the model's text is not guessed at.
        if context["answer_format"] == "string_match" and normalized:
            normalized, changes = normalise_drug_names(normalized)
            for c in changes:
                self.last_trace["events"].append(f"drug_name_normalised:{c['from']}->{c['to']}:rxcui{c['rxcui']}")

        # Step 6b: an identifier in the answer must occur in a tool result.
        # Run 2 (2026-10-07, AITX-00009): both trial searches failed and the
        # model answered NCT05678901, which no result contained.
        fabricated = self._fabricated_identifiers(normalized)
        if fabricated:
            logger.warning(f"Answer {normalized!r} names identifiers absent from every tool result: {fabricated}")
            self.last_trace["events"].append("fabricated_identifier:" + ",".join(fabricated))
            normalized = self._without_identifiers(normalized, fabricated)

        # Step 6c: settle the answer's terms, reminded or not. A term the
        # model was told about and that a tool text now contains is verified.
        # A term still in no tool text is removed when RxNorm knows it as a
        # drug, kept and recorded when not. This runs on every answer: a
        # name produced on the last loop turn got no reminder (32B run 3,
        # 2026-10-13, AITX-00002: "Ataluren" on turn 5, no tool text named
        # it, and it went out unchecked).
        still = self._unverified_terms(normalized, context, options)
        for term in term_reminded:
            if term not in still:
                self.last_trace["events"].append(f"unverified_term_verified:{term}")
        drugs = [t for t in still if is_drug_name(t)]
        kept = [t for t in still if t not in drugs]
        if drugs:
            logger.warning(f"Answer {normalized!r} names drugs absent from every tool result: {drugs}")
            self.last_trace["events"].append("unverified_term_removed:" + ",".join(drugs))
            normalized = self._without_identifiers(normalized, drugs)
        if kept:
            self.last_trace["events"].append("unverified_term_kept:" + ",".join(kept))
        self.last_trace["answer"] = normalized

        # Step 7: Build evidence items
        evidence_items = self._build_evidence(
            step1_evidence + all_evidence, normalized, task_input, messages
        )

        # Step 8: Return output
        self.last_trace["messages"] = messages
        return TaskOutput(
            id=task_input.id,
            response=normalized,
            evidence=evidence_items,
        )

    def _predict_consequence(self, gene: str, validated, evidence: list[EvidenceRecord]) -> str:
        """Run the Ensembl tool for one validated variant in step 1; text for the prompt."""
        tool = self.tools.get("query_ensembl")
        if tool is None:
            return "Predicted consequence: not run (Ensembl tool not registered)"
        if not validated.valid or not validated.genomic_variant:
            return "Predicted consequence: not run (variant did not validate)"
        kwargs = {"genomic_hgvs": validated.genomic_variant, "transcript": validated.transcript_variant.split(":")[0]}
        try:
            summary, found = self._call_tool(tool, "query_ensembl", kwargs, "step1")
        except Exception as e:
            logger.error(f"Step-1 consequence lookup failed for {gene}: {e}")
            return f"Predicted consequence: not available (Ensembl lookup failed: {e})"
        evidence.extend(found)
        return "Predicted consequence (Ensembl VEP, fetched automatically):\n" + summary

    # Which rule a fact in the retrieved consequence calls for. The trigger is
    # in the data, not in the model's judgment: pointing at the reference from
    # the tool text was ignored in every run to 2026-10-27, and the NOTES
    # instruction in 28 of 28 turns, so the rules are attached rather than
    # waited for.
    _RULE_TRIGGERS = [
        ("Splice sites removed", ["splicing", "exon-skipping"]),
        ("frameshift_variant", ["reading-frame", "nonsense-mediated-decay"]),
        ("stop_gained", ["nonsense-mediated-decay", "read-through"]),
        ("inframe_deletion", ["reading-frame"]),
        ("start_lost", ["reading-frame"]),
    ]
    # Wording in the model's own answer or thinking that rests on a general
    # rule; if the reference was never read, one notice names the topic.
    _RULE_WORDING = [
        (r"\bin[- ]frame\b|\bout of frame\b|\bframeshift\b|\breading frame\b", "reading-frame"),
        (r"\bexon skipping\b|\bskip(?:ping)? exon\b", "exon-skipping"),
        (r"\bnonsense[- ]mediated decay\b|\bNMD\b", "nonsense-mediated-decay"),
        (r"\bread[- ]through\b", "read-through"),
        (r"\bgain of function\b|\bloss of function\b|\bantagonist\b|\bagonist\b", "mechanism-direction"),
        (r"\bspliced?\b|\bsplice site\b", "splicing"),
    ]

    def _rules_for(self, consequences: list[str], evidence: list[EvidenceRecord]) -> str:
        """The reference rules the retrieved consequence calls for, as prompt
        text with their sources, fetched in step 1 like the consequence."""
        tool = self.tools.get("genetics_reference")
        if tool is None:
            return ""
        text = "\n".join(consequences)
        topics: list[str] = []
        for trigger, rules in self._RULE_TRIGGERS:
            if trigger in text:
                topics += [r for r in rules if r not in topics]
        parts = []
        for topic in topics[:3]:
            try:
                summary, found = self._call_tool(tool, "genetics_reference", {"topic": topic}, "step1")
            except Exception as e:
                logger.error(f"Step-1 rule lookup failed for {topic}: {e}")
                continue
            evidence.extend(found)
            parts.append(summary)
        if not parts:
            return ""
        self.last_trace["events"].append("rules_attached:" + ",".join(topics[:3]))
        return ("These rules of molecular genetics apply to the consequence above and are quotable "
                "evidence; read them before judging the variant's effect.\n\n" + "\n\n".join(parts))

    @staticmethod
    def _indications_only(summary: str) -> str:
        """A label's drug names and what it is indicated for. The rest of the
        label (contraindications, warnings, dosing) stays behind the tool, so
        five options do not crowd out the question; the model can still call
        search_fda_labels for the same drug to read it."""
        parts = summary.split("\n\n")
        if not parts or not parts[0].startswith("FDA Drug Label"):
            # openFDA's no-US-label coverage note; it goes in whole, because
            # its point is that silence is not evidence of absence.
            return summary
        return "\n\n".join([parts[0]] + [p for p in parts[1:] if p.startswith(("Drug:", "Indications"))])

    def _option_labels(self, options: list[str], evidence: list[EvidenceRecord]) -> str:
        """The FDA label of every option RxNorm knows as a drug, as prompt
        text with its source, fetched in step 1 like the consequence."""
        tool = self.tools.get("search_fda_labels")
        if tool is None or not options:
            return ""
        parts: list[str] = []
        fetched: list[str] = []
        for option in options:
            name = option.strip()
            if len(name) < 4 or name.lower() in ("none", "n/a", "none of the above"):
                continue
            # True only: RxNorm unreachable (None) must not fetch a label for
            # a protein domain or a GOF/LOF code.
            if is_drug_name(name) is not True:
                continue
            try:
                summary, found = self._call_tool(tool, "search_fda_labels", {"drug": name}, "step1")
            except Exception as e:
                logger.error(f"Step-1 label lookup failed for {name}: {e}")
                continue
            evidence.extend(found)
            parts.append(self._indications_only(summary))
            fetched.append(name)
            if len(fetched) >= MAX_OPTION_LABELS:
                break
        if not parts:
            return ""
        self.last_trace["events"].append("labels_attached:" + ",".join(fetched))
        return ("The options name drugs, and each label below was fetched automatically for one of them. "
                "A label states which variants, ages and conditions the drug is approved for, so it is the "
                "evidence that admits or rules out that option, and it is quotable. Call search_fda_labels "
                "for the same drug to read its contraindications, warnings and dosing.\n\n"
                + "\n\n".join(parts))

    def _unread_rules(self, text: str, called: set[str]) -> list[str]:
        """Rules the model's own wording rests on that it never read."""
        if "genetics_reference" in called or "genetics_reference" not in self.tools:
            return []
        topics = []
        for pattern, topic in self._RULE_WORDING:
            if re.search(pattern, text or "", re.I) and topic not in topics:
                topics.append(topic)
        return topics[:2]

    def _rule_reminder(self, topics: list[str]) -> str:
        return (
            f"Your answer turns on a general rule of molecular genetics ({', '.join(topics)}), and the "
            "genetics reference has not been read in this conversation. Call genetics_reference with "
            f"{{\"topic\": \"{topics[0]}\"}} and check your reasoning against the rule and its source, "
            "then answer."
        )

    def _record_turn(self, kind: str, iteration: int, response: ModelResponse) -> None:
        """Keep one model turn in the trace: kind is loop, empty_answer_retry or validation_retry."""
        self.last_trace["model_turns"].append({
            "kind": kind,
            "iteration": iteration,
            "thinking": response.thinking,
            "text": response.text,
            "tool_calls": [{"tool": tc.name, "arguments": tc.arguments} for tc in response.tool_calls],
            "thinking_truncated": getattr(response, "thinking_truncated", False),
        })
        if self.show_thinking:
            calls = ", ".join(f"{tc.name}({json.dumps(tc.arguments)})" for tc in response.tool_calls) or "none"
            print(
                f"\n[thinking, {kind} turn {iteration}]\n{response.thinking.strip() or '(no thinking)'}"
                f"\n[text] {response.text.strip() or '(empty)'}\n[tool calls] {calls}\n",
                flush=True,
            )

    def converse(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[BaseTool],
        max_turns: int = MAX_ITERATIONS,
        kind: str = "goal",
        require_evidence: bool = True,
        max_new_tokens: int = 2048,
        stop_on_repetition: bool = True,
    ) -> tuple[str, list[EvidenceRecord]]:
        """One short conversation with its own tool set: the unit of the
        report chain (decided 2026-10-15). Same loop as the question
        answerer, with the evidence reminder, tool results in the tool role,
        harness notices in the system role, and an answer-only turn when the
        last turn ends in a tool call. Tool calls go to self.last_trace, so
        the caller sets that up. Returns (final text, evidence the model
        retrieved)."""
        by_name = {t.name: t for t in tools}
        schemas = [t.schema() for t in tools]
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        evidence: list[EvidenceRecord] = []
        reminded = False
        response = None
        for turn in range(max_turns):
            response = generate(self.model, self.tokenizer, messages, tools=schemas, max_new_tokens=max_new_tokens,
                                stop_on_repetition=stop_on_repetition)
            self._record_turn(kind, turn + 1, response)
            if not response.tool_calls:
                if require_evidence and not evidence and not reminded and turn < max_turns - 1:
                    reminded = True
                    self.last_trace["events"].append("evidence_reminder")
                    messages.append({"role": "assistant", "content": response.raw})
                    messages.append(self._harness_message(
                        "You answered without looking anything up. Evidence from a tool is required; "
                        "call the tool that fits, choosing the arguments from the patient record. Then answer."
                    ))
                    continue
                self.last_trace["messages"].extend(messages)
                return response.text, evidence
            messages.append({"role": "assistant", "content": response.raw})
            for tc in response.tool_calls:
                tool = by_name.get(tc.name)
                if tool is None:
                    self.last_trace["events"].append(f"unknown_tool:{tc.name}")
                    messages.append(format_tool_result(tc.name, f"Error: unknown tool '{tc.name}'. Available: {list(by_name)}"))
                    continue
                try:
                    summary, found = self._call_tool(tool, tc.name, tc.arguments, "model")
                    evidence.extend(found)
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    summary = f"Error executing {tc.name}: {e}"
                messages.append(format_tool_result(tc.name, summary))
        # Every turn called a tool: read the last result and answer.
        self.last_trace["events"].append("final_answer_turn")
        messages.append(self._harness_message(
            "No further tool calls are possible. Answer now from the tool results above, in the required format."
        ))
        last = generate(self.model, self.tokenizer, messages, tools=None, max_new_tokens=max_new_tokens,
                        stop_on_repetition=stop_on_repetition)
        self._record_turn("final_answer", max_turns, last)
        self.last_trace["messages"].extend(messages)
        return last.text or (response.text if response else ""), evidence

    def _call_tool(
        self,
        tool: BaseTool,
        name: str,
        kwargs: dict,
        origin: str,
    ) -> tuple[str, list[EvidenceRecord]]:
        """Run a tool and record the call in the trace.

        origin says who asked for it: "model", "forced" or "auto_chain".
        An exception is recorded, then re-raised so each call site keeps
        its own handling.
        """
        # A call the model has already made returns its first result with a
        # note, instead of being made again (32B run 4, 2026-10-27, AITX-00012:
        # the same three PharmGKB lookups were repeated four times because each
        # returned an empty record, spending four turns on nothing).
        repeat = self._earlier_identical_call(name, kwargs)
        if repeat is not None:
            self.last_trace["events"].append(f"repeated_call:{name}")
            logger.info(f"Repeated call to {name}({kwargs}); returning the first result")
            self.last_trace["tool_calls"].append({"tool": name, "arguments": kwargs, "origin": origin,
                                                 "segment": self.call_segment,
                                                 "seconds": 0.0, "repeat_of": repeat["seconds"],
                                                 "evidence_urls": repeat.get("evidence_urls", []),
                                                 "summary": repeat.get("summary", "")})
            return (
                f"{repeat.get('summary', '')}\n\n[This is the result of the identical earlier call to {name}; "
                "it has not changed. A result that lacks the fact you need is not evidence of absence: try a "
                "different tool or different arguments.]",
                [],
            )

        entry = {"tool": name, "arguments": kwargs, "origin": origin, "segment": self.call_segment}
        start = time.time()
        try:
            summary, evidence = tool.execute(**kwargs)
        except Exception as e:
            entry["error"] = str(e)
            raise
        finally:
            entry["seconds"] = round(time.time() - start, 2)
            self.last_trace["tool_calls"].append(entry)
        entry["evidence_urls"] = [r.url for r in evidence]
        entry["summary"] = summary
        return summary, evidence

    def _earlier_identical_call(self, name: str, kwargs: dict) -> dict | None:
        """The first call to this tool with these arguments in this
        conversation. Scoped to the segment: the report chain bumps it per
        goal, because a later goal legitimately re-reads the same chapter and
        its records are checked against its own calls."""
        for call in self.last_trace.get("tool_calls", []):
            if (call["tool"] == name and call.get("arguments") == kwargs and "summary" in call
                    and "repeat_of" not in call and call.get("segment") == self.call_segment):
                return call
        return None

    # Identifier shapes the check recognises. Drug names are not shaped and
    # are left to the citation check that follows later.
    _IDENTIFIER = re.compile(
        r"\b(?:NCT\d{8}|rs\d{3,}|NBK\d+|(?:NM|NP|NR|NC)_\d+(?:\.\d+)?|ENS[GT]\d{11}|PA\d{5,}|MONDO:\d+)\b"
    )

    def _fabricated_identifiers(self, answer: str) -> list[str]:
        """Identifiers in the answer that no tool result or evidence URL contains."""
        found = self._IDENTIFIER.findall(answer or "")
        if not found:
            return []
        haystack = "\n".join(
            [c.get("summary", "") for c in self.last_trace["tool_calls"]]
            + [u for c in self.last_trace["tool_calls"] for u in c.get("evidence_urls", [])]
        ).upper()
        return [i for i in dict.fromkeys(found) if i.upper() not in haystack]

    @staticmethod
    def _without_identifiers(answer: str, identifiers: list[str]) -> str:
        """The answer with those identifiers removed; "None" if nothing is left."""
        text = answer
        for i in identifiers:
            text = re.sub(r"\b" + re.escape(i) + r"\b", "", text)
        text = re.sub(r"\s*,\s*,+", ",", text)
        text = text.strip(" ,;\n\t")
        return text or "None"

    def _unverified_terms(self, answer: str, context: dict, options: list[str] | None) -> list[str]:
        """Answer terms (string or multiple-choice formats) that no tool text or evidence URL contains."""
        if context["answer_format"] not in ("string_match", "multiple_choice") or not answer:
            return []
        items = [i.strip(" .;\n\t\"'") for i in answer.split(",")]
        items = [
            i for i in items
            if len(i) >= 3 and i.lower() not in ("none", "n/a", "not applicable", "unknown")
            and not self._IDENTIFIER.fullmatch(i)  # identifiers have their own check (step 6b)
        ]
        if not items:
            return []
        haystack = "\n".join(
            [c.get("summary", "") for c in self.last_trace["tool_calls"]]
            + [u for c in self.last_trace["tool_calls"] for u in c.get("evidence_urls", [])]
        ).lower()
        return [i for i in dict.fromkeys(items) if i.lower() not in haystack]

    @staticmethod
    def _is_none_answer(text: str, context: dict) -> bool:
        """A string or multiple-choice answer that says there is nothing."""
        if context["answer_format"] not in ("string_match", "multiple_choice"):
            return False
        return (text or "").strip().strip(".").lower() in ("none", "not applicable", "n/a", "no approved therapy")

    def _uncalled_recommended(self, context: dict, tried: set[str]) -> list[str]:
        priority = TOOL_PRIORITY.get(context["category"], ["search_pubmed"])
        return [t for t in priority if t in self.tools and t not in tried]

    def _none_reminder(self, uncalled: list[str], context: dict, options: list[str] | None = None) -> str:
        """One notice naming what has not been consulted. When the question
        offers named agents as options, name them and the lookup that settles
        each: the label states which variants a drug is indicated for (32B
        run 4, 2026-10-26, AITX-00001: after this notice the model called
        ClinVar, first in the category's order and unable to answer, while
        the eteplirsen label was third and never called)."""
        agents = [o for o in (options or []) if o.strip().lower() not in ("none", "n/a") and len(o.strip()) >= 4]
        by_option = ""
        if agents and "search_fda_labels" in self.tools:
            by_option = (
                f"The options name agents: {', '.join(agents)}. A drug's own label states which variants and "
                f"ages it is indicated for, so call search_fda_labels with drug set to each option still in "
                f"question (for example {{\"drug\": \"{agents[0]}\"}}) before answering None; if a label search "
                f"returns nothing, search PubMed for that drug with the gene. "
            )
        return (
            "You answered None, but the recommended tools for this question have not all been "
            f"consulted: {', '.join(uncalled)}. A fact absent from the sources you read is not "
            "evidence of absence. " + by_option + "Call the tools that can settle it, choosing the arguments "
            "from the patient record and the question, or confirm None if the evidence you have already "
            "rules the alternatives out."
        )

    def _unverified_term_reminder(self, terms: list[str], context: dict, tried: set[str]) -> str:
        """One message naming the unverified term(s) and the recommended tools not yet called."""
        priority = TOOL_PRIORITY.get(context["category"], ["search_pubmed"])
        uncalled = [t for t in priority if t in self.tools and t not in tried]
        names = ", ".join(terms)
        where = (
            f"Recommended tools for this question not yet called: {', '.join(uncalled)}. "
            if uncalled else "Every recommended tool for this question has been called; another tool may fit. "
        )
        code_hint = (
            "If it is an abbreviation or a code (a domain name, a mechanism class), search PubMed for it "
            "together with the gene to find what it stands for, then match that to the tool results. "
            if context["answer_format"] == "multiple_choice" else ""
        )
        # Name the call that can confirm a drug name: its own label, then the
        # literature if no United States label exists (32B run 4: ataluren's
        # label search returned nothing and the model stopped there).
        drug_hint = ""
        if "search_fda_labels" in self.tools:
            drug_hint = (
                f"If it is a drug, its label states what it treats and in which variants: call "
                f"search_fda_labels with {{\"drug\": \"{terms[0]}\"}}, and if that returns no label, search "
                f"PubMed for it with the gene, since openFDA holds only drugs with a United States label. "
            )
        return (
            f"Your answer names {names}, which appears in no tool result of this conversation. "
            "Confirm it with a tool call before answering, choosing the arguments from the patient "
            "record and the question. " + drug_hint + code_hint + where + "Then answer from the evidence you have."
        )

    @staticmethod
    def _harness_message(text: str) -> dict:
        """A notice from the harness, as a system message.

        Qwen3's template drops the thinking of every assistant turn before
        the last user message; a notice sent as a user message cut the
        model off from its own reasoning at exactly the moment it was being
        asked to reconsider (32B run 3, 2026-10-13: reversals at reminders,
        and the model calling its earlier answer "the user's"). A system
        message is rendered as a system turn and leaves the thinking in
        place (checked against the template 2026-10-15).
        """
        return {"role": "system", "content": text}

    def _evidence_reminder(self, context: dict) -> str:
        """One message telling the model a lookup is required and which tool is recommended first."""
        priority = TOOL_PRIORITY.get(context["category"], ["search_pubmed"])
        first = next((t for t in priority if t in self.tools), None)
        recommended = (
            f"The recommended first tool for this question is {first}; call it, or another tool "
            f"that fits better, choosing the arguments from the patient record. "
            if first else "Call the tool that fits the question, choosing the arguments from the patient record. "
        )
        return (
            "You answered without looking anything up. Evidence from a tool is required "
            "before the final answer. " + recommended + "Then answer."
        )

    def _empty_result_notice(self, context: dict, tried: set[str]) -> str:
        """One message saying the lookups found nothing and which untried tool is recommended next."""
        priority = TOOL_PRIORITY.get(context["category"], ["search_pubmed"])
        nxt = next((t for t in priority if t in self.tools and t not in tried), None)
        recommended = (
            f"The next recommended tool for this question is {nxt}; call it, or another tool that "
            f"fits better, choosing the arguments from the patient record. "
            if nxt else "Every recommended tool for this question has been tried. "
        )
        return (
            "The lookup returned no evidence. A result that lacks the fact you need is not "
            "evidence of absence. " + recommended
            + "Answer only when a tool result supports the answer, or state that no evidence was found."
        )

    @staticmethod
    def _is_valid_answer(answer: str, answer_format: str, options: list[str] | None = None) -> bool:
        """Check if a normalized answer is valid for its format.

        For multiple choice the answer must be one of the verified options when
        there are any; with no verified options any non-empty string is
        accepted, since there is nothing to check against.
        """
        if not answer or not answer.strip():
            return False
        if answer_format == "binary" and answer not in ("Yes", "No"):
            return False
        if answer_format == "numeric_match":
            try:
                float(answer)
            except ValueError:
                return False
        if answer_format == "multiple_choice" and options and answer not in options:
            return False
        return True

    def _extract_options(self, prompt: str) -> list[str]:
        """Ask the model to list a multiple-choice prompt's options, then verify them.

        The model proposes; code checks every item against the prompt text.
        A rejected or unparseable list is logged with the raw output and
        yields [], so the question proceeds without option validation.
        """
        try:
            response = generate(
                self.model,
                self.tokenizer,
                [{"role": "user", "content": OPTION_EXTRACTION_INSTRUCTION + prompt}],
                tools=None,
                max_new_tokens=256,
                enable_thinking=False,
            )
        except Exception as e:
            logger.error(f"Option extraction failed: {e}")
            return []

        candidates = parse_option_json(response.text)
        if candidates is None:
            logger.warning(f"Option extraction rejected: unparseable output {response.text!r} for prompt {prompt!r}")
            return []

        options, reason = verify_options(candidates, prompt)
        if reason:
            logger.warning(f"Option extraction rejected: {reason}; raw output {response.text!r} for prompt {prompt!r}")
            return []

        logger.info(f"Verified options: {options}")
        return options

    def _retry_for_valid_answer(
        self,
        messages: list[dict],
        answer_format: str,
        prompt: str,
        options: list[str] | None = None,
    ) -> str:
        """Re-prompt the model once with thinking disabled to get a valid answer."""
        options = options or []
        format_hint = {
            "binary": "Respond with exactly 'Yes' or 'No'.",
            "multiple_choice": (
                "Respond with exactly one of these options, spelled as given: "
                + "; ".join(options) + "."
                if options else "Respond with the exact text of one option."
            ),
            "numeric_match": "Respond with exactly one number.",
            "string_match": "Respond with a concise, exact term.",
        }.get(answer_format, "Respond concisely.")

        retry_messages = messages + [self._harness_message(
            f"Your previous answer was invalid. {format_hint} "
            "Use the tool results already provided. "
            "Respond with ONLY the answer — nothing else."
        )]

        try:
            response = generate(
                self.model,
                self.tokenizer,
                retry_messages,
                tools=None,
                max_new_tokens=256,
                enable_thinking=False,
            )
            self._record_turn("validation_retry", 0, response)
            normalized = normalize_answer(response.text, answer_format, prompt, options)
            if self._is_valid_answer(normalized, answer_format, options):
                return normalized
        except Exception as e:
            logger.error(f"Answer validation retry failed: {e}")

        # If retry also failed, return best-effort
        return normalize_answer("Unable to determine answer.", answer_format, prompt)

    def _build_evidence(
        self,
        evidence_records: list[EvidenceRecord],
        answer: str,
        task_input: TaskInput,
        messages: list[dict],
    ) -> list[EvidenceItem]:
        """Build evidence items with justifications.

        Always returns at least one EvidenceItem with a real URL.
        If no evidence was collected from tools, performs a fallback
        PubMed search to obtain a real URL.
        """
        if not evidence_records:
            # Fallback: do a PubMed search to get a real URL
            evidence_records = self._fallback_evidence(task_input)

        if not evidence_records:
            # Absolute last resort — should rarely happen
            gene = task_input.patient.genotype[0].gene if task_input.patient.genotype else "unknown"
            return [
                EvidenceItem(
                    source=f"https://pubmed.ncbi.nlm.nih.gov/?term={gene}",
                    time_accessed=int(time.time()),
                    justification=f"PubMed search page for {gene} related to: {task_input.question.prompt[:200]}",
                )
            ]

        # Deduplicate by URL, keeping the first occurrence
        seen_urls: set[str] = set()
        unique_records: list[EvidenceRecord] = []
        for record in evidence_records:
            if record.url not in seen_urls:
                seen_urls.add(record.url)
                unique_records.append(record)
        evidence_records = unique_records

        # Each citation is justified from the text the tool returned for it,
        # and the quote the model gives is checked against that text. Run 2
        # (2026-10-07): the model was shown only URLs, so every justification
        # described what a supporting source would say; none was grounded.
        records = evidence_records[:5]  # Cap at 5 evidence items
        source_texts = [self._source_text_for(r.url) for r in records]
        justifications = self._generate_justifications(records, source_texts, answer, task_input)

        # Three outcomes per source. Supported: the quote is verbatim in the
        # source's text and is cited. Declined: the model answered NOT
        # SUPPORTED, so the source is not evidence for this answer and is
        # left out (run 5: the step-1 VEP page was declined on five drug and
        # trial questions, where the consequence played no part). Unsupported:
        # the model quoted words the text does not contain; the source stays,
        # marked, because the model did claim it as evidence.
        items = []
        declined: list[EvidenceItem] = []
        for i, record in enumerate(records):
            quote, sentence = justifications[i] if i < len(justifications) else ("", "")
            if quote is None:
                self.last_trace["events"].append(f"citation_declined:{record.url}")
                logger.info(f"Citation declined by the model (NOT SUPPORTED): {record.url}")
                declined.append(EvidenceItem(
                    source=record.url, time_accessed=record.time_accessed, justification=UNSUPPORTED_JUSTIFICATION,
                ))
                continue
            supported = (
                bool(quote)
                and self._quote_in_text(quote, source_texts[i])
                and not self._is_adapter_boilerplate(quote)
            )
            if supported:
                justification = f'"{quote}" {sentence}'.strip()
            else:
                self.last_trace["events"].append(f"citation_unsupported:{record.url}")
                logger.warning(f"Citation not supported by its retrieved text: {record.url} (quote={quote!r})")
                justification = UNSUPPORTED_JUSTIFICATION
            if len(justification) > 500:
                justification = justification[:497] + "..."
            items.append(EvidenceItem(
                source=record.url,
                time_accessed=record.time_accessed,
                justification=justification,
            ))

        if not items and declined:
            # The output must carry at least one source. Keep the declined
            # ones, marked, rather than invent a citation.
            self.last_trace["events"].append("all_citations_declined")
            logger.warning("Every source was declined; keeping them marked as not stating the answer")
            return declined
        return items

    def _source_text_for(self, url: str) -> str:
        """The tool result text that returned this URL, from the run's trace."""
        for call in self.last_trace.get("tool_calls", []):
            if url in call.get("evidence_urls", []):
                return call.get("summary", "")
        return ""

    # Sentences the adapters themselves write when a lookup finds nothing.
    # They are verbatim in the tool text, so the quote check would pass them,
    # but they report an empty search, not a record (run 3, AITX-00010).
    _BOILERPLATE = re.compile(
        r"^\W*(?:No (?:clinical trials|ClinVar record|FDA drug labels?|UniProt entry|PubMed articles|OMIM entries|PharmGKB entry|Ensembl data)\b"
        r"|[\w. ]+ returned nothing\b"
        r"|[\w. ]+ has no (?:chapter|study|target|dosage curation)\b"
        r"|[\w. ]+ (?:API|lookup) error\b)",
        re.IGNORECASE,
    )

    @classmethod
    def _is_adapter_boilerplate(cls, quote: str) -> bool:
        """True if the quote is an adapter's own 'nothing found' sentence rather than a record."""
        return bool(cls._BOILERPLATE.match(quote.strip()))

    @staticmethod
    def _quote_in_text(quote: str, text: str) -> bool:
        """True if the quote occurs in the text, ignoring case, spacing and quote marks."""
        def norm(t: str) -> str:
            # Superscript digits plain and whitespace removed: a label prints
            # "m 2" where the model writes "m²" (first live report, 2026-10-20).
            t = t.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
            t = t.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
            return re.sub(r"\s+", "", t).lower()
        # A quote may end its sentence with a full stop the source continues
        # past (32B run 2, 2026-10-13, AITX-00006: the lumasiran indication
        # sentence runs into "[see Clinical Pharmacology ...]" in the label).
        q = norm(quote).strip('"\'').rstrip(".,;: ")
        return bool(q) and len(q) >= 15 and q in norm(text)

    def _generate_justifications(
        self,
        evidence_records: list[EvidenceRecord],
        source_texts: list[str],
        answer: str,
        task_input: TaskInput,
    ) -> list[tuple[str, str]]:
        """For each source, (verbatim quote, one sentence); (None, "") when the model declines, ("", "") when unreadable.

        One source per model call. Shown all sources at once, the model
        attached one source's sentence to another's link (runs 3 and 4:
        the same trial summary under three trials, a GeneReviews sentence
        under a ClinVar link); with a single text in view that cannot happen.
        """
        out: list[tuple[str, str]] = []
        for record, text in zip(evidence_records, source_texts):
            shown = text if len(text) <= SOURCE_TEXT_CHARS else text[:SOURCE_TEXT_CHARS - 3] + "..."
            prompt = (
                f"Question: {task_input.question.prompt}\n"
                f"Answer given: {answer}\n\n"
                f"SOURCE: {record.url}\n{shown or '(no text retrieved for this source)'}\n\n"
                "Copy one sentence VERBATIM from this source's text that supports the answer, then "
                "add one sentence saying how it supports it. If nothing in this text supports the "
                "answer, write exactly NOT SUPPORTED. Never quote words that are not in the text.\n\n"
                "Format:\nQUOTE: <verbatim sentence or NOT SUPPORTED>\nWHY: <one sentence>"
            )
            messages = [
                {"role": "system", "content": "You cite only what the given text says. Output only the QUOTE and WHY lines."},
                {"role": "user", "content": prompt},
            ]
            try:
                response = generate(
                    self.model, self.tokenizer, messages, tools=None, max_new_tokens=300, enable_thinking=False,
                )
            except Exception as e:
                logger.error(f"Failed to generate a justification for {record.url}: {e}")
                out.append(("", ""))
                continue
            self._record_turn("justifications", len(out) + 1, response)
            out.append(self._parse_justification(response.text))
        return out

    @staticmethod
    def _parse_justification(text: str) -> tuple[str, str]:
        q = re.search(r"QUOTE:\s*(.*?)(?:\n\s*WHY:|\Z)", text, re.S)
        w = re.search(r"WHY:\s*(.*)", text, re.S)
        quote = (q.group(1) if q else "").strip().strip('"\u201c\u201d')
        why = re.sub(r"\s+", " ", w.group(1)).strip() if w else ""
        if quote.upper().startswith("NOT SUPPORTED"):
            return None, ""  # declined: the model says the text does not support the answer
        if not quote:
            return "", ""  # unreadable reply: treated as an unsupported claim
        return quote, why

    def _fallback_evidence(self, task_input: TaskInput) -> list[EvidenceRecord]:
        """Fallback: search PubMed to get at least one real evidence URL."""
        pubmed_tool = self.tools.get("search_pubmed")
        if pubmed_tool is None:
            return []
        gene = task_input.patient.genotype[0].gene if task_input.patient.genotype else ""
        query = f"{gene} {task_input.question.prompt[:60]}"
        try:
            _, evidence = pubmed_tool.execute(query=query, max_results=2)
            return evidence
        except Exception as e:
            logger.error(f"Fallback PubMed search failed: {e}")
            return []

