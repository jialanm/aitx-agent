"""ReAct agent loop — core orchestrator."""

from __future__ import annotations

import logging
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
from .drug_names import normalise_drug_names

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5
# How much of a source's retrieved text the justification step is shown.
SOURCE_TEXT_CHARS = 1500
UNSUPPORTED_JUSTIFICATION = "This record was retrieved and consulted; its returned text does not state the answer."


class Agent:
    """ReAct agent that uses tools to answer precision medicine questions."""

    def __init__(self, model, tokenizer, tools: list[BaseTool]):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = {tool.name: tool for tool in tools}
        self.tool_schemas = [tool.schema() for tool in tools]
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

        # Step 2: For multiple choice, learn the options up front. They go
        # into the system prompt and later check the final answer. Empty
        # list means extraction was rejected: the model gets the generic
        # instruction and the answer passes through unchecked.
        options: list[str] = []
        if context["answer_format"] == "multiple_choice":
            options = self._extract_options(context["prompt"])

        # Step 3: Build initial messages
        system_prompt = build_system_prompt(
            category=context["category"],
            parsed_variants=parsed_variants,
            clinical_context=context["clinical_context"],
            answer_format=context["answer_format"],
            date_submitted=context["date_submitted"],
            options=options,
            validated=validated,
        )
        user_message = build_user_message(context["prompt"], parsed_variants)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Step 4: ReAct loop
        all_evidence: list[EvidenceRecord] = []
        final_text = ""
        reminded = False
        tried: set[str] = set()

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
                    messages.append({"role": "user", "content": self._evidence_reminder(context)})
                    continue
                if not all_evidence and reminded:
                    # Told once, still no lookup: accept the answer, record it, no loop.
                    self.last_trace["events"].append("reminder_ignored")
                    logger.warning("Model answered without a tool call after the evidence reminder")

                # Empty-answer retry: model spent all tokens thinking, produced no text
                if not response.text and response.thinking and all_evidence:
                    logger.warning("Empty answer after think-loop, re-prompting...")
                    self.last_trace["events"].append("empty_answer_retry")
                    messages.append({
                        "role": "user",
                        "content": (
                            "You already have tool results above. "
                            "Respond with ONLY the answer — no reasoning, no explanation. "
                            "Just the answer."
                        ),
                    })
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
                messages.append({"role": "user", "content": self._empty_result_notice(context, tried)})

            # If this was the last iteration and we still have tool calls
            if iteration == MAX_ITERATIONS - 1:
                final_text = response.text or "Unable to determine answer."
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
        self.last_trace["answer"] = normalized

        # Step 7: Build evidence items
        evidence_items = self._build_evidence(
            all_evidence, normalized, task_input, messages
        )

        # Step 8: Return output
        self.last_trace["messages"] = messages
        return TaskOutput(
            id=task_input.id,
            response=normalized,
            evidence=evidence_items,
        )

    def _record_turn(self, kind: str, iteration: int, response: ModelResponse) -> None:
        """Keep one model turn in the trace: kind is loop, empty_answer_retry or validation_retry."""
        self.last_trace["model_turns"].append({
            "kind": kind,
            "iteration": iteration,
            "thinking": response.thinking,
            "text": response.text,
            "tool_calls": [{"tool": tc.name, "arguments": tc.arguments} for tc in response.tool_calls],
        })

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
        entry = {"tool": name, "arguments": kwargs, "origin": origin}
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

        retry_messages = messages + [{
            "role": "user",
            "content": (
                f"Your previous answer was invalid. {format_hint} "
                "Use the tool results already provided. "
                "Respond with ONLY the answer — nothing else."
            ),
        }]

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

        items = []
        for i, record in enumerate(records):
            quote, sentence = justifications[i] if i < len(justifications) else ("", "")
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
            t = t.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
            return re.sub(r"\s+", " ", t).strip().lower()
        q = norm(quote).strip('"\'')
        return bool(q) and len(q) >= 15 and q in norm(text)

    def _generate_justifications(
        self,
        evidence_records: list[EvidenceRecord],
        source_texts: list[str],
        answer: str,
        task_input: TaskInput,
    ) -> list[tuple[str, str]]:
        """For each source, (verbatim quote, one sentence) or ("", "") when unsupported.

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
        if not quote or quote.upper().startswith("NOT SUPPORTED"):
            return "", ""
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

