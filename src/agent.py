"""ReAct agent loop — core orchestrator."""

from __future__ import annotations

import logging
import re
import time

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

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5


class Agent:
    """ReAct agent that uses tools to answer precision medicine questions."""

    def __init__(self, model, tokenizer, tools: list[BaseTool]):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = {tool.name: tool for tool in tools}
        self.tool_schemas = [tool.schema() for tool in tools]

    def run(self, task_input: TaskInput) -> TaskOutput:
        """Run the agent on a single task input.

        Returns a TaskOutput with the answer and evidence.
        """
        # Step 1: Preprocess
        context = preprocess_input(task_input)
        parsed_variants = context["parsed_variants"]

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
        )
        user_message = build_user_message(context["prompt"], parsed_variants)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Step 4: ReAct loop
        all_evidence: list[EvidenceRecord] = []
        final_text = ""

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

            if not response.tool_calls:
                if iteration == 0 and not all_evidence:
                    # Model tried to answer without any tool calls.
                    # Force the first recommended tool call for this category.
                    forced = self._force_first_tool_call(
                        context, task_input, messages, all_evidence
                    )
                    if forced:
                        # Re-prompt the model with tool results now available
                        messages.append({
                            "role": "user",
                            "content": (
                                "Now use the tool results above to answer the question. "
                                "Respond with ONLY the answer."
                            ),
                        })
                        continue

                # Empty-answer retry: model spent all tokens thinking, produced no text
                if not response.text and response.thinking and all_evidence:
                    logger.warning("Empty answer after think-loop, re-prompting...")
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
                    if retry_response.text:
                        final_text = retry_response.text
                        break

                # Accept the final answer
                final_text = response.text
                break

            # Execute tool calls
            for tc in response.tool_calls:
                tool = self.tools.get(tc.name)
                if tool is None:
                    # Unknown tool — inject error
                    logger.warning(f"Unknown tool: {tc.name}")
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
                    summary, evidence = tool.execute(**tc.arguments)
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

            # Auto-chain: if tool returned no evidence, try next priority tool
            if not all_evidence and iteration < MAX_ITERATIONS - 1:
                self._auto_chain_next_tool(
                    context, task_input, messages, all_evidence,
                    used_tools={tc.name for tc in response.tool_calls},
                )

            # If this was the last iteration and we still have tool calls
            if iteration == MAX_ITERATIONS - 1:
                final_text = response.text or "Unable to determine answer."
        else:
            if not final_text:
                final_text = "Unable to determine answer."

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
            normalized = self._retry_for_valid_answer(
                messages, context["answer_format"], context["prompt"], options
            )

        # Step 7: Build evidence items
        evidence_items = self._build_evidence(
            all_evidence, normalized, task_input, messages
        )

        # Step 8: Return output
        return TaskOutput(
            id=task_input.id,
            response=normalized,
            evidence=evidence_items,
        )

    def _force_first_tool_call(
        self,
        context: dict,
        task_input: TaskInput,
        messages: list[dict],
        all_evidence: list[EvidenceRecord],
    ) -> bool:
        """Force a tool call when the model skips tools on first iteration.

        Returns True if a tool was successfully called.
        """
        category = context["category"]
        priority = TOOL_PRIORITY.get(category, ["search_pubmed"])
        parsed_variants = context["parsed_variants"]

        # Find the first available tool from the priority list
        for tool_name in priority:
            tool = self.tools.get(tool_name)
            if tool is None:
                continue

            # Build arguments based on tool type and input data
            kwargs = self._build_auto_args(tool_name, parsed_variants, task_input)
            logger.info(f"Forcing tool call: {tool_name}({kwargs})")

            try:
                summary, evidence = tool.execute(**kwargs)
                all_evidence.extend(evidence)
                messages.append(format_tool_result(tool_name, summary))
                return True
            except Exception as e:
                logger.error(f"Forced tool call failed: {e}")
                continue

        return False

    def _build_auto_args(
        self,
        tool_name: str,
        parsed_variants: list,
        task_input: TaskInput,
    ) -> dict:
        """Build tool arguments automatically from input data."""
        gene = parsed_variants[0].gene if parsed_variants else ""
        variant_cdna = parsed_variants[0].variant_cdna if parsed_variants else ""
        transcript = parsed_variants[0].transcript if parsed_variants else ""
        clinical_context = task_input.patient.clinical_context

        if tool_name == "search_clinvar":
            return {"gene": gene, "variant": variant_cdna}
        elif tool_name == "query_ensembl":
            hgvs = f"{transcript}:{variant_cdna}" if transcript else variant_cdna
            return {"hgvs_cdna": hgvs, "gene": gene}
        elif tool_name == "search_genereviews":
            # Extract condition from clinical_context (first sentence, strip age/sex preamble)
            condition = self._extract_condition(clinical_context)
            args = {"gene": gene}
            if condition:
                args["condition"] = condition
            return args
        elif tool_name == "search_clinical_trials":
            return {"query": f"{gene} {clinical_context.split('.')[0][:50]}"}
        elif tool_name == "search_pubmed":
            prompt_keywords = task_input.question.prompt[:80]
            return {"query": f"{gene} {prompt_keywords}"}
        elif tool_name == "query_uniprot":
            protein_pos = parsed_variants[0].protein_position if parsed_variants else None
            args: dict = {"gene": gene}
            if protein_pos:
                try:
                    args["protein_position"] = int(protein_pos)
                except (ValueError, TypeError):
                    pass
            return args
        elif tool_name == "search_pharmgkb":
            return {"gene": gene}
        elif tool_name == "search_omim":
            return {"gene": gene}
        elif tool_name == "search_fda_labels":
            # Try to extract drug name from the question prompt
            prompt = task_input.question.prompt
            args = {"drug": gene}  # fallback: use gene as search term
            # Common patterns: "treatment with X", "eligible for X"
            import re as _re
            drug_match = _re.search(
                r"(?:treatment with|eligible for|treated with|therapy with)\s+(\w[\w\s\-/()]+?)(?:\?|\.|,|based)",
                prompt,
                _re.IGNORECASE,
            )
            if drug_match:
                args["drug"] = drug_match.group(1).strip()
                args["indication"] = gene
            return args
        else:
            return {"query": gene}

    @staticmethod
    def _extract_condition(clinical_context: str) -> str:
        """Extract the likely condition name from clinical_context.

        Looks for 'diagnosed with X' or 'diagnosis of X' patterns,
        falling back to the first sentence minus age/sex preamble.
        """
        # Try "diagnosed with X" or "diagnosis of X"
        m = re.search(
            r"diagnos(?:ed with|is of)\s+(.+?)(?:\.|,|\s+presents|\s+who|\s+characterized)",
            clinical_context,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()

        # Fallback: first sentence, drop leading "A XX-year-old ..." preamble
        first_sent = clinical_context.split(".")[0]
        # Remove age/sex preamble like "A 14-year-old male with"
        cleaned = re.sub(
            r"^[Aa]n?\s+\d+-year-old\s+\w+\s+(?:with\s+)?(?:a\s+)?",
            "",
            first_sent,
        ).strip()
        # Cap at 60 chars to keep it focused
        if len(cleaned) > 60:
            cleaned = cleaned[:60].rsplit(" ", 1)[0]
        return cleaned if cleaned else ""

    def _auto_chain_next_tool(
        self,
        context: dict,
        task_input: TaskInput,
        messages: list[dict],
        all_evidence: list[EvidenceRecord],
        used_tools: set[str],
    ) -> None:
        """Auto-call the next priority tool when the first returned no evidence."""
        category = context["category"]
        priority = TOOL_PRIORITY.get(category, ["search_pubmed"])
        parsed_variants = context["parsed_variants"]

        for tool_name in priority:
            if tool_name in used_tools:
                continue
            tool = self.tools.get(tool_name)
            if tool is None:
                continue

            kwargs = self._build_auto_args(tool_name, parsed_variants, task_input)
            logger.info(f"Auto-chaining to next tool: {tool_name}({kwargs})")

            try:
                summary, evidence = tool.execute(**kwargs)
                all_evidence.extend(evidence)
                messages.append(format_tool_result(tool_name, summary))
                if evidence:
                    return  # Got evidence, stop chaining
            except Exception as e:
                logger.error(f"Auto-chain tool call failed: {e}")
                continue

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

        # Generate justifications via a separate LLM call
        justifications = self._generate_justifications(
            evidence_records, answer, task_input
        )

        items = []
        for i, record in enumerate(evidence_records[:5]):  # Cap at 5 evidence items
            justification = justifications[i] if i < len(justifications) else (
                f"Evidence supporting the answer '{answer}' for the question about "
                f"{', '.join(v.gene for v in task_input.patient.genotype)}."
            )
            # Truncate justification to 500 chars
            if len(justification) > 500:
                justification = justification[:497] + "..."

            items.append(EvidenceItem(
                source=record.url,
                time_accessed=record.time_accessed,
                justification=justification,
            ))

        return items

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

    def _generate_justifications(
        self,
        evidence_records: list[EvidenceRecord],
        answer: str,
        task_input: TaskInput,
    ) -> list[str]:
        """Generate justifications for each evidence source."""
        urls = [r.url for r in evidence_records[:5]]
        n = len(urls)
        numbered_urls = "\n".join(f"{i+1}. {url}" for i, url in enumerate(urls))
        gene = ", ".join(v.gene for v in task_input.patient.genotype)

        prompt = (
            f"Question: {task_input.question.prompt}\n"
            f"Answer: {answer}\n"
            f"Gene(s): {gene}\n\n"
            f"Evidence sources:\n{numbered_urls}\n\n"
            f"Write exactly {n} justifications, one per source, explaining how each "
            f"source supports the answer. Each justification must be 1-2 sentences "
            f"and under 400 characters. Do NOT include URLs in justifications.\n\n"
            f"Format: one justification per line, numbered 1 through {n}."
        )

        messages = [
            {"role": "system", "content": "You write brief medical evidence justifications. Output only numbered lines, no other text."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = generate(
                self.model,
                self.tokenizer,
                messages,
                tools=None,
                max_new_tokens=1024,
                enable_thinking=False,
            )
            # Parse numbered lines: "1. justification text"
            cleaned = []
            for line in response.text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Remove numbering prefixes like "1.", "1)", "- 1."
                line = re.sub(r"^[-•]\s*", "", line)
                line = re.sub(r"^\d+[\.\)]\s*", "", line)
                # Skip lines that are just URLs
                if line.startswith("http") and " " not in line:
                    continue
                if line:
                    cleaned.append(line)
            return cleaned
        except Exception as e:
            logger.error(f"Failed to generate justifications: {e}")
            return []
