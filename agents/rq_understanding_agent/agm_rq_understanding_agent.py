import os
import json
import sys

# Add the parent directory to the Python path so we can import agents module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Any
from agents.base import Agent
from utils.config import load_configurations, resolve_output_dir
from utils.llm_models import llm_model, llm_reasoning
import utils.agm_helper as helper
from utils.agm_helper import get_ui_model, get_ui_reasoning_effort
import utils.prompts as prompts
# import agents.rq_understanding_agent.rq_understanding_prompts as prompts


api_key = helper.load_OpenAI_key()


def _registry_line(s, max_value_chars=60, max_sample_chars=1200):
    """One data-overview line for the task breakdown prompt.

    Includes a few sample attribute values (geometry is already excluded by
    _peek_file), not just the file name and shape. Without them the model
    judges a dataset by its file name alone: a file named
    "pa_counties_tigerweb" holding the Pennsylvania STATE polygon
    (NAME=Pennsylvania, MTFCC=G4000) was rejected as "a county layer" and the
    state boundary was requested again.
    """
    line = (f"- {s.get('file')} ({s.get('source')}) | path: {s.get('path')} | "
            f"shape: {s.get('shape')} | columns: {s.get('columns')}")
    head = s.get("head")
    if isinstance(head, dict) and head:
        sample = {}
        for col, values in head.items():
            vals = values if isinstance(values, list) else [values]
            sample[col] = [str(v)[:max_value_chars] for v in vals]
        text = json.dumps(sample, ensure_ascii=False, default=str)
        if len(text) > max_sample_chars:
            text = text[:max_sample_chars] + "…"
        line += f" | sample values (first rows): {text}"
    return line


class RQUnderstandingAgent(Agent):
    """
    Research Question Understanding Agent

    Interprets user-defined research questions and translates them into
    operational objectives, variables, hypotheses, and study parameters.
    """
    name = "RQUnderstandingAgent"

    def __init__(self):
        super().__init__()
        # Do not cache the API key — RQUnderstandingAgent is instantiated at
        # module import time, before any Flask request. Caching here would
        # freeze the stale .env value instead of picking up g.api_key
        # per-request.
        self.openai_model = llm_model("task_breakdown.research")
        self.settings = load_configurations()

    @property
    def api_key(self):
        return helper.load_OpenAI_key()

    def rq_breakdown(self, ctx):
        """
        Generates Research question breakdown given a research question and available data

        Args:
            ctx: Context containing 'rq' (research question)

        Returns:
            Research overview
        """
        self.log(ctx, "Understanding your research question ...")
        # Extract research question
        rq = ctx.get("rq", "")

        data_registry = ctx["artifacts"].get("data_registry", {})

        # data_input_summary = [f"- {s['file']} ({s['source']}) | path: {s['path']} | shape: {s['shape']}" for s in data_registry]
        data_input_summary = [_registry_line(s) for s in data_registry]
        # data_input_summary_str = '\n'.join(data_input_summary)
        # data_input_summary


        if not rq:
            self.log(ctx, "ERROR: No research question provided")
            return {"error": "Research question is required"}

        try:
            self.log(ctx, "Generating research overview...")
            system_prompt = prompts.rq_breakdown_system_prompt
            user_prompt = f"""
            **Research Question:**
            {rq}
            
            **Data Overview: **
            {data_input_summary}
            """

            # Call OpenAI (with streaming if callback is available)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

            # user_key = helper.load_OpenAI_key()
            # print(user_key)
            try:
                response = helper.client_chat_completion_stream(
                    messages=messages,
                    # model=get_ui_model(),
                    # reasoning_effort=get_ui_reasoning_effort(),
                    model=llm_model("task_breakdown.research"),
                    response_format={"type": "json_object"}, **llm_reasoning("task_breakdown.research"),
                )
                stream_callback = ctx.get("_stream_callback")
                full_content = ""
                for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                    # content = getattr(chunk.choices[0].delta, "content", None) or ""
                    if content:
                        print(content, end="", flush=True)
                        full_content += content
                        if stream_callback:
                            stream_callback(content)
                print()  # newline after streaming

            except Exception as llm_err:
                self.log(ctx, f"ERROR in LLM response: {llm_err}")
                raise

            parsed_data = json.loads(full_content)
            self.log(ctx, f"Research question breakdown generated, parsing it to the TaskManager")

            # Save summary to file
            settings = load_configurations()
            rq_out_dir = resolve_output_dir(ctx)
            os.makedirs(rq_out_dir, exist_ok=True)
            rq_breakdown_path = os.path.join(rq_out_dir, "rq_breakdown.json")
            with open(rq_breakdown_path, "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            # Store in context
            ctx.setdefault("artifacts", {})["rq_breakdown_path"] = rq_breakdown_path
            ctx["artifacts"]["rq_breakdown"] = parsed_data

            self.log(ctx, f"Data summary saved to {rq_breakdown_path}")
            return parsed_data

        except Exception as e:
            self.log(ctx, f"ERROR in RQ parsing: {str(e)}")
            return {"error": str(e)}

    def data_intent_breakdown(self, ctx):
        """
        Lightweight intent extraction for Data Retriever mode.

        Splits the user's message into one entry per distinct downloadable
        dataset and stores the result in ctx["artifacts"]["rq_breakdown"] so
        the existing data_download pipeline can consume it unchanged.
        """
        self.log(ctx, "Understanding your data request ...")
        rq = ctx.get("rq", "")
        if not rq:
            self.log(ctx, "ERROR: No data request provided")
            return {"error": "Data request is required"}

        try:
            system_prompt = prompts.data_intent_breakdown_system_prompt
            user_prompt = f"**User message:**\n{rq}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = helper.client_chat_completion_stream(
                messages=messages,
                model=llm_model("task_breakdown.data_request"),
                response_format={"type": "json_object"}, **llm_reasoning("task_breakdown.data_request"),
            )
            stream_callback = ctx.get("_stream_callback")
            full_content = ""
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                if content:
                    full_content += content
                    if stream_callback:
                        stream_callback(content)

            parsed = json.loads(full_content)
            # Normalize: guarantee a non-empty data_requests list
            requests = parsed.get("data_requests") or []
            if not isinstance(requests, list) or not requests:
                requests = [rq]
                parsed["data_requests"] = requests

            rq_out_dir = resolve_output_dir(ctx)
            os.makedirs(rq_out_dir, exist_ok=True)
            out_path = os.path.join(rq_out_dir, "rq_breakdown.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)

            ctx.setdefault("artifacts", {})["rq_breakdown_path"] = out_path
            ctx["artifacts"]["rq_breakdown"] = parsed
            self.log(
                ctx,
                f"Data intent parsed: {len(requests)} dataset request(s) identified.",
            )
            return parsed

        except Exception as e:
            self.log(ctx, f"ERROR in data intent parsing: {str(e)}")
            # Fallback: treat the whole message as a single request so the
            # pipeline still runs.
            fallback = {"intent": rq, "data_requests": [rq]}
            ctx.setdefault("artifacts", {})["rq_breakdown"] = fallback
            return fallback

    def revise_data_intent_breakdown(self, ctx, feedback):
        """
        Lightweight revise for Data Retriever mode. Applies the user's feedback
        to the existing {intent, data_requests} structure without falling back
        to the heavy research-question schema.
        """
        self.log(ctx, "Revising data request based on feedback...")
        rq = ctx.get("rq", "")
        previous = ctx.get("artifacts", {}).get("rq_breakdown", {}) or {}

        try:
            system_prompt = prompts.data_intent_revise_system_prompt
            user_prompt = (
                f"**Original user message:**\n{rq}\n\n"
                f"**Current parsed data request (to be revised):**\n"
                f"{json.dumps(previous, indent=2)}\n\n"
                f"**User feedback:**\n{feedback}"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = helper.client_chat_completion_stream(
                messages=messages,
                model=llm_model("task_breakdown.data_request_revise"),
                response_format={"type": "json_object"}, **llm_reasoning("task_breakdown.data_request_revise"),
            )
            stream_callback = ctx.get("_stream_callback")
            full_content = ""
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                if content:
                    full_content += content
                    if stream_callback:
                        stream_callback(content)

            parsed = json.loads(full_content)
            requests = parsed.get("data_requests") or []
            if not isinstance(requests, list) or not requests:
                requests = previous.get("data_requests") or [rq]
                parsed["data_requests"] = requests
            if not parsed.get("intent"):
                parsed["intent"] = previous.get("intent") or rq

            rq_out_dir = resolve_output_dir(ctx)
            os.makedirs(rq_out_dir, exist_ok=True)
            out_path = os.path.join(rq_out_dir, "rq_breakdown.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)

            ctx.setdefault("artifacts", {})["rq_breakdown_path"] = out_path
            ctx["artifacts"]["rq_breakdown"] = parsed
            self.log(
                ctx,
                f"Data request revised: {len(requests)} dataset request(s).",
            )
            return parsed

        except Exception as e:
            self.log(ctx, f"ERROR in data intent revision: {str(e)}")
            return previous

    def revise_rq_breakdown(self, ctx, feedback):
        """
        Revise the existing rq_breakdown based on the user's free-text feedback.
        Only changes the fields the feedback asks about — leaves everything else
        untouched. Streams tokens through ctx['_stream_callback'] just like
        rq_breakdown() so the frontend can render progress in the same card.

        Args:
            ctx: Context containing 'rq' and artifacts['rq_breakdown']
            feedback: User's free-text feedback string describing what to change

        Returns:
            The revised rq_breakdown dict (also written to artifacts['rq_breakdown']
            and persisted to rq_breakdown.json).
        """
        self.log(ctx, "Revising research question breakdown based on feedback...")

        rq = ctx.get("rq", "")
        previous_breakdown = ctx.get("artifacts", {}).get("rq_breakdown", {})

        if not previous_breakdown:
            self.log(ctx, "ERROR: No previous rq_breakdown to revise")
            return {"error": "No existing rq_breakdown to revise"}

        data_registry = ctx.get("artifacts", {}).get("data_registry", {})
        data_input_summary = [_registry_line(s) for s in data_registry]

        try:
            # Reuse the same base system prompt used for initial breakdown so the
            # output JSON schema stays consistent, but add a surgical-revision
            # instruction that mirrors the research-plan revise approach.
            system_prompt = prompts.rq_breakdown_system_prompt + (
                "\n\nREVISION MODE: You are revising an EXISTING research question "
                "breakdown based on the user's feedback. Return the COMPLETE revised "
                "JSON object with the same schema. Only modify the fields the user's "
                "feedback asks about — leave everything else exactly as it was."
            )
            user_prompt = f"""
            **Research Question:**
            {rq}

            **Data Overview:**
            {data_input_summary}

            **CURRENT BREAKDOWN (to be revised):**
            {json.dumps(previous_breakdown, indent=2)}

            **USER FEEDBACK:**
            {feedback}
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                response = helper.client_chat_completion_stream(
                    messages=messages,
                    model=llm_model("task_breakdown.research_revise"),
                    response_format={"type": "json_object"}, **llm_reasoning("task_breakdown.research_revise"),
                )
                stream_callback = ctx.get("_stream_callback")
                full_content = ""
                for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                    if content:
                        print(content, end="", flush=True)
                        full_content += content
                        if stream_callback:
                            stream_callback(content)
                print()

            except Exception as llm_err:
                self.log(ctx, f"ERROR in LLM revision: {llm_err}")
                raise

            revised = json.loads(full_content)
            self.log(ctx, "Research question breakdown revised successfully")

            # Persist the revised breakdown to the same artifacts key + file,
            # so subsequent workflow steps see the updated version.
            settings = load_configurations()
            rq_out_dir = resolve_output_dir(ctx)
            os.makedirs(rq_out_dir, exist_ok=True)
            rq_breakdown_path = os.path.join(rq_out_dir, "rq_breakdown.json")
            with open(rq_breakdown_path, "w", encoding="utf-8") as f:
                json.dump(revised, f, indent=2, ensure_ascii=False)
            ctx.setdefault("artifacts", {})["rq_breakdown_path"] = rq_breakdown_path
            ctx["artifacts"]["rq_breakdown"] = revised

            return revised

        except Exception as e:
            self.log(ctx, f"ERROR in RQ revision: {str(e)}")
            return {"error": str(e)}


#     def rq_breakdown(self, ctx):
#         """
#         Generates Research question breakdown given a research question and available data
#
#         Args:
#             ctx: Context containing 'rq' (research question) and optionally data registry
#
#         Returns:
#             Research overview grounded in available data
#         """
#         self.log(ctx, "Understanding your research question ...")
#         # Extract research question
#         rq = ctx.get("rq", "")
#         if not rq:
#             self.log(ctx, "ERROR: No research question provided")
#             return {"error": "Research question is required"}
#
#         # Get data context if available
#         data_summary = ctx.get("artifacts", {}).get("data_summary_data", {})
#         data_audit = ctx.get("artifacts", {}).get("data_audit_records", [])
#
#         try:
#             self.log(ctx, "Generating research overview...")
#
#             # Build data context section
#             data_context = ""
#             if data_summary:
#                 self.log(ctx, "Using data summary to ground research overview")
#                 data_context = f"""
# **Available Data Summary:**
# {json.dumps(data_summary, indent=2)}
# """
#             elif data_audit:
#                 self.log(ctx, "Using data audit records to ground research overview")
#                 # Extract key info from audit
#                 available_fields = []
#                 for dataset in data_audit:
#                     if dataset.get("ok") and "columns" in dataset:
#                         available_fields.extend(dataset["columns"])
#                 available_fields = list(set(available_fields))
#
#                 data_context = f"""
# **Available Data:**
# - Datasets: {len(data_audit)}
# - Available fields: {', '.join(available_fields[:20])}{'...' if len(available_fields) > 20 else ''}
# """
#
#             SYSTEM_PROMPT = prompts.rq_breakdown_system_prompt
#             USER_PROMPT = f"""**Research Question:**
# {rq}
# {data_context}
# """
#             # Call OpenAI (with streaming if callback is available)
#             messages = [
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": USER_PROMPT}]
#             user_key = helper.load_OpenAI_key()
#             stream_callback = ctx.get("_stream_callback")
#
#             if stream_callback:
#                 # Stream token-by-token
#                 response = helper.gibd_chat_completion_stream(
#                     user_key, "gpt-4o", messages,
#                     temperature=0.2,
#                     response_format={"type": "json_object"},
#                 )
#                 content = ""
#                 for chunk in response:
#                     delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
#                     if delta:
#                         content += delta
#                         stream_callback(delta)
#             else:
#                 response = helper.gibd_chat_completion(
#                     user_key, "gpt-4o", messages,
#                     temperature=0.2,
#                     response_format={"type": "json_object"},
#                 )
#                 content = response.choices[0].message.content
#
#             parsed_data = json.loads(content)
#
#             self.log(ctx, f"Research question breakdown generated, parsing it to the TaskManager")
#
#             #Save summary to file
#             settings = load_configurations()
#             os.makedirs(settings.output_dir, exist_ok=True)
#             rq_breakdown_path =os.path.join(settings.output_dir,"rq_breakdown.json")
#             with open(rq_breakdown_path, "w", encoding="utf-8") as f:
#                 json.dump(parsed_data, f, indent=2, ensure_ascii=False)
#             # Store in context
#             ctx.setdefault("artifacts", {})["rq_breakdown"] = rq_breakdown_path
#             ctx["artifacts"]["rq_breakdown_data"] = parsed_data
#
#             self.log(ctx, f"Data summary saved to {rq_breakdown_path}")
#             return parsed_data
#
#         except Exception as e:
#             self.log(ctx, f"ERROR in RQ parsing: {str(e)}")
#             return {"error": str(e)}
        
        
        
    def rq_parsing (self, ctx):
        """
        Parse and structure the research question.
        

        Args:
            ctx: Context containing 'rq' (research question) and optionally 'data_overview'
            task: Task specification

        Returns:
            Dictionary with parsed RQ components
        """
        self.log(ctx, "Starting research question understanding...")

        # Extract research question
        rq = ctx.get("rq", "")

        if not rq:
            self.log(ctx, "ERROR: No research question provided")
            return {"error": "Research question is required"}

        # Get data overview if available
        data_overview = ctx.get("artifacts", {}).get("data_audit_records", [])

        if data_overview:
            self.log(ctx, f"Using data overview from {len(data_overview)} dataset(s)")
        else:
            self.log(ctx, "No data overview available - parsing RQ without data context")

        self.log(ctx, f"Parsing research question: {rq[:100]}...")

        # Parse the research question using LLM
        parsed_rq = self._parse_research_question(ctx, rq, data_overview)

        if "error" in parsed_rq:
            self.log(ctx, f"ERROR: Failed to parse research question - {parsed_rq['error']}")
            return parsed_rq

        # Save parsed results
        output_path = self._save_parsed_rq(ctx, rq, parsed_rq)

        self.log(ctx, f"Research question parsed successfully → {output_path}")

        # Store in context artifacts
        ctx.setdefault("artifacts", {})["rq_parsed"] = output_path
        ctx["artifacts"]["rq_parsed_data"] = parsed_rq

        return {
            "rq_parsed_path": output_path,
            "parsed_data": parsed_rq,
            "success": True
        }


    def handle(self, ctx: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        print("Replace with rq_parsing method")


    def _parse_research_question(self, ctx: Dict[str, Any], rq: str, data_overview: list) -> Dict[str, Any]:
        """
        Use LLM to parse and structure the research question.

        Args:
            ctx: Context dictionary
            rq: Research question text
            data_overview: Data audit records from DataAgent

        Returns:
            Dictionary with structured RQ components
        """
        try:
            self.log(ctx, "Using LLM to parse research question...")

            # Extract available fields from data overview
            available_fields = []
            if data_overview:
                for dataset in data_overview:
                    if dataset.get("ok") and "head" in dataset:
                        fields = list(dataset["head"].keys())
                        available_fields.extend(fields)
                available_fields = list(set(available_fields))  # Remove duplicates

            SYSTEM_PROMPT = prompts.rq_parser_system_prompt

            

            USER_PROMPT = f"""**Research Question:**
{rq}

**Available Data Fields:**
{json.dumps(available_fields, indent=2) if available_fields else "No data fields provided"}

Parse this research question into structured components following the schema provided.
When suggesting variables, use the actual field names from the available data when possible.
Return ONLY valid JSON
"""
            


            # Call OpenAI
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}]
            
            response = helper.client_chat_completion(
                messages = messages,
                model=llm_model("task_breakdown.parse_question"),
                
                response_format={"type": "json_object"}, **llm_reasoning("task_breakdown.parse_question"),
            )

            # Parse response
            content = response.choices[0].message.content
            parsed_data = json.loads(content)

            self.log(ctx, f"Research question parsed: {len(parsed_data.get('objectives', []))} objectives identified")

            return parsed_data

        except Exception as e:
            self.log(ctx, f"ERROR in RQ parsing: {str(e)}")
            return {"error": str(e)}

    def _save_parsed_rq(self, ctx: Dict[str, Any], original_rq: str, parsed_data: Dict[str, Any]) -> str:
        """Save parsed research question to file."""
        rq_out_dir = resolve_output_dir(ctx)
        os.makedirs(rq_out_dir, exist_ok=True)
        output_path = os.path.join(rq_out_dir, "rq_parsed.json")

        # Create structured output
        output_data = {
            "original_research_question": original_rq,
            "parsed_components": parsed_data,
            "model_used": llm_model("task_breakdown.parse_question")
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.log(ctx, f"Parsed RQ saved to {output_path}")
        return output_path