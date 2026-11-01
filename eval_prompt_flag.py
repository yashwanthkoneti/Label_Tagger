import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from json_repair import repair_json
from tenacity import retry, stop_after_attempt, wait_random_exponential
from databricks.sdk import WorkspaceClient

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
SERVING_ENDPOINT_NAME = "databricks-dbrx-instruct"  # Replace with your endpoint / model name
MAX_CONCURRENT_CALLS = 10
MAX_HISTORY_TURNS = 5
CHECKPOINT_DIR = Path("./audit_checkpoints")
INPUT_CSV_PATH = "rag_transcripts.csv"
OUTPUT_MASTER_CSV = "master_rag_audit_results.csv"

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RAG_Atomic_Matrix_Auditor")


# ==========================================
# PHASE 1: ATOMIC SCHEMA WITH SAFE DEFAULTS
# ==========================================
class AtomicAuditFlags(BaseModel):
    step1_contextualized_request: str = Field(
        default="", 
        description="Reframed standalone query explicitly synthesizing preceding conversation history, implicit preferences, past constraints, and the immediate request."
    )
    audit_rationale: str = Field(
        default="", 
        description="A thorough rationale citing specific evidence from preceding history, chunks, and response to justify the state of each flag."
    )

    # Layer 1: Conversation Memory & Input State
    FLAG_HIST_HAS_EXPLICIT_CONSTRAINTS: bool = Field(
        default=False, description="Preceding turns contain explicit user constraints (account tier, jurisdiction, entity type, etc.)."
    )
    FLAG_HIST_CONSTRAINTS_CONTRADICTED: bool = Field(
        default=False, description="Assistant unprompted contradicts an active, unrevoked constraint established in preceding history."
    )
    FLAG_HIST_CONSTRAINTS_REREQUESTED: bool = Field(
        default=False, description="Assistant asks for a parameter/constraint already explicitly provided in preceding turns."
    )
    FLAG_HIST_CONTEXT_FULLY_PRESERVED: bool = Field(
        default=False, description="Assistant seamlessly preserves all historical parameters from preceding turns without loss."
    )

    # Layer 2: Retrieval & Grounding State
    FLAG_RET_CHUNK_COUNT_ZERO: bool = Field(
        default=False, description="Zero document chunks returned by the retriever."
    )
    FLAG_RET_CONTAINS_FULL_ANSWER: bool = Field(
        default=False, description="Retrieved chunks contain complete, explicit policy text needed to satisfy request."
    )
    FLAG_RET_CONTAINS_PARTIAL_ANSWER: bool = Field(
        default=False, description="Retrieved chunks cover only a fraction of the request, leaving key clauses or limits missing."
    )
    FLAG_RET_WRONG_ENTITY_OR_SCOPE: bool = Field(
        default=False, description="Retrieved chunks explicitly describe a different product, jurisdiction, or entity tier."
    )
    FLAG_RET_NOISY_OR_IRRELEVANT: bool = Field(
        default=False, description="Retrieved chunks are completely off-topic or noisy."
    )

    # Layer 3: Response Structural Type (Mutually Exclusive Shapes)
    FLAG_RESP_SHAPE_DIRECT_STATEMENT: bool = Field(
        default=False, description="Primary structure asserts factual or policy rules directly."
    )
    FLAG_RESP_SHAPE_CONDITIONAL_STATEMENT: bool = Field(
        default=False, description="Primary structure provides structured If/Then scenario logic."
    )
    FLAG_RESP_SHAPE_ASKBACK: bool = Field(
        default=False, description="Primary structure contains a question or clarification directed at the user (Prioritize True if a follow-up question is present)."
    )
    FLAG_RESP_SHAPE_BOUNDARY_DECLARATION: bool = Field(
        default=False, description="Primary structure explicitly states policy info is missing, unconfirmed, or outside knowledge limits."
    )
    FLAG_RESP_SHAPE_REFUSAL: bool = Field(
        default=False, description="Primary structure refuses request due to safety, compliance, or scope policy."
    )

    # Layer 4: Deep Response Execution & Grounding
    FLAG_EXEC_UNSUPPORTED_ASSERTION: bool = Field(
        default=False, description="Response contains factual/policy claim unsupported by retrieved chunks (hallucination)."
    )
    FLAG_EXEC_MISAPPLIED_LOGIC: bool = Field(
        default=False, description="Assistant read valid chunks but inverted, miscalculated, or misapplied policy rules."
    )
    FLAG_EXEC_INCOMPLETE_ANSWER: bool = Field(
        default=False, description="Assistant omitted available evidence from response without declaring a boundary."
    )
    FLAG_ASK_REQUIRED_BY_POLICY: bool = Field(
        default=False, description="Ask-back strictly required because policy prohibits answering without missing variable."
    )
    FLAG_ASK_VIABLE_WITHOUT_QUESTION: bool = Field(
        default=False, description="Query was fully resolvable conditionally without stopping to ask a question."
    )
    FLAG_ASK_ALL_PARTS_RELEVANT: bool = Field(
        default=False, description="In a multi-part ask-back, every sub-question is strictly necessary and relevant."
    )
    FLAG_ASK_HAS_REDUNDANT_PART: bool = Field(
        default=False, description="At least one sub-question in ask-back is redundant, trivial, or previously answered."
    )
    FLAG_ASK_HAS_RELEVANT_PART: bool = Field(
        default=False, description="At least one sub-question in ask-back is valid, necessary, and helpful."
    )
    FLAG_BOUND_ACCURATE_TO_RETRIEVAL: bool = Field(
        default=False, description="Accurately declared missing data when retrieval was actually empty or partial."
    )
    FLAG_BOUND_FALSE_DECLARATION: bool = Field(
        default=False, description="Claimed data was missing when full evidence was actually present in chunks."
    )
    FLAG_REFUSAL_EXCESSIVE: bool = Field(
        default=False, description="False-positive safety refusal on a legitimate bank policy query."
    )
    FLAG_ALT_OFFERED: bool = Field(
        default=False, description="Provided a safe alternative path or next step."
    )
    FLAG_ALT_INCOMPLETE_OR_BROKEN: bool = Field(
        default=False, description="Provided an incomplete, broken, or misleading alternative path."
    )
    FLAG_QUAL_COMMUNICATION_POOR: bool = Field(
        default=False, description="Poor formatting, excessive verbosity, or ambiguous phrasing."
    )

    # Layer 5: Input Query Complexity
    FLAG_QUERY_UNDER_SPECIFIED: bool = Field(
        default=False, description="User query lacks necessary variables required to yield a single specific answer."
    )
    FLAG_QUERY_MULTI_INTENT: bool = Field(
        default=False, description="User query contains multiple independent policy questions in a single turn."
    )
    FLAG_QUERY_OUT_OF_SCOPE: bool = Field(
        default=False, description="User query is non-banking or outside policy domain."
    )

    @field_validator("*", mode="before")
    def coerce_fields(cls, v):
        if isinstance(v, str):
            cleaned = v.strip().lower()
            if cleaned in ("true", "1", "yes"):
                return True
            if cleaned in ("false", "0", "no"):
                return False
        return v


# ==========================================
# DICTIONARY SANITIZER
# ==========================================
def sanitize_atomic_dict(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    
    for wrapper_key in ["flags", "matrix", "result", "properties", "data", "audit_output"]:
        if wrapper_key in data and isinstance(data[wrapper_key], dict):
            data = data[wrapper_key]
            break
            
    sanitized = {}
    for k, v in data.items():
        if isinstance(v, str):
            clean_v = v.strip().lower()
            if clean_v in ("true", "1", "yes"):
                sanitized[k] = True
            elif clean_v in ("false", "0", "no"):
                sanitized[k] = False
            else:
                sanitized[k] = v
        else:
            sanitized[k] = v
            
    return sanitized


# ==========================================
# PHASE 1 SYSTEM PROMPT (Per-Flag Explicit Rubric)
# ==========================================
ATOMIC_AUDITOR_SYSTEM_PROMPT = """You are an objective, mathematical audit sensor for an enterprise RAG assistant serving bank policy queries.
Analyze the preceding conversation history, retrieved document chunks, and assistant response. You MUST evaluate each flag individually based on the following explicit activation rules. Do not default all flags to false.

### STEP 1: INTENT SYNTHESIS & RATIONALE
- `step1_contextualized_request`: Write a clean, standalone operational query synthesizing prior history and the current request.
- `audit_rationale`: Provide a step-by-step breakdown explaining observations across memory, retrieval, response shape, and execution before assigning flags.

### STEP 2: CONVERSATION MEMORY FLAGS (FLAG_HIST_*)
- `FLAG_HIST_HAS_EXPLICIT_CONSTRAINTS`: Set True if preceding turns contain explicit user constraints (e.g., account tier, jurisdiction, entity type, specific limits).
- `FLAG_HIST_CONSTRAINTS_CONTRADICTED`: Set True if the assistant unprompted contradicts an active, unrevoked constraint established in preceding history.
- `FLAG_HIST_CONSTRAINTS_REREQUESTED`: Set True if the assistant asks for a parameter or constraint that was already explicitly provided in preceding turns.
- `FLAG_HIST_CONTEXT_FULLY_PRESERVED`: Set True if the assistant seamlessly preserves all historical parameters from preceding turns without loss.

### STEP 3: RETRIEVAL & GROUNDING FLAGS (FLAG_RET_*)
- `FLAG_RET_CHUNK_COUNT_ZERO`: Set True if zero document chunks are returned by the retriever.
- `FLAG_RET_CONTAINS_FULL_ANSWER`: Set True if retrieved chunks contain complete, explicit policy text needed to satisfy request.
- `FLAG_RET_CONTAINS_PARTIAL_ANSWER`: Set True if retrieved chunks cover only a fraction of the request, leaving key clauses or limits missing.
- `FLAG_RET_WRONG_ENTITY_OR_SCOPE`: Set True if retrieved chunks explicitly describe a different product, jurisdiction, or entity tier.
- `FLAG_RET_NOISY_OR_IRRELEVANT`: Set True if retrieved chunks are completely off-topic or noisy.

### STEP 4: RESPONSE SHAPE FLAGS (FLAG_RESP_SHAPE_*) - MUTUALLY EXCLUSIVE (SELECT EXACTLY ONE)
- `FLAG_RESP_SHAPE_DIRECT_STATEMENT`: Set True ONLY if primary structure asserts factual or policy rules directly.
- `FLAG_RESP_SHAPE_CONDITIONAL_STATEMENT`: Set True ONLY if primary structure provides structured If/Then scenario logic.
- `FLAG_RESP_SHAPE_ASKBACK`: Set True IF the assistant responds with a question, clarification request, or ends with a question mark directed at the user.
- `FLAG_RESP_SHAPE_BOUNDARY_DECLARATION`: Set True ONLY if primary structure explicitly states policy info is missing, unconfirmed, or outside knowledge limits.
- `FLAG_RESP_SHAPE_REFUSAL`: Set True ONLY if primary structure refuses request due to safety, compliance, or scope policy.

### STEP 5: EXECUTION, ASK-BACK, & BOUNDARY FLAGS
- `FLAG_EXEC_UNSUPPORTED_ASSERTION`: Set True if response contains factual/policy claims unsupported by retrieved chunks (hallucination).
- `FLAG_EXEC_MISAPPLIED_LOGIC`: Set True if assistant read valid chunks but inverted, miscalculated, or misapplied policy rules.
- `FLAG_EXEC_INCOMPLETE_ANSWER`: Set True if assistant omitted available evidence from response without declaring a boundary.
- `FLAG_ASK_REQUIRED_BY_POLICY`: Set True if policy strictly prohibits answering without the missing variable requested in the ask-back.
- `FLAG_ASK_VIABLE_WITHOUT_QUESTION`: Set True if query was fully resolvable conditionally without stopping to ask a question.
- `FLAG_ASK_ALL_PARTS_RELEVANT`: Set True if every sub-question in a multi-part ask-back is strictly necessary.
- `FLAG_ASK_HAS_REDUNDANT_PART`: Set True if at least one sub-question in the ask-back is redundant, trivial, or previously answered in history.
- `FLAG_ASK_HAS_RELEVANT_PART`: Set True if at least one sub-question in the ask-back is valid, necessary, and helpful.
- `FLAG_BOUND_ACCURATE_TO_RETRIEVAL`: Set True if missing data was accurately declared when retrieval was empty or partial.
- `FLAG_BOUND_FALSE_DECLARATION`: Set True if claimed data was missing when full evidence was actually present in chunks.
- `FLAG_REFUSAL_EXCESSIVE`: Set True for false-positive safety refusals on legitimate bank policy queries.
- `FLAG_ALT_OFFERED`: Set True if a safe alternative path or next step was provided.
- `FLAG_ALT_INCOMPLETE_OR_BROKEN`: Set True if provided alternative path was incomplete, broken, or misleading.
- `FLAG_QUAL_COMMUNICATION_POOR`: Set True for poor formatting, excessive verbosity, or ambiguous phrasing.

### STEP 6: INPUT QUERY COMPLEXITY FLAGS (FLAG_QUERY_*)
- `FLAG_QUERY_UNDER_SPECIFIED`: Set True if user query lacks necessary variables required to yield a single specific answer.
- `FLAG_QUERY_MULTI_INTENT`: Set True if user query contains multiple independent policy questions in a single turn.
- `FLAG_QUERY_OUT_OF_SCOPE`: Set True if user query is non-banking or outside policy domain.

### OUTPUT FORMAT:
You must output a single valid JSON object containing **exact key names** matching the schema fields:
```json
{
  "step1_contextualized_request": "string",
  "audit_rationale": "string",
  "FLAG_HIST_HAS_EXPLICIT_CONSTRAINTS": false,
  "FLAG_HIST_CONSTRAINTS_CONTRADICTED": false,
  "FLAG_HIST_CONSTRAINTS_REREQUESTED": false,
  "FLAG_HIST_CONTEXT_FULLY_PRESERVED": false,
  "FLAG_RET_CHUNK_COUNT_ZERO": false,
  "FLAG_RET_CONTAINS_FULL_ANSWER": false,
  "FLAG_RET_CONTAINS_PARTIAL_ANSWER": false,
  "FLAG_RET_WRONG_ENTITY_OR_SCOPE": false,
  "FLAG_RET_NOISY_OR_IRRELEVANT": false,
  "FLAG_RESP_SHAPE_DIRECT_STATEMENT": false,
  "FLAG_RESP_SHAPE_CONDITIONAL_STATEMENT": false,
  "FLAG_RESP_SHAPE_ASKBACK": false,
  "FLAG_RESP_SHAPE_BOUNDARY_DECLARATION": false,
  "FLAG_RESP_SHAPE_REFUSAL": false,
  "FLAG_EXEC_UNSUPPORTED_ASSERTION": false,
  "FLAG_EXEC_MISAPPLIED_LOGIC": false,
  "FLAG_EXEC_INCOMPLETE_ANSWER": false,
  "FLAG_ASK_REQUIRED_BY_POLICY": false,
  "FLAG_ASK_VIABLE_WITHOUT_QUESTION": false,
  "FLAG_ASK_ALL_PARTS_RELEVANT": false,
  "FLAG_ASK_HAS_REDUNDANT_PART": false,
  "FLAG_ASK_HAS_RELEVANT_PART": false,
  "FLAG_BOUND_ACCURATE_TO_RETRIEVAL": false,
  "FLAG_BOUND_FALSE_DECLARATION": false,
  "FLAG_REFUSAL_EXCESSIVE": false,
  "FLAG_ALT_OFFERED": false,
  "FLAG_ALT_INCOMPLETE_OR_BROKEN": false,
  "FLAG_QUAL_COMMUNICATION_POOR": false,
  "FLAG_QUERY_UNDER_SPECIFIED": false,
  "FLAG_QUERY_MULTI_INTENT": false,
  "FLAG_QUERY_OUT_OF_SCOPE": false
}


# ==========================================
# PHASE 2: DETERMINISTIC PRIORITY CASCADE ROUTER
# ==========================================
class AuditVerdict(BaseModel):
    primary_failure_theme: str
    is_defect: bool
    step1_contextualized_request: str
    audit_rationale: str
    audit_notes: List[str]
    atomic_flags_dict: Dict[str, Any]


def route_audit_flags(flags_dict: Dict[str, Any]) -> AuditVerdict:
    intent_req = flags_dict.get("step1_contextualized_request", "")
    audit_rationale = flags_dict.get("audit_rationale", "No rationale provided.")
    notes: List[str] = []

    shape_flags = [
        "FLAG_RESP_SHAPE_DIRECT_STATEMENT",
        "FLAG_RESP_SHAPE_CONDITIONAL_STATEMENT",
        "FLAG_RESP_SHAPE_ASKBACK",
        "FLAG_RESP_SHAPE_BOUNDARY_DECLARATION",
        "FLAG_RESP_SHAPE_REFUSAL"
    ]
    active_shapes = [shape for shape in shape_flags if flags_dict.get(shape, False)]
    
    if len(active_shapes) != 1:
        notes.append(f"Sanity Gate Failure: Expected exactly 1 response shape, found {len(active_shapes)}: {active_shapes}")
        return AuditVerdict(
            primary_failure_theme="UNMAPPED",
            is_defect=True,
            step1_contextualized_request=intent_req,
            audit_rationale=audit_rationale,
            audit_notes=notes,
            atomic_flags_dict=flags_dict
        )

    # Tier 1: Ask-Back Priority
    if flags_dict.get("FLAG_RESP_SHAPE_ASKBACK", False):
        is_unnecessary = (
            flags_dict.get("FLAG_HIST_CONSTRAINTS_REREQUESTED", False) or 
            flags_dict.get("FLAG_HIST_CONSTRAINTS_CONTRADICTED", False) or
            flags_dict.get("FLAG_ASK_HAS_REDUNDANT_PART", False) or
            (flags_dict.get("FLAG_ASK_VIABLE_WITHOUT_QUESTION", False) and not flags_dict.get("FLAG_ASK_REQUIRED_BY_POLICY", False))
        )
        
        if is_unnecessary:
            return AuditVerdict(
                primary_failure_theme="UNNECESSARY_CLARIFICATION",
                is_defect=True,
                step1_contextualized_request=intent_req,
                audit_rationale=audit_rationale,
                audit_notes=["Tier 1: Explicitly redundant or invalid Ask-Back detected."],
                atomic_flags_dict=flags_dict
            )
            
        return AuditVerdict(
            primary_failure_theme="EXPECTED_BEHAVIOR",
            is_defect=False,
            step1_contextualized_request=intent_req,
            audit_rationale=audit_rationale,
            audit_notes=["Tier 1: Valid Ask-Back / Clarification."],
            atomic_flags_dict=flags_dict
        )

    # Tier 2: Memory & Context Continuity Priority
    if flags_dict.get("FLAG_HIST_CONSTRAINTS_CONTRADICTED", False):
        return AuditVerdict(
            primary_failure_theme="CONTEXT_CONTINUITY_FAILURE",
            is_defect=True,
            step1_contextualized_request=intent_req,
            audit_rationale=audit_rationale,
            audit_notes=["Tier 2: Historical constraint contradicted."],
            atomic_flags_dict=flags_dict
        )

    # Tier 3: Retrieval & Grounding Boundary Priority
    if flags_dict.get("FLAG_RET_WRONG_ENTITY_OR_SCOPE", False):
        return AuditVerdict(
            primary_failure_theme="RETRIEVAL_PRECISION_ISSUE",
            is_defect=True,
            step1_contextualized_request=intent_req,
            audit_rationale=audit_rationale,
            audit_notes=["Tier 3: Wrong entity or scope retrieved."],
            atomic_flags_dict=flags_dict
        )

    if flags_dict.get("FLAG_RET_CHUNK_COUNT_ZERO", False):
        if flags_dict.get("FLAG_RESP_SHAPE_BOUNDARY_DECLARATION", False) and flags_dict.get("FLAG_BOUND_ACCURATE_TO_RETRIEVAL", False):
            return AuditVerdict(
                primary_failure_theme="EXPECTED_BEHAVIOR",
                is_defect=False,
                step1_contextualized_request=intent_req,
                audit_rationale=audit_rationale,
                audit_notes=["Tier 3: Empty retrieval with accurate boundary."],
                atomic_flags_dict=flags_dict
            )
        return AuditVerdict(
            primary_failure_theme="RETRIEVAL_COVERAGE_GAP",
            is_defect=True,
            step1_contextualized_request=intent_req,
            audit_rationale=audit_rationale,
            audit_notes=["Tier 3: Empty retrieval without valid boundary."],
            atomic_flags_dict=flags_dict
        )

    # Tier 4: Generation Logic & Execution Priority
    valid_retrieval_evidence = (
        flags_dict.get("FLAG_RET_CONTAINS_FULL_ANSWER", False) or 
        flags_dict.get("FLAG_RET_CONTAINS_PARTIAL_ANSWER", False)
    )

    if valid_retrieval_evidence:
        if flags_dict.get("FLAG_EXEC_MISAPPLIED_LOGIC", False) or flags_dict.get("FLAG_EXEC_UNSUPPORTED_ASSERTION", False):
            return AuditVerdict(
                primary_failure_theme="INTENT_MISINTERPRETATION",
                is_defect=True,
                step1_contextualized_request=intent_req,
                audit_rationale=audit_rationale,
                audit_notes=["Tier 4: Logic misapplied or hallucinated assertion."],
                atomic_flags_dict=flags_dict
            )

    if flags_dict.get("FLAG_RET_CONTAINS_PARTIAL_ANSWER", False):
        if not flags_dict.get("FLAG_RESP_SHAPE_BOUNDARY_DECLARATION", False) and flags_dict.get("FLAG_EXEC_INCOMPLETE_ANSWER", False):
            return AuditVerdict(
                primary_failure_theme="RETRIEVAL_COVERAGE_GAP",
                is_defect=True,
                step1_contextualized_request=intent_req,
                audit_rationale=audit_rationale,
                audit_notes=["Tier 4: Unhandled partial coverage gap."],
                atomic_flags_dict=flags_dict
            )

    # Tier 5: Safety & Guardrail Priority
    if flags_dict.get("FLAG_REFUSAL_EXCESSIVE", False):
        return AuditVerdict(
            primary_failure_theme="EXCESSIVE_GUARDRAIL",
            is_defect=True,
            step1_contextualized_request=intent_req,
            audit_rationale=audit_rationale,
            audit_notes=["Tier 5: Excessive safety refusal."],
            atomic_flags_dict=flags_dict
        )

    if flags_dict.get("FLAG_ALT_INCOMPLETE_OR_BROKEN", False):
        return AuditVerdict(
            primary_failure_theme="INCOMPLETE_SAFE_ALTERNATIVE",
            is_defect=True,
            step1_contextualized_request=intent_req,
            audit_rationale=audit_rationale,
            audit_notes=["Tier 5: Broken safe alternative path."],
            atomic_flags_dict=flags_dict
        )

    # Tier 6: Quality Defect & Catch-All
    if flags_dict.get("FLAG_QUAL_COMMUNICATION_POOR", False):
        return AuditVerdict(
            primary_failure_theme="QUALITY_OF_COMMUNICATION",
            is_defect=True,
            step1_contextualized_request=intent_req,
            audit_rationale=audit_rationale,
            audit_notes=["Tier 6: Poor communication quality."],
            atomic_flags_dict=flags_dict
        )

    return AuditVerdict(
        primary_failure_theme="EXPECTED_BEHAVIOR",
        is_defect=False,
        step1_contextualized_request=intent_req,
        audit_rationale=audit_rationale,
        audit_notes=["Clean Pass: Expected behavior."],
        atomic_flags_dict=flags_dict
    )


# ==========================================
# DATABRICKS OPENAI-COMPATIBLE API CLIENT
# ==========================================
w = WorkspaceClient()
openai_client = w.serving_endpoints.get_open_ai_client()

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
async def call_databricks_serving_api(prompt_text: str) -> str:
    loop = asyncio.get_running_loop()
    
    def _sync_call():
        response = openai_client.chat.completions.create(
            model=SERVING_ENDPOINT_NAME,
            messages=[
                {"role": "system", "content": ATOMIC_AUDITOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.0,
            max_tokens=2500,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    return await loop.run_in_executor(None, _sync_call)


# ==========================================
# ASYNC WORKER FUNCTION
# ==========================================
async def evaluate_single_transaction(
    row: pd.Series, 
    semaphore: asyncio.Semaphore,
    history_lookup: Dict[str, List[Dict[str, str]]]
) -> Dict[str, Any]:
    async with semaphore:
        conv_id = str(row['conversation_id'])
        trans_id = str(row['transaction_id'])
        
        full_history = history_lookup.get(trans_id, [])
        preceding_history = full_history[-MAX_HISTORY_TURNS:] if len(full_history) > MAX_HISTORY_TURNS else full_history
        
        doc_chunks = str(row['document_chunks']) if pd.notna(row['document_chunks']) else "[No Chunks Provided]"
        req_text = str(row['request']) if pd.notna(row['request']) else ""
        resp_text = str(row['response']) if pd.notna(row['response']) else ""
        
        user_prompt = f"""
### TRANSACTION AUDIT INPUT:
- Transaction ID: {trans_id}
- Conversation ID: {conv_id}
- Preceding Conversation History ({len(preceding_history)} prior turns):
{json.dumps(preceding_history, indent=2)}

- Retrieved Document Chunks:
{doc_chunks}

- Immediate User Request:
{req_text}

- Generated Assistant Response:
{resp_text}
"""
        try:
            raw_response = await call_databricks_serving_api(user_prompt)
            repaired_str = repair_json(raw_response)
            parsed = json.loads(repaired_str)

            cleaned_parsed = sanitize_atomic_dict(parsed)
            atomic_obj = AtomicAuditFlags(**cleaned_parsed)
            flags_dict = atomic_obj.model_dump()
            
            verdict = route_audit_flags(flags_dict)
            
            result_row = {
                "transaction_id": trans_id,
                "conversation_id": conv_id,
                "request_created_on": row.get('request_created_on'),
                "step1_contextualized_request": verdict.step1_contextualized_request,
                "primary_failure_theme": verdict.primary_failure_theme,
                "is_defect": verdict.is_defect,
                "audit_rationale": verdict.audit_rationale,
                "audit_notes": "; ".join(verdict.audit_notes),
                "audit_status": "SUCCESS",
                "error_message": None
            }
            result_row.update(verdict.atomic_flags_dict)
            return result_row
            
        except Exception as e:
            logger.error(f"Failed processing Transaction {trans_id}: {str(e)}")
            empty_flags = {k: False for k in AtomicAuditFlags.model_fields.keys() if k not in ('step1_contextualized_request', 'audit_rationale')}
            fail_row = {
                "transaction_id": trans_id,
                "conversation_id": conv_id,
                "request_created_on": row.get('request_created_on'),
                "step1_contextualized_request": None,
                "primary_failure_theme": "AUDIT_FAILURE",
                "is_defect": True,
                "audit_rationale": None,
                "audit_notes": f"Error: {str(e)}",
                "audit_status": "FAILED",
                "error_message": str(e)
            }
            fail_row.update(empty_flags)
            return fail_row


# ==========================================
# MAIN PIPELINE & CHECKPOINTING (Unsorted Input)
# ==========================================
async def run_audit_pipeline():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading dataset from {INPUT_CSV_PATH}...")
    df = pd.read_csv(INPUT_CSV_PATH)
    
    required_cols = {'document_chunks', 'request', 'response', 'conversation_id', 'transaction_id'}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"Input CSV missing required columns: {required_cols - set(df.columns)}")
    
    logger.info("Building secure chronological conversation history maps iteratively...")
    history_lookup: Dict[str, List[Dict[str, str]]] = {}
    
    for conv_id, group in df.groupby('conversation_id', sort=False):
        conversation_history = []
        for _, r in group.iterrows():
            trans_id = str(r['transaction_id'])
            history_lookup[trans_id] = list(conversation_history)
            
            req = str(r['request']) if pd.notna(r['request']) else ""
            resp = str(r['response']) if pd.notna(r['response']) else ""
            conversation_history.append({"user": req, "agent": resp})

    if 'request_created_on' in df.columns:
        df['request_created_on_dt'] = pd.to_datetime(df['request_created_on'], errors='coerce')
        df['date_key'] = df['request_created_on_dt'].dt.date.fillna(pd.to_datetime("1970-01-01").date())
    else:
        df['date_key'] = "all_records"

    unique_dates = sorted(df['date_key'].unique(), key=lambda x: str(x))
    logger.info(f"Found {len(unique_dates)} partitions to process.")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    
    for current_date in unique_dates:
        checkpoint_file = CHECKPOINT_DIR / f"audit_checkpoint_{current_date}.csv"
        
        if checkpoint_file.exists():
            logger.info(f"Skipping partition {current_date} — Checkpoint already exists.")
            continue
            
        logger.info(f"Processing transactions for partition: {current_date}...")
        daily_df = df[df['date_key'] == current_date]
        
        tasks = [
            evaluate_single_transaction(row, semaphore, history_lookup) 
            for _, row in daily_df.iterrows()
        ]
        
        daily_results = await asyncio.gather(*tasks)
        
        checkpoint_df = pd.DataFrame(daily_results)
        checkpoint_df.to_csv(checkpoint_file, index=False)
        logger.info(f"Saved checkpoint: {checkpoint_file}")

    logger.info("Combining daily checkpoints into master audit dataset...")
    all_checkpoints = list(CHECKPOINT_DIR.glob("audit_checkpoint_*.csv"))
    if all_checkpoints:
        master_df = pd.concat([pd.read_csv(f) for f in all_checkpoints], ignore_index=True)
        master_df.to_csv(OUTPUT_MASTER_CSV, index=False)
        logger.info(f"Pipeline complete! Master file saved to {OUTPUT_MASTER_CSV}")
    else:
        logger.warning("No checkpoint files found.")

if __name__ == "__main__":
    asyncio.run(run_audit_pipeline())
