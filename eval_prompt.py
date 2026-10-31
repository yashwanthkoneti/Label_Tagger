AUDITOR_SYSTEM_PROMPT = """You are an enterprise AI Compliance & RAG Auditor evaluating a Bank Policy Assistant.
Your task is to analyze user interactions, verify retrieval sufficiency against reframed intents, and identify root-cause failure themes using the governing policy framework below.

===============================================================================
EVALUATION PIPELINE & STATE MACHINE
===============================================================================

-------------------------------------------------------------------------------
PHASE 1: CONTEXTUAL INTENT REFRAMING (MEMORY + REQUEST)
-------------------------------------------------------------------------------
1. Read the full conversation history alongside the immediate user request.
2. Reframe the immediate request into a single, fully explicit Standalone Intent:
   - Resolve all pronouns, implicit references, context shifts, and past constraints.
   - Combine implied context from history with current requirements into a unified operational query.
   - DO NOT over-deconstruct into micro-subintents. Keep the intent holistic.

-------------------------------------------------------------------------------
PHASE 2: INTENT-DRIVEN RETRIEVAL AUDIT
-------------------------------------------------------------------------------
Compare the Reframed Standalone Intent directly against the retrieved document chunks:
- COVERAGE CHECK: Does the retrieved text contain authoritative bank policy clauses capable of satisfying the Reframed Standalone Intent?
- PRECISION CHECK: Is the retrieved evidence targeted to the correct scope (jurisdiction, account type, entity, or policy version)?

-------------------------------------------------------------------------------
PHASE 3: SEQUENTIAL EVALUATION GATES
-------------------------------------------------------------------------------
GATE 1: EXPECTED BEHAVIOR CHECK
- Did the assistant fulfill the Reframed Standalone Intent using retrieved evidence accurately, respecting past constraints without excessive clarification or unnecessary guardrails?
- If YES -> Primary Theme: EXPECTED_BEHAVIOR. Stop further gate evaluation.

GATE 2: CONTEXT CONTINUITY & AMNESIA CHECK
- Did the assistant ignore, contradict, or re-request information already established in the conversation memory?
- If YES -> Flag CONTEXT_CONTINUITY_FAILURE.

GATE 3: RETRIEVAL FAILURE CHECK
- Sub-case A (Coverage Gap): Required policy clauses for the Reframed Standalone Intent are missing from retrieved chunks -> Flag RETRIEVAL_COVERAGE_GAP.
- Sub-case B (Precision Issue): Chunks were retrieved, but apply to the wrong segment, product, or scope -> Flag RETRIEVAL_PRECISION_ISSUE.

GATE 4: BEHAVIORAL & POLICY CHECK
- INTENT MISINTERPRETATION: Did the assistant execute the wrong operational task type (e.g., summarized a process when asked for specific eligibility criteria)?
- EXCESSIVE GUARDRAIL: Did the assistant issue an unnecessary refusal/disclaimer on a legitimate policy lookup when valid evidence was available?
- UNNECESSARY CLARIFICATION: Did the assistant ask follow-up questions when the Reframed Standalone Intent could have been answered directly or via a simple conditional statement?

GATE 5: UNMAPPED THEME DISCOVERY
- Does the failure mode fall outside all predefined matrix categories?
- If YES -> Set primary_failure_theme to "UNMAPPED", and generate a concise, standardized theme name in `proposed_new_theme_name` (e.g., "STALE_POLICY_CITATION").

===============================================================================
CANONICAL ABSTRACT EXAMPLES
===============================================================================

[EXAMPLE 1: CONTEXTUAL REFRAMING + COVERAGE GAP]
- History: User states they are auditing a Commercial Business Account.
- Immediate Request: "What is the daily wire limit?"
- Reframed Intent: "What is the daily wire limit for a Commercial Business Account?"
- Context Chunks: Documents only contain daily wire limits for Retail Personal Accounts.
- Assistant Response: "The daily wire limit is $10,000." (Uses retail limit).
- Evaluation: RETRIEVAL_COVERAGE_GAP (Reframed intent required commercial policy; retrieval provided retail policy).

[EXAMPLE 2: SUBTLE CLARIFICATION VS. UNNECESSARY CLARIFICATION]
- Reframed Intent: "What are the account opening requirements for [Entity_Type]?"
- Assistant Response: "For LLCs, submit X. For Corporations, submit Y. Which entity type are you registering?"
- Evaluation: EXPECTED_BEHAVIOR (Provided a clean conditional answer while asking a valid follow-up; DO NOT flag UNNECESSARY_CLARIFICATION).

===============================================================================
OUTPUT SCHEMA REQUIREMENTS
===============================================================================
Output strictly a valid JSON object matching this structure:
{
  "reframed_standalone_intent": "Core query synthesized from conversation memory and current request.",
  "reasoning_steps": {
    "intent_and_retrieval_analysis": "Assessment of reframed intent against retrieved chunks.",
    "gate_evaluations": "Step-by-step walkthrough of Gates 1 through 5."
  },
  "primary_failure_theme": "EXACT_THEME_NAME",
  "proposed_new_theme_name": "UPPERCASE_SNAKE_CASE_THEME_NAME_OR_NULL",
  "matrix": {
    "insufficient_user_context": false,
    "unnecessary_clarification": false,
    "retrieval_coverage_gap": false,
    "retrieval_precision_issue": false,
    "context_continuity_failure": false,
    "intent_misinterpretation": false,
    "excessive_guardrail": false,
    "incomplete_safe_alternative": false,
    "quality_of_communication": false
  }
}
Do not include any code block markers or extra conversational prose.
"""


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

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RAG_Matrix_Auditor")

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
SERVING_ENDPOINT_NAME = "databricks-dbrx-instruct"
MAX_CONCURRENT_CALLS = 10
MAX_HISTORY_TURNS = 5
CHECKPOINT_DIR = Path("./audit_checkpoints")
INPUT_CSV_PATH = "rag_transcripts.csv"
OUTPUT_MASTER_CSV = "master_rag_audit_results.csv"

ALLOWED_THEMES = {
    "INSUFFICIENT_USER_CONTEXT",
    "UNNECESSARY_CLARIFICATION",
    "RETRIEVAL_COVERAGE_GAP",
    "RETRIEVAL_PRECISION_ISSUE",
    "CONTEXT_CONTINUITY_FAILURE",
    "INTENT_MISINTERPRETATION",
    "EXCESSIVE_GUARDRAIL",
    "INCOMPLETE_SAFE_ALTERNATIVE",
    "QUALITY_OF_COMMUNICATION",
    "EXPECTED_BEHAVIOR",
    "UNMAPPED"
}

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class AuditMatrix(BaseModel):
    insufficient_user_context: bool = Field(default=False)
    unnecessary_clarification: bool = Field(default=False)
    retrieval_coverage_gap: bool = Field(default=False)
    retrieval_precision_issue: bool = Field(default=False)
    context_continuity_failure: bool = Field(default=False)
    intent_misinterpretation: bool = Field(default=False)
    excessive_guardrail: bool = Field(default=False)
    incomplete_safe_alternative: bool = Field(default=False)
    quality_of_communication: bool = Field(default=False)

    @field_validator("*", mode="before")
    def coerce_booleans(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v) if v is not None else False

class ReasoningSteps(BaseModel):
    intent_and_retrieval_analysis: str = Field(default="")
    gate_evaluations: str = Field(default="")

class AuditEvaluationResponse(BaseModel):
    reframed_standalone_intent: str = Field(
        default="", description="Reframed intent synthesized from conversation history and immediate prompt."
    )
    reasoning_steps: ReasoningSteps = Field(default_factory=ReasoningSteps)
    primary_failure_theme: str = Field(default="UNMAPPED")
    proposed_new_theme_name: Optional[str] = Field(
        default=None, description="Suggested theme name if primary_failure_theme is UNMAPPED."
    )
    matrix: AuditMatrix = Field(default_factory=AuditMatrix)

    @field_validator("primary_failure_theme", mode="before")
    def coerce_theme(cls, v):
        theme = str(v).strip().upper()
        return theme if theme in ALLOWED_THEMES else "UNMAPPED"

# ==========================================
# DATABRICKS API WRAPPER
# ==========================================
w = WorkspaceClient()

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
async def call_databricks_serving_api(prompt_text: str) -> str:
    loop = asyncio.get_running_loop()
    
    def _sync_call():
        return w.serving_endpoints.query(
            name=SERVING_ENDPOINT_NAME,
            messages=[
                {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.0,
            max_tokens=1500
        )
    
    response = await loop.run_in_executor(None, _sync_call)
    
    if hasattr(response, 'choices') and len(response.choices) > 0:
        return response.choices[0].message.content
    elif isinstance(response, dict) and 'choices' in response:
        return response['choices'][0]['message']['content']
    else:
        raise ValueError(f"Unexpected Databricks response structure: {response}")

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
        bounded_history = full_history[-MAX_HISTORY_TURNS:] if len(full_history) > MAX_HISTORY_TURNS else full_history
        
        user_prompt = f"""
### TRANSACTION AUDIT INPUT:
- Transaction ID: {trans_id}
- Conversation ID: {conv_id}
- Conversation History ({len(bounded_history)} prior turns):
{json.dumps(bounded_history, indent=2)}

- Retrieved Document Chunks:
{row['document_chunks']}

- Immediate User Request:
{row['request']}

- Generated Assistant Response:
{row['response']}
"""
        try:
            raw_response = await call_databricks_serving_api(user_prompt)
            repaired_str = repair_json(raw_response)
            parsed = json.loads(repaired_str)

            if "matrix" not in parsed or not isinstance(parsed["matrix"], dict):
                parsed["matrix"] = {}

            eval_obj = AuditEvaluationResponse(**parsed)
            
            return {
                "transaction_id": trans_id,
                "conversation_id": conv_id,
                "request_created_on": row['request_created_on'],
                "reframed_standalone_intent": eval_obj.reframed_standalone_intent,
                "primary_failure_theme": eval_obj.primary_failure_theme,
                "proposed_new_theme_name": eval_obj.proposed_new_theme_name,
                "intent_and_retrieval_analysis": eval_obj.reasoning_steps.intent_and_retrieval_analysis,
                "gate_evaluations": eval_obj.reasoning_steps.gate_evaluations,
                **eval_obj.matrix.dict(),
                "audit_status": "SUCCESS",
                "error_message": None
            }
            
        except Exception as e:
            logger.error(f"Failed processing Transaction {trans_id}: {str(e)}")
            return {
                "transaction_id": trans_id,
                "conversation_id": conv_id,
                "request_created_on": row['request_created_on'],
                "reframed_standalone_intent": None,
                "primary_failure_theme": "AUDIT_FAILURE",
                "proposed_new_theme_name": None,
                "intent_and_retrieval_analysis": None,
                "gate_evaluations": None,
                **{k: None for k in AuditMatrix().dict().keys()},
                "audit_status": "FAILED",
                "error_message": str(e)
            }

# ==========================================
# MAIN PIPELINE
# ==========================================
async def run_audit_pipeline():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading dataset from {INPUT_CSV_PATH}...")
    df = pd.read_csv(INPUT_CSV_PATH)
    
    required_cols = {'document_chunks', 'request', 'response', 'conversation_id', 'transaction_id', 'request_created_on'}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"Input CSV missing required columns: {required_cols - set(df.columns)}")
    
    df['request_created_on_dt'] = pd.to_datetime(df['request_created_on'])
    df = df.sort_values(by=['conversation_id', 'request_created_on_dt'])
    
    logger.info("Building chronological conversation history maps...")
    history_lookup: Dict[str, List[Dict[str, str]]] = {}
    
    for conv_id, group in df.groupby('conversation_id', sort=False):
        conversation_history = []
        for _, r in group.iterrows():
            trans_id = str(r['transaction_id'])
            history_lookup[trans_id] = list(conversation_history)
            conversation_history.append({"user": str(r['request']), "agent": str(r['response'])})

    df['date_key'] = df['request_created_on_dt'].dt.date
    unique_dates = sorted(df['date_key'].unique())
    logger.info(f"Found {len(unique_dates)} distinct dates to process.")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    
    for current_date in unique_dates:
        checkpoint_file = CHECKPOINT_DIR / f"audit_checkpoint_{current_date}.csv"
        
        if checkpoint_file.exists():
            logger.info(f"Skipping date {current_date} — Checkpoint exists.")
            continue
            
        logger.info(f"Processing transactions for date: {current_date}...")
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

if __name__ == "__main__":
    asyncio.run(run_audit_pipeline())
