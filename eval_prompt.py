import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
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
SERVING_ENDPOINT_NAME = "databricks-dbrx-instruct"  # Replace with your endpoint / model name
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
# ROBUST PYDANTIC SCHEMAS
# ==========================================
class AuditMatrix(BaseModel):
    insufficient_user_context: bool = Field(
        default=False, description="TRUE if request lacks info required to determine policy/scenario/entity, making clarification appropriate."
    )
    unnecessary_clarification: bool = Field(
        default=False, description="TRUE if enough info exists or a conditional answer could have been provided before asking questions."
    )
    retrieval_coverage_gap: bool = Field(
        default=False, description="TRUE if the required authoritative document, clause, legal text, or evidence simply wasn't retrieved at all."
    )
    retrieval_precision_issue: bool = Field(
        default=False, description="TRUE if info was retrieved, but evidence used is tangential, from wrong context, or less applicable."
    )
    context_continuity_failure: bool = Field(
        default=False, description="TRUE if relevant info/constraints established earlier in conversation are ignored, forgotten, or re-requested."
    )
    intent_misinterpretation: bool = Field(
        default=False, description="TRUE if assistant misunderstands what user is trying to accomplish (lookup vs summary vs explanation or wrong decision trigger)."
    )
    excessive_guardrail: bool = Field(
        default=False, description="TRUE if a legitimate request is unnecessarily blocked because safety/policy boundaries are interpreted too broadly."
    )
    incomplete_safe_alternative: bool = Field(
        default=False, description="TRUE if assistant identifies a limitation, but fails to provide permissible next steps, escalation, or repository guidance."
    )
    quality_of_communication: bool = Field(
        default=False, description="TRUE if answer is substantively correct but poorly delivered (verbose, vague, repetitive, badly structured)."
    )

    @field_validator("*", mode="before")
    def coerce_booleans(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v) if v is not None else False

class AuditEvaluationResponse(BaseModel):
    step1_standalone_intent: str = Field(
        default="", description="Step 1 analysis capturing request alongside chronological conversation history up to 5 turns."
    )
    step2_context_constraint_audit: str = Field(
        default="", description="Step 2 analysis evaluating past constraints, context amnesia, role boundaries, and conversation continuity."
    )
    step3_retrieval_grounding_audit: str = Field(
        default="", description="Step 3 holistic analysis comparing response against retrieved document chunks and identifying root causes."
    )
    primary_failure_theme: str = Field(
        default="UNMAPPED", description="The single most accurate primary root-cause theme selected by the LLM from the allowed list."
    )
    matrix: AuditMatrix = Field(default_factory=AuditMatrix)

    @field_validator("step1_standalone_intent", "step2_context_constraint_audit", "step3_retrieval_grounding_audit", mode="before")
    def coerce_to_string(cls, v):
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v) if v is not None else ""

    @field_validator("primary_failure_theme", mode="before")
    def coerce_theme(cls, v):
        theme = str(v).strip().upper()
        return theme if theme in ALLOWED_THEMES else "UNMAPPED"

# ==========================================
# SYSTEM PROMPT (LLM ROOT-CAUSE SELECTION)
# ==========================================
AUDITOR_SYSTEM_PROMPT = """You are an expert external AI Auditor evaluating enterprise RAG conversations.
Follow this explicit 3-step holistic evaluation process before assigning multi-label flags and selecting the single primary failure theme.

CRITICAL DISAMBIGUATION RULES FOR ROOT CAUSES:
1. **Over-Guardrailing vs Retrieval Dead End**: If an assistant gives a generic refusal because no useful evidence was retrieved, determine if the root cause was a **RETRIEVAL_COVERAGE_GAP** (missing evidence) rather than **EXCESSIVE_GUARDRAIL**. Do not mislabel empty retrieval as a policy block.
2. **Over-Guardrailing vs Intent Misinterpretation**: If the assistant refuses because it incorrectly assumes the user is asking for a definitive decision/action it cannot take, classify the primary root cause as **INTENT_MISINTERPRETATION**.
3. **Retrieval Coverage vs Precision**: Distinguish clearly between required evidence not being retrieved at all (**RETRIEVAL_COVERAGE_GAP**) vs. evidence being retrieved but being tangential or from the wrong segment (**RETRIEVAL_PRECISION_ISSUE**).
4. **Clarification Audit**: Do not mark clarification requests as uniformly valid. Test whether a useful conditional answer could have been provided before asking follow-up questions. If unnecessary, flag **UNNECESSARY_CLARIFICATION**.
5. **Context Continuity / Amnesia**: Pay close attention to multi-turn memory. If constraints established earlier in the conversation are ignored or re-requested, flag **CONTEXT_CONTINUITY_FAILURE**.

STEP 1: STANDALONE INTENT EXTRACTION
- Analyze the immediate user request alongside past conversation history up to 5 turns (strictly evaluated without future peeking).
- State the explicit intent and goal of the user request.

STEP 2: CONTEXT & CONSTRAINT AUDIT
- Check if previous conversation constraints or established facts were respected, or if context amnesia occurred.
- Evaluate whether clarification was genuinely required or if the assistant created unnecessary loops.

STEP 3: HOLISTIC RETRIEVAL & GROUNDING CHECK
- Evaluate the response against the provided document chunks.
- Separate primary root-cause failure from secondary compounding issues.

ALLOWED THEMES TO CHOOSE FROM FOR primary_failure_theme:
1. INSUFFICIENT_USER_CONTEXT: Request lacks info required to determine policy/scenario/entity/product/role. Clarification appropriate.
2. UNNECESSARY_CLARIFICATION: Enough info already exists or conditional answer was possible, but assistant asks additional questions or creates loops.
3. RETRIEVAL_COVERAGE_GAP: Required authoritative document, clause, appendix, legal text, or evidence simply wasn't retrieved.
4. RETRIEVAL_PRECISION_ISSUE: Information retrieved, but evidence used is tangential, from wrong segment/context, or less applicable.
5. CONTEXT_CONTINUITY_FAILURE: Relevant info already established earlier in conversation is ignored, forgotten (context amnesia), or unnecessarily re-requested.
6. INTENT_MISINTERPRETATION: Assistant misunderstands what user is trying to accomplish (navigation, summarization, lookup, explanation, or wrong decision trigger).
7. EXCESSIVE_GUARDRAIL: Legitimate request unnecessarily blocked/restricted because safety/policy boundaries are interpreted too broadly.
8. INCOMPLETE_SAFE_ALTERNATIVE: Correctly identifies limit, but fails to provide permissible assistance like criteria, repository guidance, or next steps.
9. QUALITY_OF_COMMUNICATION: Answer substantively available but poorly delivered (incomplete, overly verbose, vague, repetitive, badly structured).
10. EXPECTED_BEHAVIOR: No failure detected; fully successful and appropriate turn.
11. UNMAPPED: Anomalous behavior or failure mode that does not fall into any of the above categories, or when an issue occurs but cannot be categorized under existing themes.

OUTPUT REQUIREMENTS:
Output strictly a valid JSON object matching this schema:
{
  "step1_standalone_intent": "...",
  "step2_context_constraint_audit": "...",
  "step3_retrieval_grounding_audit": "...",
  "primary_failure_theme": "EXACT_THEME_NAME_FROM_LIST_ABOVE",
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
Do not include any code block markers or conversational preamble outside the JSON object.
"""

# ==========================================
# DATABRICKS CLIENT & ASYNC API WRAPPER
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
        
        # Retrieve history strictly accumulated UP TO this transaction (no future peek)
        full_history = history_lookup.get(trans_id, [])
        bounded_history = full_history[-MAX_HISTORY_TURNS:] if len(full_history) > MAX_HISTORY_TURNS else full_history
        
        user_prompt = f"""
### AUDIT TARGET:
- Transaction ID: {trans_id}
- Conversation ID: {conv_id}
- History Context ({len(bounded_history)} prior turns):
{json.dumps(bounded_history, indent=2)}

- Document Chunks (Retrieved):
{row['document_chunks']}

- Immediate User Request:
{row['request']}

- Generation Stage Response:
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
                "primary_failure_theme": eval_obj.primary_failure_theme,
                "step1_standalone_intent": eval_obj.step1_standalone_intent,
                "step2_context_constraint_audit": eval_obj.step2_context_constraint_audit,
                "step3_retrieval_grounding_audit": eval_obj.step3_retrieval_grounding_audit,
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
                "primary_failure_theme": "AUDIT_FAILURE",
                "step1_standalone_intent": None,
                "step2_context_constraint_audit": None,
                "step3_retrieval_grounding_audit": None,
                **{k: None for k in AuditMatrix().dict().keys()},
                "audit_status": "FAILED",
                "error_message": str(e)
            }

# ==========================================
# MAIN PIPELINE & CHECKPOINTING
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
    
    logger.info("Building secure chronological conversation history maps...")
    history_lookup: Dict[str, List[Dict[str, str]]] = {}
    
    # Iterate conversation groups and build history incrementally per transaction
    for conv_id, group in df.groupby('conversation_id', sort=False):
        conversation_history = []
        for _, r in group.iterrows():
            trans_id = str(r['transaction_id'])
            # Store a copy of history strictly occurring BEFORE this transaction turn
            history_lookup[trans_id] = list(conversation_history)
            # Append current turn to rolling history for subsequent turns in this conversation
            conversation_history.append({"user": str(r['request']), "agent": str(r['response'])})

    df['date_key'] = df['request_created_on_dt'].dt.date
    unique_dates = sorted(df['date_key'].unique())
    logger.info(f"Found {len(unique_dates)} distinct dates to process.")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    
    for current_date in unique_dates:
        checkpoint_file = CHECKPOINT_DIR / f"audit_checkpoint_{current_date}.csv"
        
        if checkpoint_file.exists():
            logger.info(f"Skipping date {current_date} — Checkpoint already exists.")
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
    else:
        logger.warning("No checkpoint files found.")

if __name__ == "__main__":
    asyncio.run(run_audit_pipeline())
