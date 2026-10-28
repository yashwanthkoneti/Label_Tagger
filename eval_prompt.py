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
SERVING_ENDPOINT_NAME = "databricks-dbrx-instruct"  # Replace with your Databricks Serving Endpoint
MAX_CONCURRENT_CALLS = 10                          # Concurrency limit for parallel requests
MAX_HISTORY_TURNS = 5                              # 5-turn sliding window threshold
CHECKPOINT_DIR = Path("./audit_checkpoints")
INPUT_CSV_PATH = "rag_transcripts.csv"
OUTPUT_MASTER_CSV = "master_rag_audit_results.csv"

# Deterministic Theme Weights (Resolves overlapping indicators safely without LLM logic order errors)
THEME_WEIGHTS = {
    "MATERIAL_RISK_FAILURE": 120,      # Could cause materially wrong action (Priority 1)
    "ROLE_BREACH": 110,                # Agent broke role/persona constraints (Priority 2)
    "OVER_GUARDRAILING": 100,          # Safety/Policy false positive (Priority 3)
    "RETRIEVAL_DEAD_END": 80,          # Vector/Retrieval pipeline failure (Priority 4)
    "RETRIEVAL_PRECISION_ISSUE": 75,   # Retrieval noise or precision failure (Priority 5)
    "CLARIFICATION_FATIGUE": 60,       # Conversational UX loop bug (Priority 6)
    "VERSION_CONTROL_FAILURE": 50,     # Outdated document/version used (Priority 7)
    "INTENT_MISCLASSIFICATION": 45,    # Query interpretation error (Priority 8)
    "CONTEXT_AMNESIA": 40,             # State management/Memory loss (Priority 9)
    "CLARIFICATION_REQUESTED": 30,     # Valid conversational askback (Priority 10)
    "VERBOSITY_FATIGUE": 20,           # Formatting/Style issue (Priority 11)
    "UNMAPPED_FAILURE": 10,            # Edge-case failure (Priority 12)
}

# ==========================================
# ROBUST PYDANTIC SCHEMAS (ERROR-PROOF)
# ==========================================
class AuditMatrix(BaseModel):
    has_role_breach: bool = Field(
        default=False, description="TRUE if the chat agent broke its role or persona boundaries."
    )
    has_material_risk: bool = Field(
        default=False, description="TRUE if the response could cause a materially wrong or dangerous action."
    )
    has_retrieval_precision_issue: bool = Field(
        default=False, description="TRUE if retrieved chunks contained noise or lacked search precision."
    )
    has_intent_misclassification: bool = Field(
        default=False, description="TRUE if the query intent was misunderstood."
    )
    has_version_control_failure: bool = Field(
        default=False, description="TRUE if outdated versions or deprecated documents were used."
    )
    is_over_guardrailed: bool = Field(
        default=False, description="TRUE if user request was safe, but the agent output a compliance/refusal block."
    )
    has_retrieval_dead_end: bool = Field(
        default=False, description="TRUE if Step 3 shows required information was missing/empty in document_chunks."
    )
    has_clarification_fatigue: bool = Field(
        default=False, description="TRUE if the user already answered a clarification previously, but agent asked again."
    )
    has_context_amnesia: bool = Field(
        default=False, description="TRUE if Step 2 shows the agent ignored explicit constraints set in earlier history turns."
    )
    has_verbosity_fatigue: bool = Field(
        default=False, description="TRUE if response is accurate but contains >50% redundant corporate fluff."
    )
    is_valid_askback: bool = Field(
        default=False, description="TRUE if Step 2 shows the agent asked a helpful clarifying question."
    )
    is_unmapped_failure: bool = Field(
        default=False, description="TRUE if failure does not fit other categories."
    )

    @field_validator("*", mode="before")
    def coerce_booleans(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v) if v is not None else False

class AuditEvaluationResponse(BaseModel):
    step1_standalone_intent: str = Field(
        default="", description="Step 1 analysis capturing request alongside conversation history up to 5 turns."
    )
    step2_constraint_audit: str = Field(
        default="", description="Step 2 analysis listing explicit past constraints and role compliance."
    )
    step3_retrieval_grounding_audit: str = Field(
        default="", description="Step 3 analysis comparing response to document_chunks and sources."
    )
    completeness_score: Union[int, str] = Field(
        default=0, description="Completeness score on 0-3 scale or string 'self-abstained'."
    )
    matrix: AuditMatrix = Field(default_factory=AuditMatrix)
    relevancy_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0.0 to 1.0 relevancy score."
    )
    grounding_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0.0 to 1.0 grounding score against document_chunks."
    )

    @field_validator("step1_standalone_intent", "step2_constraint_audit", "step3_retrieval_grounding_audit", mode="before")
    def coerce_to_string(cls, v):
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v) if v is not None else ""

    @field_validator("completeness_score", mode="before")
    def coerce_completeness(cls, v):
        if str(v).lower() in ["self-abstained", "abstained", "none", "null"]:
            return "self-abstained"
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    @field_validator("relevancy_score", "grounding_score", mode="before")
    def coerce_scores(cls, v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

# ==========================================
# SYSTEM PROMPT (UPDATED WITH ALL THEMES)
# ==========================================
AUDITOR_SYSTEM_PROMPT = """You are an expert external AI Auditor evaluating an enterprise RAG conversation.
To prevent arbitrary scoring, you MUST follow this explicit 3-step evaluation process before assigning scores and matrix flags:

STEP 1: STANDALONE INTENT EXTRACTION
- Analyze the user request alongside the conversation history (up to 5 turns).
- State the explicit standalone intent of the user request, including any implied constraints, entities, or preferences.

STEP 2: CONSTRAINT & FACT CHECKING
- List any explicit constraints set by the user in previous turns.
- Did the current response respect all history constraints and remain within role boundaries? (YES/NO + explain)
- Did the agent ask a question back to clarify the prompt? (YES/NO)

STEP 3: RETRIEVAL GROUNDING CHECK
- Compare the response against the provided document_chunks and source quality.
- Identify retrieval precision issues, version control failures, or missing information.
- Identify if the response contains claims not supported by the document_chunks (hallucinations/fabrications).

OUTPUT REQUIREMENTS:
Output strictly a valid JSON object matching this schema:
{
  "step1_standalone_intent": "...",
  "step2_constraint_audit": "...",
  "step3_retrieval_grounding_audit": "...",
  "completeness_score": 3,
  "relevancy_score": 0.85,
  "grounding_score": 1.0,
  "matrix": {
    "has_role_breach": false,
    "has_material_risk": false,
    "has_retrieval_precision_issue": false,
    "has_intent_misclassification": false,
    "has_version_control_failure": false,
    "is_over_guardrailed": false,
    "has_retrieval_dead_end": false,
    "has_clarification_fatigue": false,
    "has_context_amnesia": false,
    "has_verbosity_fatigue": false,
    "is_valid_askback": false,
    "is_unmapped_failure": false
  }
}
Do not include any code block markers or conversational preamble outside the JSON object.
"""

# ==========================================
# DETERMINISTIC SINGLE CATEGORY RESOLVER
# ==========================================
def resolve_primary_theme(matrix: AuditMatrix) -> str:
    """Deterministically selects exactly ONE primary theme based on priority weights."""
    active_themes = []
    
    if matrix.has_material_risk:
        active_themes.append(("MATERIAL_RISK_FAILURE", THEME_WEIGHTS["MATERIAL_RISK_FAILURE"]))
    if matrix.has_role_breach:
        active_themes.append(("ROLE_BREACH", THEME_WEIGHTS["ROLE_BREACH"]))
    if matrix.is_over_guardrailed:
        active_themes.append(("OVER_GUARDRAILING", THEME_WEIGHTS["OVER_GUARDRAILING"]))
    if matrix.has_retrieval_dead_end:
        active_themes.append(("RETRIEVAL_DEAD_END", THEME_WEIGHTS["RETRIEVAL_DEAD_END"]))
    if matrix.has_retrieval_precision_issue:
        active_themes.append(("RETRIEVAL_PRECISION_ISSUE", THEME_WEIGHTS["RETRIEVAL_PRECISION_ISSUE"]))
    if matrix.has_clarification_fatigue:
        active_themes.append(("CLARIFICATION_FATIGUE", THEME_WEIGHTS["CLARIFICATION_FATIGUE"]))
    if matrix.has_version_control_failure:
        active_themes.append(("VERSION_CONTROL_FAILURE", THEME_WEIGHTS["VERSION_CONTROL_FAILURE"]))
    if matrix.has_intent_misclassification:
        active_themes.append(("INTENT_MISCLASSIFICATION", THEME_WEIGHTS["INTENT_MISCLASSIFICATION"]))
    if matrix.has_context_amnesia:
        active_themes.append(("CONTEXT_AMNESIA", THEME_WEIGHTS["CONTEXT_AMNESIA"]))
    if matrix.is_valid_askback:
        active_themes.append(("CLARIFICATION_REQUESTED", THEME_WEIGHTS["CLARIFICATION_REQUESTED"]))
    if matrix.has_verbosity_fatigue:
        active_themes.append(("VERBOSITY_FATIGUE", THEME_WEIGHTS["VERBOSITY_FATIGUE"]))
    if matrix.is_unmapped_failure:
        active_themes.append(("UNMAPPED_FAILURE", THEME_WEIGHTS["UNMAPPED_FAILURE"]))

    if not active_themes:
        return "NONE"

    active_themes.sort(key=lambda x: x[1], reverse=True)
    return active_themes[0][0]

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
    """Calls Databricks Model Serving endpoint asynchronously with backoff retry logic."""
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
    """Processes a single transaction safely with sliding history window and concurrency semaphore."""
    async with semaphore:
        conv_id = str(row['conversation_id'])
        trans_id = str(row['transaction_id'])
        
        full_history = history_lookup.get(conv_id, [])
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
            
            # Robust JSON repair replacing naive stripping to prevent 300+ validation failures
            repaired_str = repair_json(raw_response)
            parsed = json.loads(repaired_str)

            if "matrix" not in parsed or not isinstance(parsed["matrix"], dict):
                parsed["matrix"] = {}

            eval_obj = AuditEvaluationResponse(**parsed)
            primary_theme = resolve_primary_theme(eval_obj.matrix)
            
            return {
                "transaction_id": trans_id,
                "conversation_id": conv_id,
                "request_created_on": row['request_created_on'],
                "completeness_score": eval_obj.completeness_score,
                "relevancy_score": eval_obj.relevancy_score,
                "grounding_score": eval_obj.grounding_score,
                "primary_failure_theme": primary_theme,
                "askback_triggered": eval_obj.matrix.is_valid_askback,
                "step1_standalone_intent": eval_obj.step1_standalone_intent,
                "step2_constraint_audit": eval_obj.step2_constraint_audit,
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
                "completeness_score": None,
                "relevancy_score": None,
                "grounding_score": None,
                "primary_failure_theme": "AUDIT_FAILURE",
                "askback_triggered": None,
                "step1_standalone_intent": None,
                "step2_constraint_audit": None,
                "step3_retrieval_grounding_audit": None,
                **{k: None for k in AuditMatrix().dict().keys()},
                "audit_status": "FAILED",
                "error_message": str(e)
            }

# ==========================================
# DAILY CHECKPOINTING & MAIN PIPELINE
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
    
    logger.info("Building multi-turn conversation lookup maps...")
    history_lookup: Dict[str, List[Dict[str, str]]] = {}
    for conv_id, group in df.groupby('conversation_id'):
        history_list = []
        for _, r in group.iterrows():
            history_list.append({"user": str(r['request']), "agent": str(r['response'])})
        history_lookup[str(conv_id)] = history_list

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
