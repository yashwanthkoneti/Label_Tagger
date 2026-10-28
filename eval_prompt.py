import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage
from json_repair import repair_json
from pydantic import BaseModel, Field, field_validator

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_AUDIT_PIPELINE")

# ==========================================
# 1. CONFIGURATION & CLIENT SETUP
# ==========================================
SERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME", "databricks-dbrx-instruct")
MAX_CONCURRENCY = 10  # Semaphore limit for async workers
MAX_HISTORY_TURNS = 5
CHECKPOINT_PATH = "/dbfs/tmp/rag_audit_checkpoints/audit_results_checkpoint.json"

w = WorkspaceClient()

AUDITOR_SYSTEM_PROMPT = """You are an expert Enterprise RAG System Auditor. 
Your task is to analyze a given RAG transaction using a strict 3-step Chain-of-Thought (CoT) framework, evaluate its correctness, completeness, source grounding, safety, and compliance, and return a strict JSON object.

### EVALUATION STEPS (MUST BE FOLLOWED IN ORDER):
1. step1_standalone_intent: Formulate a clear, standalone user intent by resolving references (pronouns, coreferences, implicit follow-ups) using the provided conversation history context.
2. step2_constraint_audit: Evaluate whether the generated response strictly obeys all system rules, persona boundaries, role constraints, and safety guardrails, without suffering from clarification or verbosity fatigue.
3. step3_retrieval_grounding_audit: Compare the generated response against the retrieved document chunks. Audit whether every claim made in the response is directly supported by retrieved chunks, identifying any hallucinations, precision failures, or dead ends.

### JSON OUTPUT SCHEMA:
Output MUST be a valid JSON object matching this exact structure:
{
  "step1_standalone_intent": "Detailed reasoning for Step 1...",
  "step2_constraint_audit": "Detailed reasoning for Step 2...",
  "step3_retrieval_grounding_audit": "Detailed reasoning for Step 3...",
  "completeness_score": 3,  // Integer 0-3 scale or string "self-abstained"
  "relevancy_score": 1.0,   // Float 0.0 to 1.0
  "grounding_score": 1.0,   // Float 0.0 to 1.0
  "matrix": {
    "has_role_breach": false,
    "has_material_risk": false,
    "has_retrieval_precision_issue": false,
    "has_intent_misclassification": false,
    "has_context_amnesia": false,
    "has_version_control_failure": false,
    "is_over_guardrailed": false,
    "has_retrieval_dead_end": false,
    "has_clarification_fatigue": false,
    "has_verbosity_fatigue": false,
    "is_valid_askback": false,
    "is_unmapped_failure": false
  }
}"""


# ==========================================
# 2. ROBUST PYDANTIC SCHEMAS (ERROR-PROOF)
# ==========================================
class AuditMatrix(BaseModel):
    has_role_breach: bool = Field(default=False)
    has_material_risk: bool = Field(default=False)
    has_retrieval_precision_issue: bool = Field(default=False)
    has_intent_misclassification: bool = Field(default=False)
    has_context_amnesia: bool = Field(default=False)
    has_version_control_failure: bool = Field(default=False)
    is_over_guardrailed: bool = Field(default=False)
    has_retrieval_dead_end: bool = Field(default=False)
    has_clarification_fatigue: bool = Field(default=False)
    has_verbosity_fatigue: bool = Field(default=False)
    is_valid_askback: bool = Field(default=False)
    is_unmapped_failure: bool = Field(default=False)

    @field_validator("*", mode="before")
    def coerce_booleans(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v) if v is not None else False


class AuditEvaluationResponse(BaseModel):
    step1_standalone_intent: str = Field(default="")
    step2_constraint_audit: str = Field(default="")
    step3_retrieval_grounding_audit: str = Field(default="")
    completeness_score: Union[int, str] = Field(default=0)
    relevancy_score: float = Field(default=0.0)
    grounding_score: float = Field(default=0.0)
    matrix: AuditMatrix = Field(default_factory=AuditMatrix)

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
# 3. DATABRICKS SERVING API WRAPPER
# ==========================================
async def call_databricks_serving_api(prompt_text: str) -> str:
    """Calls Databricks Serving using the official SDK WorkspaceClient asynchronously."""
    def _invoke():
        response = w.serving_endpoints.query(
            name=SERVING_ENDPOINT_NAME,
            messages=[
                ChatMessage(role="system", content=AUDITOR_SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt_text)
            ]
        )
        return response.choices[0].message.content

    return await asyncio.to_thread(_invoke)


# ==========================================
# 4. PRIORITY THEME RESOLUTION MATRIX
# ==========================================
def resolve_primary_theme(matrix: AuditMatrix) -> str:
    if matrix.has_material_risk:
        return "MATERIAL_RISK_FAILURE"
    if matrix.has_role_breach:
        return "ROLE_BREACH"
    if matrix.has_version_control_failure:
        return "VERSION_CONTROL_FAILURE"
    if matrix.has_intent_misclassification:
        return "INTENT_MISCLASSIFICATION"
    if matrix.has_retrieval_precision_issue or matrix.has_retrieval_dead_end:
        return "RETRIEVAL_PRECISION_ISSUE"
    if matrix.has_context_amnesia:
        return "CONTEXTUAL_AWARENESS_LIMITATION"
    if matrix.is_over_guardrailed:
        return "OVER_GUARDRAILED"
    if matrix.has_clarification_fatigue:
        return "CLARIFICATION_FATIGUE"
    if matrix.has_verbosity_fatigue:
        return "VERBOSITY_FATIGUE"
    if matrix.is_unmapped_failure:
        return "UNMAPPED_FAILURE"
    return "SUCCESS_OR_MINOR"


# ==========================================
# 5. ASYNCHRONOUS TRANSACTION EVALUATION WORKER
# ==========================================
async def evaluate_single_transaction(
    row: pd.Series, 
    semaphore: asyncio.Semaphore,
    history_lookup: Dict[str, List[Dict[str, str]]]
) -> Dict[str, Any]:
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
{row.get('document_chunks', 'N/A')}

- Immediate User Request:
{row.get('request', 'N/A')}

- Generation Stage Response:
{row.get('response', 'N/A')}
"""
        try:
            raw_response = await call_databricks_serving_api(user_prompt)
            
            # Repair malformed/truncated JSON artifacts safely
            repaired_str = repair_json(raw_response)
            parsed = json.loads(repaired_str)

            # Ensure matrix exists and fallback fields appropriately
            if "matrix" not in parsed or not isinstance(parsed["matrix"], dict):
                parsed["matrix"] = {}

            eval_obj = AuditEvaluationResponse(**parsed)
            primary_theme = resolve_primary_theme(eval_obj.matrix)
            
            return {
                "transaction_id": trans_id,
                "conversation_id": conv_id,
                "request_created_on": row.get('request_created_on'),
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
                "request_created_on": row.get('request_created_on'),
                "completeness_score": None,
                "relevancy_score": None,
                "grounding_score": None,
                "primary_failure_theme": "AUDIT_FAILURE",
                "askback_triggered": None,
                "step1_standalone_intent": None,
                "step2_constraint_audit": None,
                "step3_retrieval_grounding_audit": None,
                "audit_status": "FAILED",
                "error_message": str(e)
            }


# ==========================================
# 6. PIPELINE ORCHESTRATOR & CHECKPOINTING
# ==========================================
async def run_batch_audit_pipeline(df_transactions: pd.DataFrame, history_lookup: Dict[str, List[Dict[str, str]]]):
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [evaluate_single_transaction(row, semaphore, history_lookup) for _, row in df_transactions.iterrows()]
    
    logger.info(f"Starting asynchronous batch evaluation of {len(tasks)} transactions...")
    results = await asyncio.gather(*tasks)
    
    df_results = pd.DataFrame(results)
    
    # Save checkpoint results
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    df_results.to_json(CHECKPOINT_PATH, orient="records", indent=2)
    logger.info(f"Pipeline complete. Successfully checkpointed {len(df_results)} rows to {CHECKPOINT_PATH}")
    
    return df_results
