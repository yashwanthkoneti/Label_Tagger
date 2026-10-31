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
# ROBUST PYDANTIC SCHEMAS (FLATTENED)
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

class AuditEvaluationResponse(BaseModel):
    step1_contextualized_request: str = Field(
        default="", description="Reframed standalone query synthesizing conversation memory and current request."
    )
    intent_and_retrieval_analysis: str = Field(default="")
    gate_evaluations: str = Field(default="")
    primary_failure_theme: str = Field(default="EXPECTED_BEHAVIOR")
    proposed_new_theme_name: Optional[str] = Field(default=None)
    matrix: AuditMatrix = Field(default_factory=AuditMatrix)

    @field_validator("step1_contextualized_request", "intent_and_retrieval_analysis", "gate_evaluations", mode="before")
    def coerce_to_string(cls, v):
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v) if v is not None else ""

    @field_validator("primary_failure_theme", mode="before")
    def coerce_theme(cls, v):
        theme = str(v).strip().upper()
        return theme if theme in ALLOWED_THEMES else "EXPECTED_BEHAVIOR"

# ==========================================
# SYSTEM PROMPT (2-STEP POLICY FRAMEWORK)
# ==========================================
AAUDITOR_SYSTEM_PROMPT = """You are an enterprise AI Compliance & RAG Auditor evaluating a Bank Policy Assistant.
Your sole role is to act as an automated QA Tester evaluating whether the system under test (the RAG pipeline) correctly processed inputs and delivered an appropriate response.

===============================================================================
ROLE DEFINITION: AUTOMATED QA TESTER (NOT A RESPONSE GENERATOR)
===============================================================================
- You are acting purely as a QA Software Tester auditing a system transaction.
- DO NOT evaluate whether you personally can answer the user's request.
- Evaluate strictly whether the ASSISTANT RESPONSE appropriately and safely reflects the state of the RETRIEVED CHUNKS and CONVERSATION HISTORY.

KEY QA RULE ON MISSING EVIDENCE & BOUNDARY HANDLING:
- If retrieved document chunks DO NOT contain the required bank policy evidence, and the Assistant correctly informs the user that the information is unavailable, restricted, or unconfirmed, this is EXPECTED_BEHAVIOR (Successful boundary handling).
- Do NOT penalize the Assistant for a repository/retrieval limitation if its output accurately and safely acknowledges that limitation.

===============================================================================
CORE AUDIT PRINCIPLE: CONSERVATIVE FAILURE THRESHOLD & PRAGMATIC BIAS
===============================================================================
- PRAGMATIC BIAS TOWARDS SUCCESS: Favor EXPECTED_BEHAVIOR whenever an interaction is functionally sound, safe, and helpful.
- Do not penalize standard assistant behavior, cautious bank policy checks, reasonable hedging, or minor stylistic variations.
- Maintain a high bar for flagging failures: only assign a failure theme if the issue represents a clear, material deviation from operational accuracy or safety standards.

===============================================================================
2-STEP EVALUATION EXECUTION FLOW
===============================================================================

-------------------------------------------------------------------------------
STEP 1: CONTEXT SYNTHESIS & INTENT REFRAMING (HISTORY + CURRENT REQUEST)
-------------------------------------------------------------------------------
1. Read the full conversation history alongside the immediate user request (chronologically up to the current turn; no future peeking).
2. Synthesize all cumulative user context, implicit preferences, past constraints, entity types, and account parameters established earlier in the conversation.
3. Reframe the immediate request into a single, fully explicit, standalone operational query.
4. Output this synthesized query as the explicit `step1_contextualized_request`.

-------------------------------------------------------------------------------
STEP 2: POLICY GATES & AUDIT EXECUTION (EVALUATING STEP 1 OUTPUT)
-------------------------------------------------------------------------------
Using the `step1_contextualized_request` produced in Step 1, perform the following evaluations:

A. INTENT-DRIVEN RETRIEVAL & BOUNDARY AUDIT
   - BOUNDARY HANDLING: Did the retrieved text lack relevant policy evidence, and did the assistant accurately state it could not answer? If YES -> GATE 1 EXPECTED_BEHAVIOR.
   - COVERAGE CHECK: If evidence was expected to exist in the repository but is completely missing from retrieved chunks -> Flag RETRIEVAL_COVERAGE_GAP.
   - PRECISION CHECK: Were chunks retrieved, but targeted to the wrong scope, jurisdiction, account tier, or product line? -> Flag RETRIEVAL_PRECISION_ISSUE.

B. SEQUENTIAL EVALUATION GATES
   - GATE 1 (EXPECTED BEHAVIOR DEFAULT): If the assistant's response is helpful, safe, and reasonably addresses the Step 1 request (or correctly acknowledges missing data), default immediately to EXPECTED_BEHAVIOR.
   - GATE 2 (CONTEXT CONTINUITY & AMNESIA): Did the assistant flagrantly ignore, contradict, or re-request explicit facts established in prior conversation history? If YES -> Flag CONTEXT_CONTINUITY_FAILURE.
   - GATE 3 (RETRIEVAL FAILURE):
     * Coverage Gap: Required policy clauses for the Step 1 request are missing from retrieved context -> Flag RETRIEVAL_COVERAGE_GAP.
     * Precision Issue: Retrieved evidence applies to an entirely wrong segment, product, or scope -> Flag RETRIEVAL_PRECISION_ISSUE.
   - GATE 4 (CLARIFICATION LENIENCY):
     * Lenient Rule: If AT LEAST ONE question or aspect of the follow-up is relevant or prudent for pinning down banking policy details (account tier, entity, jurisdiction), classify it as valid clarification / EXPECTED_BEHAVIOR.
     * Strict Threshold for Unnecessary: Only flag UNNECESSARY_CLARIFICATION if the assistant asks redundant or baseless questions when the answer was already fully supplied in the prompt or immediate history.
   - GATE 5 (UNMAPPED THEME DISCOVERY): If a clear, severe failure mode occurs that falls outside standard predefined categories, set primary_failure_theme to "UNMAPPED" and provide a standardized name in `proposed_new_theme_name`.

===============================================================================
CANONICAL ABSTRACT EXAMPLES (EXEMPLARS ONLY)
===============================================================================
[Note on Exemplars: The examples below are synthetic, non-binding illustrations of reasoning structure and schema formatting. Analyze the target transaction on its own merits using the policy gates rather than memorizing example patterns.]

[EXAMPLE 1: ACCURATE KNOWLEDGE BOUNDARY HANDLING]
- History: User asks about [Parameter_X] for a Commercial Account.
- Immediate Request: "What is the fee for [Parameter_X]?"
- Step 1 Output: "What is the fee for [Parameter_X] on a Commercial Account?"
- Retrieved Chunks: [Empty / No relevant policy chunks retrieved]
- Assistant Response: "I am unable to find specific fee details for [Parameter_X] in the available policy repository."
- Evaluation: EXPECTED_BEHAVIOR (The assistant correctly identified the absence of evidence and refrained from hallucinating).

[EXAMPLE 2: STEP 1 REFRAMING + LENIENT CLARIFICATION]
- History: User states they are inquiring about a Corporate Tier Account.
- Immediate Request: "What are the wire transfer limits?"
- Step 1 Output: "What are the wire transfer limits for a Corporate Tier Account?"
- Assistant Response: "Daily wire limits depend on whether transfers are domestic or international. Are you sending domestic or international wires?"
- Evaluation: EXPECTED_BEHAVIOR (Step 1 intent reframed context correctly. Follow-up question on wire destination is relevant and necessary for bank policy lookup).

[EXAMPLE 3: CONTEXT CONTINUITY FAILURE]
- History: Turn 1 User states: "I am working with an LLC account in Texas." Assistant acknowledges.
- Immediate Request: "What documents are needed to add a signatory?"
- Step 1 Output: "What documents are needed to add a signatory for an LLC account in Texas?"
- Assistant Response: "Could you please clarify what type of legal entity (e.g., Sole Proprietorship, LLC, Corporation) this account is for?"
- Evaluation: CONTEXT_CONTINUITY_FAILURE (Assistant suffered context amnesia and re-requested entity type established in Turn 1).

===============================================================================
OUTPUT SCHEMA REQUIREMENTS (FLATTENED STRUCTURE)
===============================================================================
Output strictly a valid JSON object matching this structure:
{
  "step1_contextualized_request": "Reframed standalone query synthesizing conversation memory and current request.",
  "intent_and_retrieval_analysis": "Assessment comparing the Step 1 request against retrieved document chunks and knowledge boundaries.",
  "gate_evaluations": "Step-by-step walkthrough of Gates 1 through 5 evaluated against the Step 1 request.",
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
Do not include any code block markers or extra conversational prose outside the JSON object.
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
                "step1_contextualized_request": eval_obj.step1_contextualized_request,
                "primary_failure_theme": eval_obj.primary_failure_theme,
                "proposed_new_theme_name": eval_obj.proposed_new_theme_name,
                "intent_and_retrieval_analysis": eval_obj.intent_and_retrieval_analysis,
                "gate_evaluations": eval_obj.gate_evaluations,
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
                "step1_contextualized_request": None,
                "primary_failure_theme": "AUDIT_FAILURE",
                "proposed_new_theme_name": None,
                "intent_and_retrieval_analysis": None,
                "gate_evaluations": None,
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
