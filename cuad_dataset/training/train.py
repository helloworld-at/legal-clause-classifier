"""
QLoRA Fine-tuning: Qwen/Qwen3-4B-Instruct on CUAD Contract Clause Classification
==================================================================================
Stack:  transformers 5.x  |  peft 0.19+  |  trl 1.9+  |  bitsandbytes 0.50+

Config:
  - Quantization  : 4-bit NF4 via bitsandbytes
  - LoRA          : r=16, alpha=32, dropout=0.05, target q/k/v/o_proj
  - Data          : ../data/train.jsonl  /  ../data/val.jsonl
  - Epochs        : 3  |  Batch: 4  |  Grad accum: 4  |  LR: 2e-4
  - Scheduler     : cosine  |  Warmup ratio: 0.03
  - Max seq len   : 1024
  - Loss          : completion-only (assistant turns only, via SFTConfig)
  - Logging       : Weights & Biases (set WANDB_API_KEY env var, or run `wandb login`)
  - Adapter out   : ./output/

Usage:
  # From the repo root (works on Linux/Colab/Mac/Windows):
  python training/train.py

  # To skip W&B logging:
  WANDB_MODE=disabled python training/train.py

  # On Colab — set your W&B key before running:
  #   import os; os.environ['WANDB_API_KEY'] = 'your-key'
  #   Or use Colab Secrets (left sidebar key icon)
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — all relative to this script's directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"

# ---------------------------------------------------------------------------
# Model & training hyper-parameters
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-4B-Instruct"
WANDB_PROJECT = "cuad-qlora-qwen3-4b"

QLORA_CFG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

TRAIN_CFG = dict(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_length=1024,                      # trl 1.9+: was max_seq_length
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
)

# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a legal contract analysis assistant. "
    "You classify contract clauses into risk categories and provide brief justifications."
)


def build_assistant_response(labels: List[str]) -> str:
    """Build a deterministic, structured assistant response."""
    if len(labels) == 1:
        cat = labels[0]
        return (
            f"Category: {cat}\n"
            f"Justification: This clause falls under '{cat}' because it directly "
            f"governs the contractual obligation or risk described by that category."
        )
    cat_str = ", ".join(labels)
    return (
        f"Categories: {cat_str}\n"
        f"Justification: This clause spans multiple categories ({cat_str}) "
        f"because it simultaneously addresses several distinct contractual obligations "
        f"and risk areas within a single provision."
    )


def format_example(record: Dict, tokenizer: AutoTokenizer) -> Dict[str, str]:
    """
    Format a CUAD record as a chat-templated string.

    User   → instruction + clause text
    Assist → structured label(s) + one-sentence justification

    SFTConfig(completion_only_loss=True) will mask the loss on all tokens
    except the assistant turn — no DataCollator needed.
    """
    user_content = (
        f"{record['instruction']}\n\n"
        f"Contract Clause:\n{record['clause_text']}"
    )
    assist_content = build_assistant_response(record["labels"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assist_content},
    ]

    # tokenize=False → raw string; add_generation_prompt=False → include <|im_end|>
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Validate data files exist before downloading the model
    for label, path in [("train", TRAIN_FILE), ("val", VAL_FILE)]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} file not found: {path}\n"
                f"Run `python preprocess.py` from cuad_dataset/ first."
            )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Model         : {MODEL_ID}")
    logger.info(f"Output dir    : {OUTPUT_DIR}")
    logger.info(f"bf16 enabled  : {TRAIN_CFG['bf16']}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU           : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    logger.info("[1/5] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        padding_side="right",   # right-pad for causal LM training
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. Datasets
    # ------------------------------------------------------------------
    logger.info("[2/5] Building chat-formatted datasets")
    train_records = load_jsonl(TRAIN_FILE)
    val_records = load_jsonl(VAL_FILE)
    logger.info(f"  Train examples : {len(train_records)}")
    logger.info(f"  Val examples   : {len(val_records)}")

    train_dataset = Dataset.from_list(
        [format_example(r, tokenizer) for r in train_records]
    )
    val_dataset = Dataset.from_list(
        [format_example(r, tokenizer) for r in val_records]
    )

    # ------------------------------------------------------------------
    # 3. Compute logging / eval cadence
    # ------------------------------------------------------------------
    effective_batch = (
        TRAIN_CFG["per_device_train_batch_size"]
        * TRAIN_CFG["gradient_accumulation_steps"]
    )
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    total_steps = steps_per_epoch * TRAIN_CFG["num_train_epochs"]
    eval_steps = max(1, steps_per_epoch // 2)   # evaluate twice per epoch
    log_steps = max(1, steps_per_epoch // 10)   # log ~10× per epoch

    logger.info(f"  Effective batch size : {effective_batch}")
    logger.info(f"  Steps / epoch        : {steps_per_epoch}")
    logger.info(f"  Total steps          : {total_steps}")
    logger.info(f"  Eval every           : {eval_steps} steps")

    # ------------------------------------------------------------------
    # 4. 4-bit NF4 quantization config (passed directly to SFTTrainer)
    # ------------------------------------------------------------------
    logger.info("[3/5] Configuring 4-bit NF4 quantization & LoRA")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if TRAIN_CFG["bf16"] else torch.float16,
        bnb_4bit_use_double_quant=True,   # nested quant: saves ~0.4 bits/param
    )

    # ------------------------------------------------------------------
    # 5. LoRA config (passed directly to SFTTrainer — trl 1.9 applies
    #    get_peft_model + prepare_model_for_kbit_training internally)
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=QLORA_CFG["r"],
        lora_alpha=QLORA_CFG["lora_alpha"],
        lora_dropout=QLORA_CFG["lora_dropout"],
        target_modules=QLORA_CFG["target_modules"],
        bias="none",
        inference_mode=False,
    )

    # ------------------------------------------------------------------
    # 6. Base model (loaded in 4-bit; SFTTrainer handles kbit prep)
    # ------------------------------------------------------------------
    logger.info("[4/5] Loading base model in 4-bit NF4")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if TRAIN_CFG["bf16"] else torch.float16,
    )
    model.config.use_cache = False          # required for gradient checkpointing

    # Prepare for k-bit training: cast layer-norms to fp32, enable grad checkpointing
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # ------------------------------------------------------------------
    # 7. SFTConfig
    # ------------------------------------------------------------------
    sft_config = SFTConfig(
        # Output
        output_dir=str(OUTPUT_DIR),
        # Training duration
        num_train_epochs=TRAIN_CFG["num_train_epochs"],
        per_device_train_batch_size=TRAIN_CFG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN_CFG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAIN_CFG["gradient_accumulation_steps"],
        # Optimiser
        learning_rate=TRAIN_CFG["learning_rate"],
        lr_scheduler_type=TRAIN_CFG["lr_scheduler_type"],
        warmup_ratio=TRAIN_CFG["warmup_ratio"],
        optim="paged_adamw_8bit",           # 8-bit paged AdamW — reduces optimiser memory
        max_grad_norm=1.0,
        # Precision
        bf16=TRAIN_CFG["bf16"],
        fp16=TRAIN_CFG["fp16"],
        # Sequence length (trl 1.9+: max_length, not max_seq_length)
        max_length=TRAIN_CFG["max_length"],
        # Evaluation
        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_on_start=True,                 # baseline eval before any training steps
        # Logging
        logging_strategy="steps",
        logging_steps=log_steps,
        logging_first_step=True,
        report_to="wandb",
        run_name="qwen3-4b-cuad-qlora",
        # Checkpointing
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Data
        dataset_text_field="text",
        # Loss masking: only back-prop through assistant tokens
        completion_only_loss=True,
        # Gradient checkpointing (already enabled in prepare_model_for_kbit_training)
        gradient_checkpointing=True,
        # Misc
        dataloader_num_workers=0,           # safe default: avoids multiprocessing issues on all platforms
        remove_unused_columns=True,
        seed=42,
    )

    # ------------------------------------------------------------------
    # 8. SFTTrainer
    # ------------------------------------------------------------------
    logger.info("[5/5] Initialising SFTTrainer and starting training")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,         # trl 1.9+: `processing_class`, not `tokenizer`
        peft_config=lora_config,            # trl 1.9+: pass LoraConfig directly
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,  # stop if val_loss stagnates for 3 evals
                early_stopping_threshold=1e-3,
            )
        ],
    )

    # ------------------------------------------------------------------
    # 9. Train
    # ------------------------------------------------------------------
    logger.info("Starting training …")
    train_result = trainer.train()
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    # ------------------------------------------------------------------
    # 10. Final evaluation
    # ------------------------------------------------------------------
    logger.info("Running final evaluation …")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # ------------------------------------------------------------------
    # 11. Save LoRA adapter + tokenizer
    # ------------------------------------------------------------------
    logger.info(f"Saving LoRA adapter to: {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # ------------------------------------------------------------------
    # 12. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info(f"  Train loss  : {train_metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"  Val   loss  : {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    logger.info(f"  Adapter dir : {OUTPUT_DIR}")
    logger.info("=" * 60)

    # Finish W&B run gracefully (no-op if WANDB_MODE=disabled)
    try:
        import wandb
        if wandb.run is not None:
            wandb.summary.update({
                "final_train_loss": train_metrics.get("train_loss"),
                "final_eval_loss": eval_metrics.get("eval_loss"),
                "total_steps": total_steps,
            })
            wandb.finish()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
