"""
Import & config sanity check for train.py (trl 1.9+ / transformers 5.x).
Does NOT download the model or run training.
Run from cuad_dataset/:  python training/check_imports.py
"""

import sys
print("Python: {}\n".format(sys.version))

ok = True

def check(label, fn):
    global ok
    try:
        result = fn()
        suffix = "  ({})".format(result) if result else ""
        print("[OK] {}{}".format(label, suffix))
    except Exception as e:
        print("[FAIL] {}: {}".format(label, e))
        ok = False

# --- Package imports --------------------------------------------------------
check("torch",          lambda: __import__("torch").__version__)
check("transformers",   lambda: __import__("transformers").__version__)
check("peft",           lambda: __import__("peft").__version__)
check("trl",            lambda: __import__("trl").__version__)
check("bitsandbytes",   lambda: __import__("bitsandbytes").__version__)
check("accelerate",     lambda: __import__("accelerate").__version__)
check("wandb",          lambda: __import__("wandb").__version__)
check("datasets",       lambda: __import__("datasets").__version__)

# --- GPU -------------------------------------------------------------------
import torch
print("\nGPU / compute:")
print("  CUDA available : {}".format(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("  Device         : {}".format(torch.cuda.get_device_name(0)))
    print("  VRAM           : {:.1f} GB".format(
        torch.cuda.get_device_properties(0).total_memory / 1e9))
    print("  BF16 support   : {}".format(torch.cuda.is_bf16_supported()))
else:
    print("  [!] No CUDA GPU detected. bitsandbytes 4-bit quantization requires a CUDA GPU.")
    print("      On Colab: Runtime -> Change runtime type -> T4 GPU")

# --- Path checks ------------------------------------------------------------
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

print("\nPath checks:")
for label, p in [
    ("data/train.jsonl", DATA_DIR / "train.jsonl"),
    ("data/val.jsonl",   DATA_DIR / "val.jsonl"),
    ("training/output",  SCRIPT_DIR / "output"),
]:
    exists = "OK" if p.exists() else "MISSING"
    print("  [{}] {:<20s}  ->  {}".format(exists, label, p))

# --- LoRA config smoke test -------------------------------------------------
print()
def check_lora():
    from peft import LoraConfig, TaskType
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    return "r={}, alpha={}, dropout={}, targets={}".format(
        cfg.r, cfg.lora_alpha, cfg.lora_dropout, cfg.target_modules)
check("LoraConfig (r=16, alpha=32)", check_lora)

# --- BitsAndBytes config smoke test -----------------------------------------
def check_bnb():
    from transformers import BitsAndBytesConfig
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return "load_in_4bit=True, quant_type=nf4, double_quant=True"
check("BitsAndBytesConfig (4-bit NF4)", check_bnb)

# --- SFTConfig API (trl 1.9+: max_length, completion_only_loss) -------------
def check_sftconfig():
    import inspect
    from trl import SFTConfig
    params = inspect.signature(SFTConfig.__init__).parameters
    assert "max_length" in params, "max_length not in SFTConfig"
    assert "completion_only_loss" in params, "completion_only_loss missing"
    assert "max_seq_length" not in params, "old max_seq_length still present"
    return "max_length OK, completion_only_loss OK"
check("SFTConfig API (trl 1.9+)", check_sftconfig)

# --- SFTTrainer API (processing_class, peft_config) -------------------------
def check_sfttrainer():
    import inspect
    from trl import SFTTrainer
    params = inspect.signature(SFTTrainer.__init__).parameters
    assert "processing_class" in params, "processing_class missing"
    assert "peft_config" in params, "peft_config missing"
    return "processing_class OK, peft_config OK"
check("SFTTrainer API (trl 1.9+)", check_sfttrainer)

# --- Summary ----------------------------------------------------------------
print()
if ok:
    print("All checks passed. Ready to run: python training/train.py")
else:
    print("Some checks failed. Fix the issues above before running training.")
    sys.exit(1)
