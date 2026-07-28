import os
import json
import numpy as np
from collections import defaultdict, Counter
from huggingface_hub import hf_hub_download
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def main():
    # 1. Output directory using relative path
    output_dir = os.path.join(".", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading/fetching CUAD_v1.json from Hugging Face ('theatticusproject/cuad')...")
    file_path = hf_hub_download(
        repo_id='theatticusproject/cuad',
        filename='CUAD_v1/CUAD_v1.json',
        repo_type='dataset'
    )
    print(f"Loaded source file from cache: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        cuad = json.load(f)

    # 2. Extract clause texts scoped per contract (contract_id, clause_text)
    # Mapping: contract_id -> clause_text -> set of categories
    contract_clause_map = defaultdict(lambda: defaultdict(set))
    
    for idx, contract in enumerate(cuad['data']):
        contract_id = contract.get('title', f'contract_{idx}')
        for p in contract['paragraphs']:
            for qa in p['qas']:
                cat = qa['id'].split('__')[-1]
                if not qa.get('is_impossible', False) and qa.get('answers'):
                    for ans in qa['answers']:
                        txt = ans['text'].strip()
                        if txt:
                            contract_clause_map[contract_id][txt].add(cat)

    processed_examples = []
    instruction_str = "Classify this contract clause into its risk category/categories and give a one-sentence justification."
    
    for contract_id, clauses in contract_clause_map.items():
        for clause_text, cats in clauses.items():
            processed_examples.append({
                "clause_text": clause_text,
                "labels": sorted(list(cats)),
                "instruction": instruction_str
            })

    total_examples = len(processed_examples)
    all_categories = sorted(list(set(c for ex in processed_examples for c in ex['labels'])))
    cat2idx = {c: i for i, c in enumerate(all_categories)}
    num_cats = len(all_categories)

    print(f"Total processed contract-scoped clause examples: {total_examples}")
    print(f"Total categories identified ({num_cats}): {all_categories}\n")

    # 3. Create feature array X and multi-label indicator Y
    X = np.arange(total_examples)
    Y = np.zeros((total_examples, num_cats), dtype=int)
    for i, ex in enumerate(processed_examples):
        for c in ex['labels']:
            Y[i, cat2idx[c]] = 1

    # 4. Multi-Label Stratified Split (70 / 15 / 15)
    # Split 1: 70% Train, 30% Temp (Val + Test)
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(msss1.split(X, Y))

    X_temp, Y_temp = X[temp_idx], Y[temp_idx]
    # Split 2: 50% of Temp -> 15% Val, 15% Test
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_sub_idx, test_sub_idx = next(msss2.split(X_temp, Y_temp))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    # 5. Rare Category Safety Guarantee (Ensure min 1 in Val & Test if category has >= 2 samples)
    for col, cat_name in enumerate(all_categories):
        tot = Y[:, col].sum()
        n_val = Y[list(val_set), col].sum()
        n_test = Y[list(test_set), col].sum()
        
        if n_val == 0 and tot >= 2:
            train_with_cat = [i for i in train_set if Y[i, col] == 1]
            if train_with_cat:
                swap_idx = train_with_cat[0]
                train_set.remove(swap_idx)
                val_set.add(swap_idx)
                
        if n_test == 0 and tot >= 2:
            train_with_cat = [i for i in train_set if Y[i, col] == 1]
            if train_with_cat:
                swap_idx = train_with_cat[0]
                train_set.remove(swap_idx)
                test_set.add(swap_idx)

    train_idx = np.array(sorted(list(train_set)))
    val_idx = np.array(sorted(list(val_set)))
    test_idx = np.array(sorted(list(test_set)))

    train_examples = [processed_examples[i] for i in train_idx]
    val_examples = [processed_examples[i] for i in val_idx]
    test_examples = [processed_examples[i] for i in test_idx]

    # 6. Save JSONL files
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for ex in test_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("=" * 80)
    print(f"Data files successfully created in '{output_dir}':")
    print(f"  - train.jsonl: {len(train_examples):6d} samples ({len(train_examples)/total_examples:.2%})")
    print(f"  - val.jsonl:   {len(val_examples):6d} samples ({len(val_examples)/total_examples:.2%})")
    print(f"  - test.jsonl:  {len(test_examples):6d} samples ({len(test_examples)/total_examples:.2%})")
    print("=" * 80)

    # 7. Print Class Distribution Stats
    train_counts = Y[train_idx].sum(axis=0)
    val_counts = Y[val_idx].sum(axis=0)
    test_counts = Y[test_idx].sum(axis=0)
    total_counts = Y.sum(axis=0)

    print("\nCLASS DISTRIBUTION STATISTICS AFTER 70/15/15 STRATIFIED SPLIT:")
    header = f"{'Category Name':<38} | {'Total':<6} | {'Train Count (%)':<16} | {'Val Count (%)':<16} | {'Test Count (%)':<16}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i, cat_name in enumerate(all_categories):
        tot = total_counts[i]
        tr = train_counts[i]
        va = val_counts[i]
        te = test_counts[i]
        
        tr_pct = tr / len(train_idx) * 100
        va_pct = va / len(val_idx) * 100
        te_pct = te / len(test_idx) * 100
        
        print(f"{cat_name:<38} | {tot:<6d} | {tr:4d} ({tr_pct:5.2f}%)   | {va:4d} ({va_pct:5.2f}%)   | {te:4d} ({te_pct:5.2f}%)")
    
    print("-" * len(header))

    # 8. Create README.md in ./data/
    readme_path = os.path.join(output_dir, "README.md")
    readme_content = f"""# CUAD Preprocessed Dataset (Instruction Tuning Format)

## Overview
This directory contains the preprocessed Contract Understanding Atticus Dataset (CUAD) formatted for instruction-tuning contract clause classification tasks.

- **Source**: The Atticus Project / Hugging Face (`theatticusproject/cuad`)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Total Contract-Scoped Clause Examples**: {total_examples}
- **Categories**: 41 Risk Categories

## Files
- `train.jsonl` ({len(train_examples)} examples, ~70% split)
- `val.jsonl` ({len(val_examples)} examples, ~15% split)
- `test.jsonl` ({len(test_examples)} examples, ~15% split)

## Format Specification
Each line is a JSON object with the following schema:
```json
{{
  "clause_text": "<Clause Text>",
  "labels": ["Category 1", "Category 2"],
  "instruction": "Classify this contract clause into its risk category/categories and give a one-sentence justification."
}}
```

## Citation
```bibtex
@article{{hendrycks2021cuad,
  title={{CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review}},
  author={{Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer}},
  journal={{arXiv preprint arXiv:2103.06268}},
  year={{2021}}
}}
```
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"\nDataset metadata and citation saved to '{readme_path}'.")

if __name__ == "__main__":
    main()
