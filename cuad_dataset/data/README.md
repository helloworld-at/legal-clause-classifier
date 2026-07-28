# CUAD Preprocessed Dataset (Instruction Tuning Format)

## Overview
This directory contains the preprocessed Contract Understanding Atticus Dataset (CUAD) formatted for instruction-tuning contract clause classification tasks.

- **Source**: The Atticus Project / Hugging Face (`theatticusproject/cuad`)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Total Contract-Scoped Clause Examples**: 12391
- **Categories**: 41 Risk Categories

## Files
- `train.jsonl` (8685 examples, ~70% split)
- `val.jsonl` (1850 examples, ~15% split)
- `test.jsonl` (1856 examples, ~15% split)

## Format Specification
Each line is a JSON object with the following schema:
```json
{
  "clause_text": "<Clause Text>",
  "labels": ["Category 1", "Category 2"],
  "instruction": "Classify this contract clause into its risk category/categories and give a one-sentence justification."
}
```

## Citation
```bibtex
@article{hendrycks2021cuad,
  title={CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal={arXiv preprint arXiv:2103.06268},
  year={2021}
}
```
