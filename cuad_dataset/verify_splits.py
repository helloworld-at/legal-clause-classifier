import os
import json

def verify():
    data_dir = os.path.join(".", "data")
    files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    
    total_examples = 0
    categories = set()
    expected_instruction = "Classify this contract clause into its risk category/categories and give a one-sentence justification."
    
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        assert os.path.exists(file_path), f"Missing file: {file_path}"
        
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                record = json.loads(line.strip())
                
                # Check keys
                assert "clause_text" in record, f"Missing 'clause_text' in {filename}:{line_no}"
                assert "labels" in record, f"Missing 'labels' in {filename}:{line_no}"
                assert "instruction" in record, f"Missing 'instruction' in {filename}:{line_no}"
                
                # Check types and values
                assert isinstance(record["clause_text"], str) and len(record["clause_text"]) > 0
                assert isinstance(record["labels"], list) and len(record["labels"]) > 0
                assert record["instruction"] == expected_instruction
                
                for cat in record["labels"]:
                    categories.add(cat)
                
                count += 1
                
        print(f"Verified {filename}: {count} records, all records match schema.")
        total_examples += count

    print("=" * 60)
    print(f"Verification successful!")
    print(f"Total dataset records verified across train/val/test: {total_examples}")
    print(f"Total distinct categories verified across dataset: {len(categories)}")
    assert len(categories) == 41, f"Expected 41 categories, found {len(categories)}"
    print("=" * 60)

if __name__ == "__main__":
    verify()
