"""
Fix splits.json - Create a valid JSON file
"""
import os
import json

# Define the directory
data_dir = './data/processed'
os.makedirs(data_dir, exist_ok=True)

# Define the splits
splits = {
    "train": [
        "Bengaluru_CBD", "Whitefield", "Hyderabad", "Chennai",
        "Pune", "Mumbai", "Delhi", "Ahmedabad",
        "Peenya", "Electronic_City",
        "Mysuru", "Bhubaneswar",
        "Bannerghatta", "Ramanagara"
    ],
    "val": [
        "Hosur", "Mangalore", "Mandya"
    ],
    "test": [
        "Chennai_Port", "Hoskote", "Chikkaballapur"
    ]
}

# Save as JSON
file_path = os.path.join(data_dir, 'splits.json')
with open(file_path, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"✅ Created: {file_path}")

# Verify it's valid
with open(file_path, 'r') as f:
    loaded = json.load(f)
    print(f"✅ Valid JSON! Train: {len(loaded['train'])}, Val: {len(loaded['val'])}, Test: {len(loaded['test'])}")