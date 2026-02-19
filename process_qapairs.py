import random
import os
import pandas as pd
import json
import numpy as np


dataset = pd.read_csv('qapairs.csv', dtype=str)
print(dataset.head())
print(dataset.columns)
print(dataset['task'].unique())

tasks = ['Multiclass_Product_Classification', 'Product_Matching',
         'Product_Substitute_Identification', 'Product_Relation_Prediction',
         'Answer_Generation', 'Sequential_Recommendation']

for task in tasks:
    data = []
    task_data = dataset[dataset['task'] == task]
    for index, row in task_data.iterrows():
        instruction = row['instruction']
        input_text = row['input']
        options = row['options']
        if pd.notna(options) and options != "" and options != "null":
            if not input_text.endswith('\n'):
                input_text += '\n'
            input_text += f"options: {options}"
        output_text = row['adjust_output']
        data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    with open(f"./datasets/{task}.json", "w") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

random.seed(42)
np.random.seed(42)

train_dir = "./datasets/train"
os.makedirs(train_dir, exist_ok=True)
test_dir = "./datasets/test"
os.makedirs(test_dir, exist_ok=True)

all_train_data = []

print("Starting to process data for each task...")

for task in tasks:
    print(f"Processing task: {task}")

    with open(f"./datasets/{task}.json", "r", encoding='utf-8') as f:
        task_data = json.load(f)

    print(f"  - {task} total data: {len(task_data)}")

    random.shuffle(task_data)

    split_point = int(len(task_data) * 0.8)
    train_data = task_data[:split_point]
    test_data = task_data[split_point:]

    print(f"  - Train: {len(train_data)}, Test: {len(test_data)}")

    all_train_data.extend(train_data)

    train_file_path = os.path.join(train_dir, f"{task}_train.json")
    with open(train_file_path, "w", encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    print(f"  - Train saved to: {train_file_path}")

    test_file_path = os.path.join(test_dir, f"{task}_test.json")
    with open(test_file_path, "w", encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    print(f"  - Test saved to: {test_file_path}")

print(f"\nTotal combined training data: {len(all_train_data)}")
random.shuffle(all_train_data)

train_file_path = "./datasets/train_all.json"
with open(train_file_path, "w", encoding='utf-8') as f:
    json.dump(all_train_data, f, ensure_ascii=False, indent=4)

print(f"All training data merged and saved to: {train_file_path}")

print("\n=== Data Split Statistics ===")
for task in tasks:
    with open(f"./datasets/test/{task}_test.json", "r", encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"{task}: Test set {len(test_data)} items")

print(f"Merged training set: {len(all_train_data)} items")
print("Data processing completed!")
