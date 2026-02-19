import json
import os
import re
from sklearn.metrics import f1_score
from bert_score import score


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def extract_answer(output, task_type):
    if task_type == 'classification':
        pattern = r'([A-Z])'
        match = re.search(pattern, output, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return 'A'

    if task_type == 'yes_no':
        pattern = r'\b(yes|no)\b'
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
        else:
            return 'no'

    if task_type == 'generation':
        return output.strip()

    return output.strip()


def evaluate_accuracy(predictions, references):
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(references) if references else 0


def evaluate_F1(predictions, references):
    return f1_score(references, predictions, pos_label='yes')


def evaluate_macro_F1(predictions, references):
    return f1_score(references, predictions, average='macro')


def evaluate_bert_score(predictions, references):
    P, R, F1 = score(predictions, references, lang='en', verbose=True)
    return F1.mean().item()


def main():
    classification_tasks = ['Multiclass_Product_Classification',
                            'Product_Relation_Prediction', 'Sequential_Recommendation']
    yes_no_tasks = ['Product_Matching', 'Product_Substitute_Identification']
    generation_tasks = ['Answer_Generation']

    task_to_dir = {
        'Multiclass_Product_Classification': 'MPC_eval',
        'Product_Relation_Prediction': 'PRP_eval',
        'Sequential_Recommendation': 'SR_eval',
        'Product_Matching': 'PM_eval',
        'Product_Substitute_Identification': 'PSI_eval',
        'Answer_Generation': 'AG_eval'
    }

    result = {}
    all_tasks = list(task_to_dir.keys())

    for task in all_tasks:
        task_dir = task_to_dir[task]
        task_type = 'classification' if task in classification_tasks else 'yes_no' if task in yes_no_tasks else 'generation'

        jsonl_file = os.path.join(task_dir, 'generated_predictions.jsonl')
        json_file = os.path.join(task_dir, 'generated_predictions.json')

        if os.path.exists(jsonl_file):
            data = load_jsonl(jsonl_file)
        elif os.path.exists(json_file):
            data = load_json(json_file)
        else:
            print(f"Warning: No prediction file found for {task}")
            continue

        predictions = [extract_answer(item['predict'], task_type) for item in data]
        references = [extract_answer(item['label'], task_type) for item in data]

        if task == 'Multiclass_Product_Classification':
            accuracy = evaluate_accuracy(predictions, references)
            result[task] = accuracy
            print(f"{task}: {accuracy:.4f}")
        if task == 'Product_Relation_Prediction':
            macro_f1 = evaluate_macro_F1(predictions, references)
            result[task] = macro_f1
            print(f"{task}: {macro_f1:.4f}")
        if task == 'Sequential_Recommendation':
            accuracy = evaluate_accuracy(predictions, references)
            result[task] = accuracy
            print(f"{task}: {accuracy:.4f}")
        if task == 'Product_Matching':
            f1 = evaluate_F1(predictions, references)
            result[task] = f1
            print(f"{task}: {f1:.4f}")
        if task == 'Product_Substitute_Identification':
            f1 = evaluate_F1(predictions, references)
            result[task] = f1
            print(f"{task}: {f1:.4f}")
        if task == 'Answer_Generation':
            bert_f1 = evaluate_bert_score(predictions, references)
            result[task] = bert_f1
            print(f"{task}: {bert_f1:.4f}")

    with open('./evaluation_results.json', 'w') as file:
        json.dump(result, file, indent=4)

    print("\nEvaluation results saved to ./evaluation_results.json")


if __name__ == "__main__":
    main()
