import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _scores(result):
    labels = result['labels']
    preds = result['preds']

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds, average='weighted', zero_division=0),
        'f1': f1_score(labels, preds, average='weighted', zero_division=0),
    }


def build_ablation_table(results):
    rows = []

    for name, result in results.items():
        row = {'experiment': name}
        row.update(_scores(result))
        row['train_f1'] = result.get('train_f1')
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    return table


def print_delta_analysis(results, baseline_key='baseline'):
    if baseline_key not in results:
        print('\nNo baseline result found.')
        return

    baseline = _scores(results[baseline_key])

    print('\nDelta vs baseline:')
    for name, result in results.items():
        if name == baseline_key:
            continue

        scores = _scores(result)
        print(f'{name}:')
        print(f"  accuracy: {scores['accuracy'] - baseline['accuracy']:+.4f}")
        print(f"  f1:       {scores['f1'] - baseline['f1']:+.4f}")


def print_overfitting_check(results):
    print('\nTrain vs test F1:')

    for name, result in results.items():
        train_f1 = result.get('train_f1')
        test_f1 = _scores(result)['f1']

        if train_f1 is None:
            print(f'{name}: train F1 unavailable, test F1={test_f1:.4f}')
            continue

        gap = train_f1 - test_f1
        print(f'{name}: train={train_f1:.4f}, test={test_f1:.4f}, gap={gap:+.4f}')
