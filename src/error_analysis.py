import pandas as pd


def run_error_analysis(texts, labels, preds, label_names=None, n_samples=5):
    rows = []

    for text, label, pred in zip(texts, labels, preds):
        if label == pred:
            continue

        rows.append({
            'text': text,
            'label': label,
            'pred': pred,
            'label_name': _label_name(label, label_names),
            'pred_name': _label_name(pred, label_names),
        })

    errors = pd.DataFrame(rows)

    print(f'Errors: {len(errors)}')
    if len(errors) == 0:
        return errors

    for _, row in errors.head(n_samples).iterrows():
        print('\nExpected:', row['label_name'])
        print('Predicted:', row['pred_name'])
        print('Text:', _shorten(row['text']))

    return errors


def compare_errors_across_experiments(error_dfs):
    print('\nError counts:')

    for name, errors in error_dfs.items():
        print(f'{name}: {len(errors)}')


def _label_name(label, label_names):
    if not label_names:
        return label

    return label_names[int(label)]


def _shorten(text, limit=300):
    text = str(text).replace('\n', ' ').strip()

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + '...'
