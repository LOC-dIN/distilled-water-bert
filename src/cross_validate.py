import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.augment import augment_backtranslation
from src.config import *
from src.dataset import tokenize
from src.evaluate import compute_metrics
from src.utils import set_seed


def run_significance_tests(results):
    names = list(results.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]

            print(f'\n{a} vs {b}')
            mcnemar_test(
                results[a]['labels'],
                results[a]['preds'],
                results[b]['preds'],
            )


def mcnemar_test(labels, preds_a, preds_b):
    labels = np.array(labels)
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)

    a_correct = preds_a == labels
    b_correct = preds_b == labels

    a_only = np.sum(a_correct & ~b_correct)
    b_only = np.sum(~a_correct & b_correct)
    total = a_only + b_only

    print(f'a only correct: {a_only}')
    print(f'b only correct: {b_only}')

    if total == 0:
        print('p-value: unavailable')
        return None

    result = binomtest(min(a_only, b_only), total, p=0.5)
    print(f'p-value: {result.pvalue:.6f}')

    return result.pvalue


def wilcoxon_test(scores_a, scores_b):
    if len(scores_a) != len(scores_b):
        raise ValueError('Wilcoxon test needs paired score lists.')

    try:
        result = wilcoxon(scores_a, scores_b)
    except ValueError:
        print('statistic: unavailable')
        print('p-value: unavailable')
        return None

    print(f'statistic: {result.statistic:.6f}')
    print(f'p-value: {result.pvalue:.6f}')

    return result


def stratified_kfold_cv(full_dataset, n_splits=5, experiment_name='baseline'):
    from src.train import _load_transformers_classes

    Trainer, TrainingArguments, DistilBertForSequenceClassification = _load_transformers_classes()

    set_seed(SEED)

    labels = list(full_dataset[LABEL_COLUMN])
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_results = []
    fold_preds = []
    fold_labels = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        print(f'\nFold {fold}/{n_splits}: {experiment_name}')

        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=VALIDATION_SIZE,
            stratify=np.array(labels)[train_idx],
            random_state=SEED,
        )

        train_dataset = full_dataset.select(train_idx)
        val_dataset = full_dataset.select(val_idx)
        test_dataset = full_dataset.select(test_idx)

        mode = EXPERIMENTS[experiment_name]
        train_dataset = augment_backtranslation(train_dataset, mode=mode)

        train_dataset = train_dataset.map(tokenize, batched=True)
        val_dataset = val_dataset.map(tokenize, batched=True)
        test_dataset = test_dataset.map(tokenize, batched=True)

        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
        )

        training_args = TrainingArguments(
            output_dir=f'./results/{experiment_name}_fold_{fold}',
            eval_strategy='epoch',
            logging_strategy='epoch',
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        output = trainer.predict(test_dataset)
        preds = np.argmax(output.predictions, axis=-1)
        labels_out = output.label_ids
        f1 = f1_score(labels_out, preds, average='weighted', zero_division=0)

        fold_results.append({'fold': fold, 'f1': f1})
        fold_preds.append(preds.tolist())
        fold_labels.append(labels_out.tolist())

    return fold_results, fold_preds, fold_labels
