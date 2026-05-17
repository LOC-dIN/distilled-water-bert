import time
import argparse
import numpy as np
from src.train import run_experiment
from src.evaluate import full_evaluation
from src.ablation import build_ablation_table, print_delta_analysis, print_overfitting_check
from src.error_analysis import run_error_analysis, compare_errors_across_experiments
from src.cross_validate import stratified_kfold_cv, run_significance_tests, wilcoxon_test


EXPERIMENTS = [
    'baseline',
    'backtranslation_en',
    'backtranslation_ko'
]

LABEL_NAMES = ['real', 'fake']


def main(run_cv=False):
    all_results = {}
    error_dfs = {}
    cv_scores = {}

    for exp in EXPERIMENTS:
        print(f'\nRunning experiment: {exp}\n')

        start = time.time()
        results = run_experiment(exp)
        duration = time.time() - start

        print(f"{exp} took {duration:.2f} seconds")

        all_results[exp] = results

        print(f'\nEvaluation: {exp}')
        full_evaluation(
            labels=results['labels'],
            preds=results['preds'],
            probs=results['probs'],
            label_names=LABEL_NAMES,
        )

    print('\nABLATION RESULTS:')
    build_ablation_table(all_results)
    print_delta_analysis(all_results)
    print_overfitting_check(all_results)

    print('\nERROR ANALYSIS:')
    for exp, results in all_results.items():
        print(f'\n{exp}:')
        error_dfs[exp] = run_error_analysis(
            texts=results['texts'],
            labels=results['labels'],
            preds=results['preds'],
            label_names=LABEL_NAMES,
        )

    compare_errors_across_experiments(error_dfs)

    print('\nSIGNIFICANCE TESTS:')
    significance_inputs = {
        exp: {
            'preds': np.array(results['preds']),
            'labels': np.array(results['labels']),
        }
        for exp, results in all_results.items()
    }
    run_significance_tests(significance_inputs)

    if run_cv:
        from src.dataset import load_full_tokenized_dataset

        full_dataset = load_full_tokenized_dataset()

        for exp in EXPERIMENTS:
            fold_results, _, _ = stratified_kfold_cv(
                full_dataset=full_dataset,
                n_splits=5,
                experiment_name=exp,
            )
            cv_scores[exp] = [result['f1'] for result in fold_results]

        print('\nWILCOXON TESTS:')
        names = list(cv_scores.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                print(f'\n{names[i]} vs {names[j]}')
                wilcoxon_test(cv_scores[names[i]], cv_scores[names[j]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', action='store_true')
    args = parser.parse_args()

    main(run_cv=args.cv)
