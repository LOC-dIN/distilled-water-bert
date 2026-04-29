import time
from src.train import run_experiment

def main():

    experiments = [
        'baseline',
        'backtranslation_en',
        'backtranslation_ko'
    ]

    all_results = {}

    for exp in experiments:
        print(f'\nRunning experiment: {exp}\n')

        start = time.time()
        results = run_experiment(exp)
        duration = time.time() - start

        print(f"{exp} took {duration:.2f} seconds")

        all_results[exp] = results

    print('\nFINAL RESULTS:')
    for k, v in all_results.items():
        print(f"{k}:")
        print(v["test"])
        print("-" * 40)


if __name__ == '__main__':
    main()
