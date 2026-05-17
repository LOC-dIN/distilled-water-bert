import numpy as np
import importlib
import traceback
from sklearn.metrics import f1_score
from src.config import *
from src.evaluate import compute_metrics
from src.utils import set_seed
from src.dataset import prepare_datasets, tokenize
from src.augment import augment_backtranslation


def _log(message):
    print(f'[train] {message}', flush=True)


def _fail(stage, error):
    print(f'\n[train] Failed during: {stage}', flush=True)
    print(f'[train] {type(error).__name__}: {error}', flush=True)
    traceback.print_exc()
    raise RuntimeError(f'run_experiment failed during {stage}') from error


def _check_columns(dataset, name):
    missing = []

    for column in [TEXT_COLUMN, LABEL_COLUMN]:
        if column not in dataset.column_names:
            missing.append(column)

    if missing:
        raise ValueError(f'{name} dataset is missing columns: {missing}')


def _load_transformers_classes():
    _log('importing Trainer and model classes')

    trainer_modules = [
        'transformers.integrations',
        'safetensors.torch',
        'torch',
        'torch.distributed',
        'huggingface_hub',
        'transformers.configuration_utils',
        'transformers.data.data_collator',
        'transformers.debug_utils',
        'transformers.feature_extraction_sequence_utils',
        'transformers.feature_extraction_utils',
        'transformers.hyperparameter_search',
        'transformers.image_processing_utils',
        'transformers.integrations.deepspeed',
        'transformers.integrations.fsdp',
        'transformers.integrations.liger',
        'transformers.integrations.neftune',
        'transformers.integrations.peft',
        'transformers.integrations.tpu',
        'transformers.modelcard',
        'transformers.modeling_utils',
        'transformers.models.auto.modeling_auto',
        'transformers.optimization',
        'transformers.processing_utils',
        'transformers.tokenization_utils_base',
        'transformers.trainer_callback',
        'transformers.trainer_optimizer',
        'transformers.trainer_pt_utils',
        'transformers.trainer_utils',
        'transformers.training_args',
    ]

    for module in trainer_modules:
        importlib.import_module(module)

    from transformers import DistilBertForSequenceClassification
    from transformers.trainer import Trainer
    from transformers.training_args import TrainingArguments

    return Trainer, TrainingArguments, DistilBertForSequenceClassification


def run_experiment(experiment_name="baseline", debug=True):

    try:
        set_seed(SEED)

        if experiment_name not in EXPERIMENTS:
            raise ValueError(f'Unknown experiment: {experiment_name}')
    except Exception as error:
        _fail('setup', error)

    try:
        _log('loading dataset')
        train_dataset, val_dataset, test_dataset = prepare_datasets()

        _check_columns(train_dataset, 'train')
        _check_columns(val_dataset, 'validation')
        _check_columns(test_dataset, 'test')
    except Exception as error:
        _fail('dataset loading', error)

    mode = EXPERIMENTS[experiment_name]
    raw_texts = list(test_dataset[TEXT_COLUMN])

    # Before Augmentation
    if debug:
        print("\n=== BEFORE AUGMENTATION ===")
        print(train_dataset[0][TEXT_COLUMN])

    try:
        _log(f'augmenting training data: {mode or "none"}')
        train_dataset = augment_backtranslation(train_dataset, mode=mode)
    except Exception as error:
        _fail('augmentation', error)

    # After Augmentation
    if debug:
        print("\n=== AFTER AUGMENTATION ===")
        print(train_dataset[0][TEXT_COLUMN])


    try:
        _log('tokenizing datasets')
        train_dataset = train_dataset.map(tokenize, batched=True)
        val_dataset = val_dataset.map(tokenize, batched=True)
        test_dataset = test_dataset.map(tokenize, batched=True)
    except Exception as error:
        _fail('tokenization', error)


    try:
        Trainer, TrainingArguments, DistilBertForSequenceClassification = _load_transformers_classes()

        _log(f'loading model: {MODEL_NAME}')
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS
        )
    except Exception as error:
        _fail('model loading', error)

    try:
        _log('building trainer')
        training_args = TrainingArguments(
            output_dir=f"./results/{experiment_name}",
            eval_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            eval_accumulation_steps=16,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
    except Exception as error:
        _fail('trainer setup', error)

    try:
        _log('starting training')
        trainer.train()
    except RuntimeError as error:
        if 'out of memory' in str(error).lower():
            _fail('training: out of memory, try lowering BATCH_SIZE in src/config.py', error)

        _fail('training', error)
    except Exception as error:
        _fail('training', error)

    try:
        _log('predicting train set')
        train_output = trainer.predict(train_dataset)
        train_preds = np.argmax(train_output.predictions, axis=-1)
        train_f1 = f1_score(
            train_output.label_ids,
            train_preds,
            average='weighted',
            zero_division=0
        )
    except RuntimeError as error:
        if 'out of memory' in str(error).lower():
            _fail('train prediction: out of memory, try lowering BATCH_SIZE in src/config.py', error)

        _fail('train prediction', error)
    except Exception as error:
        _fail('train prediction', error)

    try:
        _log('predicting test set')
        test_output = trainer.predict(test_dataset)
        test_logits = test_output.predictions
        test_labels = test_output.label_ids
        test_preds = np.argmax(test_logits, axis=-1)

        exp_logits = np.exp(test_logits - test_logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        pos_probs = probs[:, 1]
    except RuntimeError as error:
        if 'out of memory' in str(error).lower():
            _fail('test prediction: out of memory, try lowering BATCH_SIZE in src/config.py', error)

        _fail('test prediction', error)
    except Exception as error:
        _fail('test prediction', error)

    if debug:
        print("\n=== TRAIN METRICS ===")
        print({'f1': train_f1})

        print("\n=== TEST METRICS ===")
        print(compute_metrics((test_logits, test_labels)))

    return {
        "labels": test_labels.tolist(),
        "preds": test_preds.tolist(),
        "probs": pos_probs.tolist(),
        "texts": raw_texts,
        "train_f1": float(train_f1),
    }
