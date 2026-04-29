from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification
from src.config import *
from src.utils import set_seed
from src.dataset import prepare_datasets, tokenize
from src.augment import augment_backtranslation

def run_experiment(experiment_name="baseline", debug=True):

    set_seed(SEED)

    train_dataset, val_dataset, test_dataset = prepare_datasets()

    mode = EXPERIMENTS[experiment_name]

    # Before Augmentation
    if debug:
        print("\n=== BEFORE AUGMENTATION ===")
        print(train_dataset[0][TEXT_COLUMN])

    # Training Data Augmentation
    train_dataset = augment_backtranslation(train_dataset, mode=mode)

    # After Augmentation
    if debug:
        print("\n=== AFTER AUGMENTATION ===")
        print(train_dataset[0][TEXT_COLUMN])


    # Tokenization
    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)


    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    training_args = TrainingArguments(
        output_dir=f"./results/{experiment_name}",
        eval_strategy="epoch",
        logging_strategy="epoch",
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
    )

    trainer.train()

    # Training Metric
    train_metrics = trainer.evaluate(train_dataset)

    test_metrics = trainer.evaluate(test_dataset)

    if debug:
        print("\n=== TRAIN METRICS ===")
        print(train_metrics)

        print("\n=== TEST METRICS ===")
        print(test_metrics)

    return {
        "train": train_metrics,
        "test": test_metrics
    }