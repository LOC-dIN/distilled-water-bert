# Model Settings
MODEL_NAME = 'distilbert/distilbert-base-multilingual-cased'
BACKTRANSLATION_MODEL_NAME = 'facebook/nllb-200-distilled-600M'
NUM_LABELS = 2
MAX_LENGTH = 512 # DistilBERT has a fixed 512 token max for the input

# Training Hyperparameters
'''
Base Batch Size Value is 16 (Perfect Middle Ground for Hardware Constraints)

Increase value by 2x if these conditions are met (Slow Training Time):
    -> GPU is underutilized
    -> Updates feel slow

Decrease value by 1/2 if these conditions are met:
    -> Loss fluctuates heavily

'''
BATCH_SIZE = 16


'''
Base Learning Rate Value is 2e-5 (2x10^-5 or 0.00002)

Increase value by 1 if these conditions are met:
    -> Training loss stays high
    -> Validation loss also high
    -> Accuracy stuck

Decrease value by 1 if these conditions are met:
    -> Training loss jumps randomly
    -> Model performance fluctuates

'''
LEARNING_RATE = 2e-5


'''
Base Epoch Value is 3

Increase value by 1 or 2 if these conditions are met (Underfitting):
    -> Training loss still high
    -> Validation performance poor
    -> Model hasn’t “learned enough”

Decrease value by 1 if these conditions are met (Overfitting):
    -> Training loss keeps decreasing
    -> Validation loss starts increasing
    -> Big gap between train vs validation performance

If epochs value increase/decrease doesn't change anything:
    -> Use better regularization
    -> Use stronger augmentation

'''
EPOCHS = 3


'''
Base Weight Decay Value is 0.01

Increase value by +0.05 if these conditions are met (Overfitting):
    -> Training accuracy very high
    -> Validation accuracy much lower

'''
WEIGHT_DECAY = 0.01


# Dataset Settings
TEXT_COLUMN = 'article'
LABEL_COLUMN = 'label'
SEED = 2026
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Tokenization Settings
PADDING = 'max_length'
TRUNCATION = True

# Experiment Map
EXPERIMENTS = {
    'baseline': None,
    'backtranslation_en': 'en',
    'backtranslation_ko': 'ko'
}
