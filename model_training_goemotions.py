# model_traning_goemotions.py
import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset, Sequence, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)


print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available?", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------
# Config
# -----------------------
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "my_goemotions_model"
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
SEED = 42
NUM_LABELS = 28

set_seed(SEED)


def main():
    # Check if we have a GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)  # e.g. 'cuda' or 'cpu'

    # 1) Load dataset
    dataset = load_dataset("go_emotions")

    # 2) Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3) Tokenize and create multi-hot float labels
    def tokenize_and_convert_labels(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

        # Convert each 'labels' list to a float multi-hot vector of length 28
        new_labels = []
        for label_list in batch["labels"]:
            multi_hot = np.zeros(NUM_LABELS, dtype=np.float32)
            for idx in label_list:
                multi_hot[idx] = 1.0
            new_labels.append(multi_hot.tolist())

        tokenized["labels"] = new_labels
        return tokenized

    dataset = dataset.map(tokenize_and_convert_labels, batched=True)

    # 4) Force the labels column to float, so HF 'Dataset' recognizes them as float
    dataset = dataset.cast_column("labels", Sequence(Value("float32"), length=NUM_LABELS))

    # 5) Remove any extra single-label column, if it exists
    col_names = dataset["train"].column_names
    if "label" in col_names:  # Sometimes the dataset has a single-label column too
        dataset = dataset.remove_columns(["label"])

    # 6) Convert final dataset to PyTorch Tensors
    keep_cols = ["input_ids", "attention_mask", "labels"]
    dataset["train"].set_format(type="torch", columns=keep_cols)
    dataset["validation"].set_format(type="torch", columns=keep_cols)
    dataset["test"].set_format(type="torch", columns=keep_cols)

    # 7) Create the multi-label classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"  # ensures BCEWithLogitsLoss
    )

    # 8) Define training arguments -- note `no_cuda=False` forces use of GPU if available
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",  # Evaluate each epoch
        save_strategy="epoch",  # Save model each epoch
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        no_cuda=False  # <--- ensures Trainer uses GPU if torch.cuda.is_available() is True
    )

    # 9) Define the custom metrics function for multi-label
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Convert logits to probabilities with sigmoid
        probs = 1 / (1 + np.exp(-logits))
        # Binarize at 0.5
        preds = (probs >= 0.5).astype(int)

        # Micro-averaged stats
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # 10) Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,  # optional, for logging or saving
        compute_metrics=compute_metrics,
    )

    # Move model to device (usually Trainer handles this, but we can do it explicitly too)
    model.to(device)

    # 11) Train the model
    trainer.train()

    # 12) Evaluate on the test set
    test_metrics = trainer.evaluate(dataset["test"])
    print("\nTest set performance:", test_metrics)

    # 13) Save model & tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel + tokenizer saved to '{OUTPUT_DIR}'.")

    # 14) Optional: additional analysis
    preds_output = trainer.predict(dataset["test"])
    test_logits, test_labels = preds_output.predictions, preds_output.label_ids
    test_probs = 1 / (1 + np.exp(-test_logits))
    test_preds = (test_probs >= 0.5).astype(int)
    # ... further analysis or classification reports


if __name__ == "__main__":
    main()
