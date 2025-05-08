import os
import math
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Config
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 5
DATASET_NAME = "SetFit/sst5"


# Load Model
def load_base_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    return model


# Compute Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')

    print("\nClassification Report:")
    print(classification_report(labels, predictions))

    return {
        "accuracy": acc,
        "f1": f1
    }


# Plot Loss Curve
def plot_loss(train_losses, eval_losses, output_path="bert-loss-curve.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker='o')
    plt.plot(epochs, eval_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


# Preprocessing
def preprocess(tokenizer, dataset):
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized


# Train Model
def train_model(tokenized_dataset, tokenizer):
    model = load_base_model()

    args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer


# Evaluate Model
def evaluate_model(trainer, test_dataset):
    print("\nEvaluating model on test set...")
    results = trainer.evaluate(test_dataset)

    if "eval_loss" in results:
        try:
            perplexity = math.exp(results["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        results["perplexity"] = perplexity
        print(f"Perplexity: {perplexity:.2f}")

    print("Test Results:", results)
    return results


# Test Single Example
def test_single_example(trainer, tokenizer):
    model = trainer.model
    model.eval()

    examples = [
        "I absolutely loved this movie! It was amazing.",
        "This film was a complete waste of time."
    ]

    inputs = tokenizer(examples, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    for text, pred in zip(examples, predictions):
        print(f"\nInput: {text}")
        print(f"Predicted Label: {pred.item()}")


# === MAIN ===
def main():
    dataset = load_dataset(DATASET_NAME)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenized_dataset = preprocess(tokenizer, dataset)

    trainer = train_model(tokenized_dataset, tokenizer)

    evaluate_model(trainer, tokenized_dataset["test"])
    test_single_example(trainer, tokenizer)


if __name__ == "__main__":
    main()