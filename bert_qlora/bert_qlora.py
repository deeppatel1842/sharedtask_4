import os
import math
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType

# Config
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 5
DATASET_NAME = "SetFit/sst5"


# Load QLoRA Model
def load_qlora_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense"]
    )

    model = get_peft_model(model, lora_config)

    print("QLoRA BERT model loaded with 4-bit quantization and LoRA adapters.")
    model.print_trainable_parameters()

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
def plot_loss(train_losses, eval_losses, output_path="bert-qlora-loss-curve.png"):
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
    model = load_qlora_model()

    # Print parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n\U0001F4CA Parameter Summary:")
    print(f"Total parameters       : {total_params:,}")
    print(f"Trainable parameters   : {trainable_params:,}")
    print(f"Trainable percentage   : {100 * trainable_params / total_params:.2f}%\n")

    args = TrainingArguments(
        output_dir="./results",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_num_workers=4,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model.to(device)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    examples = [
        "I absolutely loved this movie! It was amazing.",
        "This film was a complete waste of time."
    ]

    inputs = tokenizer(examples, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    for text, pred in zip(examples, predictions):
        print(f"\nInput: {text}")
        print(f"Predicted Label: {pred.item()}")


# === MAIN ===
def main():
    dataset = load_dataset(DATASET_NAME)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenized_dataset = preprocess(tokenizer, dataset)

    trainer = train_model(tokenized_dataset, tokenizer)

    evaluate_model(trainer, tokenized_dataset["test"])
    test_single_example(trainer, tokenizer)


if __name__ == "__main__":
    main()