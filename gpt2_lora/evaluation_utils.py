# evaluation_utils.py

import matplotlib.pyplot as plt
import evaluate
import math


# 1. Plot training and validation loss
def plot_loss(train_losses, eval_losses, output_path="experiments/gpt2-loss-curve.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker='o')
    plt.plot(epochs, eval_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Evaluation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


# 2. Compute ROUGE during training
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    decoded_preds = [str(p).strip() for p in predictions]
    decoded_labels = [str(l).strip() for l in labels]

    results = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"]
    }


# 3. Evaluate final model (Perplexity + ROUGE)
def evaluate_model(trainer, eval_dataset):
    print("\n Evaluating model on validation data...")

    eval_results = trainer.evaluate(eval_dataset=eval_dataset)

    # Calculate perplexity
    perplexity = math.exp(eval_results["eval_loss"]) if eval_results["eval_loss"] < 100 else float("inf")
    print(f"\n Perplexity: {perplexity:.2f}")

    for key in eval_results:
        if key.startswith("eval_rouge"):
            print(f"{key}: {eval_results[key]:.4f}")

    return perplexity, {k: eval_results[k] for k in eval_results if k.startswith("eval_rouge")}

