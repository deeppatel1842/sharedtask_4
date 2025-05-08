from train import run_training
from test import run_testing
from evaluation_utils import plot_loss
import os

def configure_scratch_cache():
    os.environ["HF_HOME"] = "/scratch/cs529315/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/scratch/cs529315/huggingface"
    os.environ["HF_DATASETS_CACHE"] = "/scratch/cs529315/datasets"
    os.environ["XDG_CACHE_HOME"] = "/scratch/cs529315/.cache"

def main():
    configure_scratch_cache()
    print(" Starting GPT-2 Training Pipeline")

    # 1. Train the model and get the trainer object
    trainer = run_training()

    # 2. Plot training vs validation loss
    plot_loss(trainer.train_losses, trainer.eval_losses)

    # 3. Run a sample test generation
    run_testing(trainer)

if __name__ == "__main__":
    main()

