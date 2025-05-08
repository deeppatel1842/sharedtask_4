# gpt2_base_model.py

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from config import MODEL_NAME, DATASET_NAME, MAX_LENGTH, BATCH_SIZE, EPOCHS
from evaluation_utils import plot_loss, compute_metrics, evaluate_model


class GPT2BaseTrainer:
    def __init__(self):
        # Cache paths for Hugging Face
        os.environ["HF_HOME"] = "/scratch/cs529315/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = "/scratch/cs529315/huggingface"
        os.environ["HF_DATASETS_CACHE"] = "/scratch/cs529315/datasets"
        os.environ["XDG_CACHE_HOME"] = "/scratch/cs529315/.cache"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.dataset = load_dataset(DATASET_NAME, cache_dir="/scratch/cs529315/hf_cache")

        self.train_losses = []
        self.eval_losses = []

    def tokenize_function(self, example):
        return self.tokenizer(
            example["document"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    def preprocess(self):
        print("Tokenizing...")
        tokenized = self.dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=self.dataset["train"].column_names,
        )
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.tokenized_dataset = tokenized

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def get_training_args(self):
        return TrainingArguments(
            output_dir="./experiments/gpt2-base",
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            evaluation_strategy="epoch",
            logging_dir="./logs",
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
        )

    def train(self):
        trainer = Trainer(
            model=self.model,
            args=self.get_training_args(),
            train_dataset=self.tokenized_dataset["train"].select(range(500)),      # or full dataset
            eval_dataset=self.tokenized_dataset["validation"].select(range(100)),  # or full dataset
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,  # For ROUGE
        )

        print("Starting training...")
        train_result = trainer.train()
        self.train_losses.append(train_result.training_loss)

        eval_result = trainer.evaluate()
        self.eval_losses.append(eval_result["eval_loss"])

        trainer.save_model("./experiments/gpt2-base")
        self.tokenizer.save_pretrained("./experiments/gpt2-base")

        print("Training complete!")
        print(f"Train Loss: {self.train_losses[-1]}")
        print(f"Eval Loss: {self.eval_losses[-1]}")

        plot_loss(self.train_losses, self.eval_losses)

        # Run final ROUGE + Perplexity Evaluation
        evaluate_model(
            trainer,
            self.tokenized_dataset["validation"].select(range(1000))
        )

