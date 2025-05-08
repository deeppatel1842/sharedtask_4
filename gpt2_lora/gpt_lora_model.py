# gpt2_lora_model.py

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType

from config import MODEL_NAME, DATASET_NAME, MAX_LENGTH, BATCH_SIZE, EPOCHS
from evaluation_utils import plot_loss, compute_metrics, evaluate_model


class GPT2LoRATrainer:
    def __init__(self):
        # Cache paths for Hugging Face
        os.environ["HF_HOME"] = "/scratch/cs529315/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = "/scratch/cs529315/huggingface"
        os.environ["HF_DATASETS_CACHE"] = "/scratch/cs529315/datasets"
        os.environ["XDG_CACHE_HOME"] = "/scratch/cs529315/.cache"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model and prepare for LoRA
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        base_model.resize_token_embeddings(len(self.tokenizer))

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none"
        )

        self.model = get_peft_model(base_model, peft_config)
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
            output_dir="./experiments/gpt2-lora",
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            logging_dir="./logs",
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
        )

    def train(self):
        print("Starting LoRA fine-tuning...")
        trainer = Trainer(
            model=self.model,
            args=self.get_training_args(),
            train_dataset=self.tokenized_dataset["train"].select(range(500)),
            eval_dataset=self.tokenized_dataset["validation"].select(range(100)),
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )

        train_result = trainer.train()
        self.train_losses.append(train_result.training_loss)

        eval_result = trainer.evaluate()
        self.eval_losses.append(eval_result["eval_loss"])

        trainer.save_model("./experiments/gpt2-lora")
        self.tokenizer.save_pretrained("./experiments/gpt2-lora")

        print("Training complete!")
        print(f"Train Loss: {self.train_losses[-1]}")
        print(f"Eval Loss: {self.eval_losses[-1]}")

        plot_loss(self.train_losses, self.eval_losses)

        # Final evaluation (ROUGE + Perplexity)
        evaluate_model(
            trainer,
            self.tokenized_dataset["validation"].select(range(1000))
        )
