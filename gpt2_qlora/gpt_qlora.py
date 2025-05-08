import os
import math
import torch
import wandb
import evaluate
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType


class GPT2qloraTrainer:
    def __init__(self):
        # Cache config
        os.environ["HF_HOME"] = "/scratch/cs529315/qlora_cache"
        os.environ["TRANSFORMERS_CACHE"] = "/scratch/cs529315/qlora_cache"
        os.environ["HF_DATASETS_CACHE"] = "/scratch/cs529315/datasets"
        os.environ["XDG_CACHE_HOME"] = "/scratch/cs529315/.cache"

        self.model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.trainer = None

    def preprocess(self):
        print("Loading and tokenizing dataset...")
        dataset = load_dataset("xsum", trust_remote_code=True)

        def preprocess_batch(example):
            texts = [
                "summarize: " + doc + self.tokenizer.eos_token + summ
                for doc, summ in zip(example["document"], example["summary"])
            ]
            return self.tokenizer(texts, padding="max_length", truncation=True, max_length=512)

        tokenized = dataset.map(preprocess_batch, batched=True, remove_columns=dataset["train"].column_names)
        self.train_dataset = tokenized["train"]
        self.eval_dataset = tokenized["validation"]

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        logits = logits.to(self.model.device)
        labels = labels.to(self.model.device)

        preds = logits.argmax(dim=-1)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_tokens = [word_tokenize(pred) for pred in decoded_preds]
        label_tokens = [[word_tokenize(label)] for label in decoded_labels]

        bleu_score = self.bleu.compute(predictions=pred_tokens, references=label_tokens)["bleu"]
        rouge_scores = self.rouge.compute(predictions=decoded_preds, references=decoded_labels)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.transpose(1, 2), labels).item()
        perplexity = math.exp(loss) if loss < 20 else float("inf")

        return {
            "bleu": bleu_score,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "loss": loss,
            "perplexity": perplexity
        }

    def train(self):
        print("Loading quantized GPT-2 with LoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none"
        ))

        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")

        wandb.login()
        wandb.init(project="gpt2-qlora", name="gpt2-qlora-xsum")

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=100,
            fp16=True,
            report_to="wandb",
            run_name="gpt2-qlora-xsum"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            compute_metrics=self.compute_metrics
        )

        print("Starting training...")
        self.trainer.train()

    @property
    def state(self):
        return self.trainer.state

    def evaluate(self):
        return self.trainer.evaluate()

    def predict(self, inputs):
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids, max_new_tokens=60)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

