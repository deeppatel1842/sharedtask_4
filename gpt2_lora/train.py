from gpt_lora_model import GPT2LoRATrainer

def run_training():
    trainer = GPT2LoRATrainer()
    trainer.preprocess()
    trainer.train()
    return trainer
