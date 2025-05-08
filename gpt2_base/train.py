from gpt2_base_model import GPT2BaseTrainer

def run_training():
    trainer = GPT2BaseTrainer()
    trainer.preprocess()
    trainer.train()
    return trainer
