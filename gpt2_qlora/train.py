from gpt_qlora import GPT2qloraTrainer

def run_training():
    trainer = GPT2qloraTrainer()
    trainer.preprocess()
    trainer.train()
    return trainer
