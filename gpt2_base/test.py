def run_testing(trainer):
    print("Running inference on test input...")
    tokenizer = trainer.tokenizer
    model = trainer.model
    model.eval()

    prompt = "The government announced a new policy to"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n Generated text:")
    print(generated)
