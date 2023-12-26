from transformers import pipeline, set_seed
generator=pipeline('text-generation', model='gpt2')

set_seed(123)
generated_text=generator('Piggy roblox game is', max_length=20, num_return_sequences=3)
print(generated_text)

