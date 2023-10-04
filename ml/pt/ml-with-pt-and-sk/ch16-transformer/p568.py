from transformers import GPT2Tokenizer 
tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
text="Let us encoded this sentence"
encoded_input=tokenizer(text, return_tensors='pt')
print(encoded_input)

from transformers import GPT2Model
model=GPT2Model.from_pretrained('gpt2')
output=model(**encoded_input)
print(output['last_hidden_state'].shape)
print(output)
