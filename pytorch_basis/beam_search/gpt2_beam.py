from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "I am a"
input_ids = tokenizer.encode(text, return_tensors='pt')

output = model.generate(input_ids, 
            max_length=10,  
            num_beams=5, 
            num_return_sequences=5, 
            early_stopping=True)

print(output.shape)
# torch.Size([5, 10])

for out in output:
  print(tokenizer.decode(out))