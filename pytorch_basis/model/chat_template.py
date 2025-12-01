from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-4B-Instruct')
print('Has chat_template:', hasattr(tokenizer, 'chat_template'))
print('Chat template length:', len(tokenizer.chat_template) if tokenizer.chat_template else 0)