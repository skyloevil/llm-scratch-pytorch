from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B-Base",
    trust_remote_code=True,
    device_map="cpu",          # 或者"auto"
    torch_dtype="auto"
)
print(model)                   # 打印顶层结构


#for name, module in model.named_modules():
#    print(name, module.__class__.__name__)


for name, param in model.named_parameters():
    print(f"{name}: {list(param.shape)}\n")


