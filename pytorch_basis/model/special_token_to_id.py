"""
演示如何将特殊 token 转换成 token ID

Qwen3-VL 使用的特殊 token:
- <|vision_start|>: 视觉内容开始标记
- <|vision_end|>: 视觉内容结束标记
- <|image_pad|>: 图像占位符
- <|video_pad|>: 视频占位符
- <|im_start|>: ChatML 消息开始标记
- <|im_end|>: ChatML 消息结束标记
"""

from transformers import AutoTokenizer


def main():
    # 加载 Qwen3-VL tokenizer
    print("正在加载 Qwen3-VL tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-4B-Instruct')

    print("\n" + "="*60)
    print("方法 1: 使用 tokenizer.convert_tokens_to_ids()")
    print("="*60)

    # 定义要转换的特殊 token
    special_tokens = [
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|im_start|>",
        "<|im_end|>",
    ]

    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"{token:20s} → {token_id}")

    print("\n" + "="*60)
    print("方法 2: 使用 tokenizer.encode()")
    print("="*60)

    # 使用 encode 方法（会添加特殊 token 如 BOS）
    for token in special_tokens:
        # add_special_tokens=False 避免添加 BOS/EOS
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        print(f"{token:20s} → {token_ids}")

    print("\n" + "="*60)
    print("方法 3: 完整的视觉 placeholder 转换")
    print("="*60)

    # 完整的 image placeholder
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    image_ids = tokenizer.encode(image_placeholder, add_special_tokens=False)
    print(f"\nImage placeholder:")
    print(f"  文本: {image_placeholder}")
    print(f"  Token IDs: {image_ids}")
    print(f"  Token 数量: {len(image_ids)}")

    # 完整的 video placeholder
    video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
    video_ids = tokenizer.encode(video_placeholder, add_special_tokens=False)
    print(f"\nVideo placeholder:")
    print(f"  文本: {video_placeholder}")
    print(f"  Token IDs: {video_ids}")
    print(f"  Token 数量: {len(video_ids)}")

    print("\n" + "="*60)
    print("方法 4: 查看 tokenizer 的特殊 token 映射")
    print("="*60)

    # 查看所有添加的特殊 token
    print("\n所有特殊 token:")
    if hasattr(tokenizer, 'added_tokens_encoder'):
        for token, token_id in sorted(tokenizer.added_tokens_encoder.items(),
                                     key=lambda x: x[1]):
            if any(special in token for special in ['vision', 'image', 'video', 'im_']):
                print(f"  {token:20s} → {token_id}")

    print("\n" + "="*60)
    print("方法 5: 实际应用示例")
    print("="*60)

    # 模拟一个包含视频的对话
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},  # 这会被替换为 video placeholder
                {"type": "text", "text": "这个视频里有什么?"}
            ]
        }
    ]

    # 手动构建 prompt (模拟 chat template 的输出)
    prompt = "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>这个视频里有什么?<|im_end|>\n<|im_start|>assistant\n"

    print(f"\n构建的 prompt:\n{prompt}")

    # 转换为 token IDs
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"\nToken IDs: {token_ids}")
    print(f"Token 数量: {len(token_ids)}")

    # 解码回文本验证
    decoded = tokenizer.decode(token_ids)
    print(f"\n解码回的文本:\n{decoded}")

    print("\n" + "="*60)
    print("方法 6: 使用 apply_chat_template 自动处理")
    print("="*60)

    # 注意: apply_chat_template 不会实际替换 video placeholder
    # 这需要在模型层面处理
    # 但我们可以看到它生成的文本格式
    messages = [
        {
            "role": "user",
            "content": "这个视频里有什么?"
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"\napply_chat_template 生成的 prompt:\n{formatted_prompt}")

    # 转换为 token IDs
    formatted_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )
    print(f"\nToken IDs: {formatted_ids}")
    print(f"Token 数量: {len(formatted_ids)}")


if __name__ == "__main__":
    main()