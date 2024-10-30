import os
import numpy as np
import pickle
import tiktoken

# 获取 GPT-2 的编码器
enc = tiktoken.get_encoding("gpt2")

# 解码函数
def dec(tokens):
    # 将 token ID 转换为对应的字符串
    return enc.decode(tokens)

# 数据集路径
data_path = r'D:\vscode_code\Deep_learning\nanoChatGPT\data\Chat\train.bin'
val_data_path = r'D:\vscode_code\Deep_learning\nanoChatGPT\data\Chat\val.bin'
meta_path = r'D:\vscode_code\Deep_learning\nanoChatGPT\data\meta.pkl'

# 从二进制文件中读取训练数据
train_ids = np.fromfile(data_path, dtype=np.uint16)
val_ids = np.fromfile(val_data_path, dtype=np.uint16)


vocab_size = meta['vocab_size']
# char2num = meta['char2num']
# num2char = meta['num2char']

# 保存元信息，用于后续编码解码
meta = {
    'vocab_size':vocab_size,
}


with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

print(f"训练数据长度：{len(train_ids)}")
print(f"验证数据长度：{len(val_ids)}")
print(f"字符表大小：{vocab_size:,}")



# 示例：解码部分训练数据并输出
sample_length = 100  # 你可以根据需要调整
decoded_sample = dec(train_ids[:sample_length])
print(f"解码示例（前{sample_length}个tokens）:\n{decoded_sample}")
