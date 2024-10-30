import os
import requests
import tiktoken
import pickle
import numpy as np

# 初始化训练和验证 ID 列表
train_ids = []
val_ids = []
# 获取 GPT-2 的编码器
enc = tiktoken.get_encoding("gpt2")

# 解码函数
def dec(tokens):
    # 将 token ID 转换为对应的字符串
    return enc.decode(tokens)

'''
def download_file(url):
    """下载数据集文件并保存为 dataset.txt"""
    print("正在下载...")
    response = requests.get(url)
    if response.status_code == 200:
        with open('dataset.txt', 'wb') as f:
            f.write(response.content)
            print("下载数据集成功，正在进行分词处理")
    else:
        print('下载文件时出错:', response.status_code)

# 下载数据集文件
download_file('https://huggingface.co/VatsaDev/ChatGpt-nano/resolve/main/Dataset.txt')

def split_file(filename, output_dir, chunk_size):
    """将文件分割成多个小文件，每个小文件包含指定数量的行"""
    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 读取原始文件的所有行
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 计算要分割的小文件的数量
    n_chunks = len(lines) // chunk_size
    for i in range(n_chunks):
        start = i * chunk_size  # 当前小文件的起始行
        end = min((i + 1) * chunk_size, len(lines))  # 当前小文件的结束行

        chunk_lines = lines[start:end]  # 获取当前小文件的行

        # 生成小文件的输出文件名
        output_filename = os.path.join(output_dir, f'{i}-dataset.txt')
        # 将当前小文件的行写入新的文件
        with open(output_filename, 'w') as f:
            f.writelines(chunk_lines)

# 将下载的文件分割成每个包含 100,000 行的小文件
split_file('dataset.txt', 'output', 100000)
'''
def is_numbers(string):
    """检查字符串的前两个字符是否为数字"""
    two_chars = string[-6:-4]  # 获取前两个字符

    try:
        int(two_chars)  # 尝试将其转换为整数
        return True  # 如果成功，则返回 True
    except ValueError:
        return False  # 如果失败，则返回 False

# 存储所有出现过的字符
all_chars = set()

# 遍历指定目录中的所有文件

output_path = r'D:\vscode_code\Deep_learning\nanoChatGPT\data\Chat'
for filename in os.listdir(output_path):
    if filename.endswith('.txt'):  # 仅处理以 .txt 结尾的文件
        if is_numbers(filename):  # 检查文件名是否为数字
            num_prefix = int(filename[-6:-4])  # 获取前两位数字
            with open(os.path.join(output_path, filename), 'r', encoding='utf-8') as f:
                data = f.read()  # 读取文件内容
                # 添加所有字符到集合
                all_chars.update(data)

                print(f"文件: {filename}, 内容长度: {len(data)}")  # 打印文件内容长度
            if num_prefix < 48:  # 如果文件名的前两位数字小于 48
                tokens = enc.encode_ordinary(data)  # 编码文件内容
                train_ids += tokens  # 将内容编码并添加到训练 ID 列表
                print(f"训练数据包含 {len(tokens)} 个 tokens")  # 打印token数量和内容
            elif num_prefix > 48:  # 如果文件名的前两位数字大于 48
                tokens = enc.encode_ordinary(data)  # 编码文件内容
                val_ids += tokens  # 将内容编码并添加到验证 ID 列表
                print(f"验证数据包含 {len(tokens)} 个 tokens")  # 打印token数量和内容


                
# 打印训练和验证 ID 的 token 数量
print(f"训练数据包含 {len(train_ids):,} 个 tokens")
print(f"验证数据包含 {len(val_ids):,} 个 tokens")

vocab_size = len(all_chars)

# 保存元信息，用于后续编码解码
meta = {
    'vocab_size':vocab_size,
}

meta_path = r'D:\vscode_code\Deep_learning\nanoChatGPT\data\meta.pkl'
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)
'''
# 将训练和验证 ID 列表转换为 NumPy 数组，并指定数据类型为 uint16
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# 将训练和验证数据分别保存为 train.bin 和 val.bin
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train1.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val1.bin'))
'''