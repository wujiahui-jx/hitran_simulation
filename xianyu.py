import numpy as np
from numpy.linalg import inv
import time
#%%
def text_to_numbers(text):
    """将文本转换为数字列表。"""
    return [ord(char) - ord('A') for char in text.upper() if char.isalpha()]

#%%
def numbers_to_text(numbers):
    """将数字列表转换回文本。"""
    #return ''.join(chr(num + ord('A')) for num in numbers)
    return ''.join([chr(num + ord('A')) for num in numbers])

#%%
def matrix_encrypt_decrypt(text, A, B, mode='encrypt'):
    """使用矩阵仿射变换进行加密或解密。"""
    numbers = text_to_numbers(text)
    print(numbers)
    transformed = []

    # 处理每个4字母的块
    # 处理每个4字母的块
    # 处理每个4字母的块
    for i in range(0, len(numbers), 4):
        block = np.squeeze(np.array(numbers[i:i + 4]))

        # 处理索引超出的情况
        if block.ndim == 0:  # 如果是1D数组
            block = np.expand_dims(block, axis=0)  # 将其转换为2D数组

        if block.shape[0] < 4:
            # 如果不足4个元素，可以根据具体需求进行处理，例如补充0
            block = np.pad(block, (0, 4 - block.shape[0]), mode='constant')

        if mode == 'encrypt':
            # 加密：C = A * M + B (mod 26)
            transformed_block = (np.dot(A, block) + B) % 26
        elif mode == 'decrypt':
            # 解密：M = A_inv * (C - B) (mod 26)
            A_inv = inv(A)  # 计算A的逆矩阵
            transformed_block = np.dot(A_inv, (block - B)) % 26
        transformed.extend([int(num) for num in transformed_block])

    return numbers_to_text(transformed) # 四舍五入并转换为整数

#%%
# 定义矩阵A和B
A =  np.squeeze(np.array([[3, 13, 21, 9], [15, 10, 6, 25], [10, 17, 4, 8], [1, 23, 7, 2]]))
B =  np.squeeze(np.array([1, 21, 8, 17]))

# 明文
plaintext = "PLEASE SEND ME THE BOOK, MY CREDIT CARD NO IS SIX ONE TWO ONE THE EIGHT SIX ZERO ONE SIX EIGHT FOUR NINE SEVEN ZERO TWO"

# 开始加密的时间
start_time_encrypt = time.time()
# 加密
encrypted_text = matrix_encrypt_decrypt(plaintext, A, B, mode='encrypt')
# 加密结束的时间