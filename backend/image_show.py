# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 10:31
# @Author  : blue
# @Description :


import csv
import random
import base64
from PIL import Image
from io import BytesIO

# 指定你的 TSV 文件路径
tsv_file_path = 'C:\Static\压缩包\MUGE\\valid_imgs.tsv'

# 读取 TSV 文件
with open(tsv_file_path, 'r', newline='', encoding='utf-8') as file:
    # 使用制表符分隔符创建 CSV 读取器
    reader = csv.reader(file, delimiter='\t')

    # 获取文件中的所有行
    all_rows = list(reader)

    # 获取文件的标题行
    header = all_rows[0]

    # 随机选择 5 行（不包括标题）
    random_rows = random.sample(all_rows[1:], 5)

    # 处理获取的图片信息
    for row in random_rows:
        # row 是一个包含每一行数据的列表
        # 你可以通过索引访问每一列的数据
        # 例如，如果文件的第一列是图片路径，可以使用 row[0]

        # 在这里，你可以处理图片信息
        print(row)
        # 图片信息
        image_info = row
        # 解码 Base64 图片数据
        print(image_info[0])
        image_data = base64.b64decode(image_info[1])
        # 使用 Pillow 加载图像
        image = Image.open(BytesIO(image_data))
        # 展示图像
        image.show()
        input()
