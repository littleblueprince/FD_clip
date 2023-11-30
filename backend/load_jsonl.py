# -*- coding: utf-8 -*-
# @Time    : 2023/11/16 21:26
# @Author  : blue
# @Description :

import base64
import csv
from io import BytesIO

import jsonlines
import torch
import torch.nn.functional as F
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name


def image_show(tsv_file_path, image_id):
    # 读取 TSV 文件
    with open(tsv_file_path, 'r', newline='', encoding='utf-8') as file:
        # 使用制表符分隔符创建 CSV 读取器
        reader = csv.reader(file, delimiter='\t')
        all_rows = list(reader)
        for image_info in all_rows:
            if image_info[0] == str(image_id):
                image_data = base64.b64decode(image_info[1])
                # 使用 Pillow 加载图像
                image = Image.open(BytesIO(image_data))
                # 展示图像
                image.show()


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()

text = clip.tokenize(["铁锅", "白色鞋子"]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)
    # 对特征进行归一化使用归一化后的图文特征用于下游任务torch.Size([1, 512])
    text_features /= text_features.norm(dim=-1, keepdim=True)
tsv_file_path = 'C:\Static\压缩包\MUGE\\valid_imgs.tsv'
file_path = './data/test.jsonl'
# 使用jsonlines库逐行读取JSONL文件
with jsonlines.open(file_path) as reader:
    for line in reader:
        image_id = line['image_id']
        image_feature = torch.Tensor(line['feature'])
        # 计算余弦相似度
        for text_feature in text_features:
            similarity = F.cosine_similarity(image_feature, text_feature.cpu(), dim=-1)
            if similarity > 0.38:
                print('-' * 80)
                print("Cosine Similarity:", similarity)
                print("image_id:", line['image_id'])
                image_show(tsv_file_path, line['image_id'])
