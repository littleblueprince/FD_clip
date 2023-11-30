# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 10:15
# @Author  : blue
# @Description :


import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()
image = preprocess(Image.open("../examples/pokemon.jpeg")).unsqueeze(0).to(device)
print(image.shape)

text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # 对特征进行归一化使用归一化后的图文特征用于下游任务
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

print(f"image_features.shape", image_features.shape)
print('-' * 80)
print(f"text_features.shape", text_features.shape)


# 计算余弦相似度
for text_feature in text_features:
    similarity = F.cosine_similarity(image_features, text_feature, dim=-1)
    print("Cosine Similarity:", similarity)
