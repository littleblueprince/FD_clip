# -*- coding: utf-8 -*-
# @Time    : 2023/11/14 10:46
# @Author  : blue
# @Description :


import torch
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name,available_models

print("Available models:", available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()
image = preprocess(Image.open("C://Static/Chinese-CLIP/examples/pokemon.jpeg")).unsqueeze(0).to(device)
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # 对特征进行归一化使用归一化后的图文特征用于下游任务
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    print(image_features)
    # print(text_features)

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)