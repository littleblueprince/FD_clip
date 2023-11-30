# -*- coding: utf-8 -*-
# @Time    : 2023/11/16 14:04
# @Author  : blue
# @Description :  将图片的embedding建立并且存储到jsonl文件中
import base64
import csv
import json
from io import BytesIO
import torch
from PIL import Image
from tqdm import tqdm
from cn_clip.clip import load_from_name


def decode_base64_image(base64_string):
    """

    @param base64_string: base64字符串
    @return:
    """
    # 添加等号 '=' 进行填充
    while len(base64_string) % 4 != 0:
        base64_string += '='
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    model.eval()
    tsv_file_path = r'C:\Static\压缩包\MUGE\valid_imgs.tsv'
    store_file_path = "./data/test.jsonl"

    with open(tsv_file_path, 'r', newline='', encoding='utf-8') as file:
        # 使用制表符分隔符创建 CSV 读取器
        reader = csv.reader(file, delimiter='\t')
        # 获取文件中的所有行
        base64_images = list(reader)
    vectors = []
    with open(store_file_path, "w") as fout:
        for item in tqdm(base64_images):
            image_id = item[0]
            image = preprocess(decode_base64_image(item[1])).unsqueeze(0).to(device)
            image_feature = model.encode_image(image)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature.tolist()})))
