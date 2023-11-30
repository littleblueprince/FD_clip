# -*- coding: utf-8 -*-
# @Time    : 2023/11/21 10:37
# @Author  : blue
# @Description :


import base64
import csv
from io import BytesIO

import faiss
import torch
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name


def get_topK(query, index, model, k):
    """
    将query向量化并且搜索返回topk个id
    :param query:
    :param index:
    :param model:
    :param k:
    :return:
    """
    text = clip.tokenize(query).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        # 对特征进行归一化使用归一化后的图文特征用于下游任务torch.Size([1, 512])
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu()
    _, I = index.search(text_features, k)
    # 返回的I是近邻的ID集合
    return I.tolist()


def query_index(store_path, query, model, k):
    '''
    读取index然后调用get_topK接口对于query内容进行搜索返回id
    并且编写合适的prompt输出
    :param store_path:
    :param query:
    :param model:
    :param k: top_k
    :return:
    '''
    index = faiss.read_index(store_path)
    # 搜索
    I = get_topK(query, index, model, k)
    ret = []
    # I[0]是所有相关的小文件对应的id的list，按照相关度排序，-1表示不存在
    for i in I[0]:
        ret.append(str(i))
    # 输出相关信息
    return ret


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    model.eval()

    # index数据的路径
    index_file_path = './data/index.index'
    dim = 512
    # 向量查询，找出的相似度topk的k
    k = 5
    # 根据索引query
    query = ["水晶鞋"]

    related_texts = query_index(index_file_path, query, model, k)
    for related_text in related_texts:
        print(related_text)

    # 指定的 TSV 文件路径
    tsv_file_path = 'C:\Static\压缩包\MUGE\\valid_imgs.tsv'

    # 读取 TSV 文件
    with open(tsv_file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        all_rows = list(reader)
        for row in all_rows:
            if row[0] in related_texts:
                print(row[0])
                image_data = base64.b64decode(row[1])
                # 使用 Pillow 加载图像
                image = Image.open(BytesIO(image_data))
                # 展示图像
                image.show()
                print('请输入:')
                input()
