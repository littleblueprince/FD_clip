# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 10:32
# @Author  : blue
# @Description :

import math

import faiss
import jsonlines
import numpy as np
import torch

from cn_clip.clip import load_from_name


def construct_index(jsonl_file_path, model, dim, store_file_path):
    '''
    根据embedding建立index
    :param store_file_path: (str)
    :param model:
    :param dim:embedding的张量维度
    :return:
    '''
    # 加载embedding结果
    image_id_list = np.array([])
    image_base64_list = []
    with jsonlines.open(jsonl_file_path) as jsonl_reader:
        for line in jsonl_reader:
            image_id_list = np.append(image_id_list, int(line['image_id']))
            image_base64_list.append(np.array(line['feature'])[0])
    image_id_list = np.asarray(image_id_list.astype('int32'))
    # TODO:聚类的中心数量 需要设计一下来控制内存使用与检索速度
    nlist = int(2 * math.sqrt(len(image_id_list)))
    # 倒排索引以及内积做相似性度量
    # TODO:倒排文件索引的量化器更换？相似性度量更换?
    quantiser = faiss.IndexFlatIP(dim)
    image_base64_list = np.array(image_base64_list)
    index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    assert not index.is_trained
    index.train(image_base64_list)  # 在向量集上训练索引
    assert index.is_trained
    index.add_with_ids(image_base64_list, image_id_list)  # 向索引中添加embedding以及对应小文件id
    faiss.write_index(index, store_file_path)  # 存储索引


if __name__ == "__main__":
    store_file_path = "./data/index.index"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    model.eval()
    # jsonl_file  image_id-image_base64
    jsonl_file_path = './data/test.jsonl'
    # embedding后的向量的维度,需要根据模型变换而变换
    dim = 512
    # 向量查询，找出的相似度topk的k
    k = 5
    construct_index(jsonl_file_path, model, dim, store_file_path)
