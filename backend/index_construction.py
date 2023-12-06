# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 10:32
# @Author  : blue
# @Description :

import math
import faiss
import jsonlines
import numpy as np

def construct_index(jsonl_file_path, dim, store_file_path, index_type='IVFFlat', metric='L2'):
    '''
    根据embedding建立index
    :param jsonl_file_path: (str) JSONL文件路径，包含图像ID和特征
    :param dim: (int) 特征向量的维度
    :param store_file_path: (str) 索引文件存储路径
    :param index_type: (str) 索引类型 ('Flat', 'IVFFlat', 'HNSW', 'PQ', 'IVFPQ')
    :param metric: (str) 相似度度量方法 ('L2', 'IP')
    :return:
    '''
    # 加载embedding结果
    image_id_list = np.array([])
    image_base64_list = []
    with jsonlines.open(jsonl_file_path) as jsonl_reader:
        for line in jsonl_reader:
            image_id_list = np.append(image_id_list, int(line['image_id']))
            image_base64_list.append(np.array(line['feature'])[0])
    image_base64_list = np.array(image_base64_list)
    image_id_list = image_id_list.astype('int32')

    # 设置相似度度量方法
    if metric == 'L2':
        metric = faiss.METRIC_L2
    elif metric == 'IP':
        metric = faiss.METRIC_INNER_PRODUCT
    else:
        raise ValueError("Unsupported metric")

    # 根据索引类型构建索引
    if index_type == 'Flat':
        if metric == faiss.METRIC_L2:
            index = faiss.IndexFlatL2(dim)
        else:
            index = faiss.IndexFlatIP(dim)

    elif index_type == 'IVFFlat':
        nlist = int(2 * math.sqrt(len(image_id_list)))
        quantiser = faiss.IndexFlat(dim, metric)
        index = faiss.IndexIVFFlat(quantiser, dim, nlist, metric)

    elif index_type == 'HNSW':
        index = faiss.IndexHNSWFlat(dim, 32)

    elif index_type == 'PQ':
        m = 8  # 子向量数量，需要根据dim调整
        index = faiss.IndexPQ(dim, m, 8)

    elif index_type == 'IVFPQ':
        nlist = int(2 * math.sqrt(len(image_id_list)))
        m = 8  # 子向量数量
        quantiser = faiss.IndexFlat(dim, metric)
        index = faiss.IndexIVFPQ(quantiser, dim, nlist, m, 8)

    else:
        raise ValueError("Unsupported index type")

    # 训练索引（如果需要）并添加数据
    if 'IVF' in index_type or index_type == 'PQ':
        if not index.is_trained:
            index.train(image_base64_list)  # 训练索引
            assert index.is_trained  # 确保索引已经被训练

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
    index_type = 'IVFFlat'  # 更改为所需的索引类型
    metric = 'L2'  # 更改为所需的相似度度量方法
    construct_index(jsonl_file_path, dim, store_file_path, index_type, metric)
