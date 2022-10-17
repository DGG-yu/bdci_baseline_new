from model import GRTE
from transformers import WEIGHTS_NAME
from bert4keras.tokenizers import Tokenizer
from util import *
import os
import torch
from transformers import BertConfig
import json
import argparse

def predict():
    # 拼接地址
    # output_path=output
    output_path = os.path.join(args.output_path)
    # test_path_path=bdci/test.json
    test_path = os.path.join(args.base_path, args.dataset, "test.json")
    # rel2id_path=bdci/rel2id.json
    rel2id_path = os.path.join(args.base_path, args.dataset, "rel2id.json")
    # test_pred_path=bdci/test_pred.json
    test_pred_path = os.path.join(output_path, "test_pred.json")

    label_list=["N/A","SMH","SMT","SS","MMH","MMT","MSH","MST"]
    id2label,label2id={},{}
    for i,l in enumerate(label_list):
        id2label[str(i)]=l
        label2id[l]=i
        batch_ex = label2id[-1]
        print(batch_ex)

    test_data = json.load(open(test_path, 'r', encoding= 'utf-8'))
    id2predicate, predicate2id = json.load(open(rel2id_path, 'r', encoding='utf-8'))

    tokenizer = Tokenizer(args.bert_vocab_path)
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p = len(id2predicate)
    config.num_label = len(label_list)
    config.rounds = args.rounds
    config.fix_bert_embeddings = args.fix_bert_embeddings

    print(label2id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--cuda_id', default="1", type=str)
    parser.add_argument('--dataset', default='bdci', type=str)
    parser.add_argument('--rounds', default=4, type=int)

    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--fix_bert_embeddings', default=False, type=bool)
    parser.add_argument('--bert_vocab_path', default="./pretrain_models/vocab.txt", type=str)
    parser.add_argument('--bert_config_path', default="./pretrain_models/config.json", type=str)
    parser.add_argument('--bert_model_path', default="./pretrain_models/pytorch_model.bin", type=str)
    parser.add_argument('--max_len', default=200, type=int)
    parser.add_argument('--base_path', default="./dataset", type=str)
    parser.add_argument('--output_path', default="output", type=str)

    args = parser.parse_args()

    predict()