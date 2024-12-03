# basic imports
import json
import os

import jieba
# other external imports
import pandas as pd
# torch imports
import torch
# internal imports
from model_and_train import MyDataset, prepare_dataset_df, prepare_tokenizer
from nltk.translate.bleu_score import sentence_bleu
from sacrebleu.metrics import CHRF
from torch.utils.data import DataLoader
from tqdm import tqdm
# transformers imports
from transformers import BertTokenizer, EncoderDecoderModel

chrf = CHRF(word_order=2)  # word_order=2 to be chrf++.

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# hyper-parameters.
## for model.
MAX_TGT_LEN = 512
MAX_SRC_LEN = 512

## for decoding.
output_dir = "./mytest"
os.makedirs(output_dir, exist_ok=True)
early_stopping = True
num_beams = 2
length_penalty = 1.0
batch_size = 16
metric_res_filepath = os.path.join(output_dir, "metric_res.json")
decoding_res_filepath = os.path.join(output_dir, "decoding_res.json")
trained_model_dir = "/home/zychen/hwproject/my_modeling_phase_1/train.lr_0.0001.bsz_28.step_400000.layer_12-12/checkpoint-64000"

dataset_dir = "/home/zychen/hwproject/my_modeling_phase_1/dataset"
data_file = f"{dataset_dir}/testset_10k.jsonl"


def no_blank(sen):
    return "".join(sen.split())


if __name__ == "__main__":

    encoder_ckpt_dir = "/home/zychen/hwproject/my_modeling_phase_1/Tokenizer_PretrainedWeights/lilt-roberta-en-base"

    tgt_tokenizer_dir = "/home/zychen/hwproject/my_modeling_phase_1/Tokenizer_PretrainedWeights/bert-base-chinese-tokenizer"

    src_tokenizer, tgt_tokenizer = prepare_tokenizer(
        src_tokenizer_dir=encoder_ckpt_dir,
        tgt_tokenizer_dir=tgt_tokenizer_dir,
    )
    dataset_df = prepare_dataset_df(data_file=data_file)
    my_dataset = MyDataset(df=dataset_df,
                           src_tokenizer=src_tokenizer,
                           tgt_tokenizer=tgt_tokenizer,
                           max_src_length=512,
                           max_target_length=512)
    print(len(my_dataset))
    from torch.utils.data import Subset
    num_test = 5000  #total 10k
    my_dataset = Subset(my_dataset, range(0, num_test))
    my_dataloader = DataLoader(
        my_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # loading model and config from pretrained folder
    model = EncoderDecoderModel.from_pretrained(trained_model_dir)
    # device='cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(model)

    # decoding testset
    pred_res_list = []
    gt_list = []

    for batch in tqdm(my_dataloader):
        # predict use generate
        with torch.no_grad():
            encoder_outputs = model.encoder(
                input_ids=batch["input_ids"].to(device),
                bbox=batch["bbox"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                encoder_outputs=encoder_outputs,
                max_length=MAX_TGT_LEN,
                early_stopping=early_stopping,
                num_beams=num_beams,
                length_penalty=length_penalty,
                use_cache=True,
                decoder_start_token_id=0)

        # decode
        pred_str = tgt_tokenizer.batch_decode(outputs,
                                              skip_special_tokens=True)
        labels = batch["labels"]
        labels[labels == -100] = tgt_tokenizer.pad_token_id
        label_str = tgt_tokenizer.batch_decode(labels,
                                               skip_special_tokens=True)

        pred_res_list += pred_str
        gt_list += label_str

    gt_list = [no_blank(sen) for sen in gt_list]
    pred_res_list = [no_blank(sen) for sen in pred_res_list]

    # write the decoding res and compute metric.
    img_name_list = dataset_df["img_path"].iloc[0:num_test].tolist()
    text_src_list = dataset_df["text_src"].iloc[0:num_test].tolist()
    bleu_list = []
    chrf_list = []

    pred_res_seg_list = [" ".join(jieba.cut(item)) for item in pred_res_list]
    gt_seg_list = [" ".join(jieba.cut(item)) for item in gt_list]
    print(len(text_src_list), len(pred_res_seg_list), len(gt_seg_list))
    # print(img_name_list, pred_res_list, gt_seg_list)
    assert len(img_name_list) == len(pred_res_seg_list) == len(gt_seg_list)

    with open(decoding_res_filepath, "w") as decoding_res_file:
        for img_name, text_src, pred_res_seg, gt_seg in zip(
                img_name_list, text_src_list, pred_res_seg_list, gt_seg_list):

            instance_bleu = sentence_bleu([gt_seg.split()],
                                          pred_res_seg.split())
            bleu_list.append(instance_bleu)

            instance_chrf = chrf.sentence_score(
                hypothesis=pred_res_seg,
                references=[gt_seg],
            ).score
            chrf_list.append(instance_chrf)

            res_dict = {
                "img_name": img_name,
                "text_src": text_src,
                "instance_bleu": instance_bleu,
                "instance_chrf": instance_chrf,
                "trans_res_seg": pred_res_seg,
                "gt_seg": gt_seg,
            }

            record = f"{json.dumps(res_dict, ensure_ascii=False)}\n"
            decoding_res_file.write(record)

    trans_avg_bleu = sum(bleu_list) / len(bleu_list)
    trans_avg_chrf = sum(chrf_list) / len(chrf_list)
    with open(metric_res_filepath, "w") as metric_res_file:
        eval_res_dict = {
            "trans_avg_bleu": trans_avg_bleu,
            "trans_avg_chrf": trans_avg_chrf,
        }
        json.dump(eval_res_dict, metric_res_file, indent=4, ensure_ascii=False)
