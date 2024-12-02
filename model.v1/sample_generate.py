# basic imports
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# transformers imports
from transformers import LiltConfig, BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertTokenizer, LayoutLMv3Tokenizer, LiltModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from datasets import load_dataset

# torch imports
import torch
from torch.utils.data import Dataset, DataLoader

# internal imports

# other external imports
import pandas as pd


def prepare_tokenizer(src_tokenizer_dir, tgt_tokenizer_dir):
    src_tokenizer = LayoutLMv3Tokenizer.from_pretrained(src_tokenizer_dir)
    tgt_tokenizer = BertTokenizer.from_pretrained(tgt_tokenizer_dir)

    return src_tokenizer, tgt_tokenizer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(device)
    checkpoints_dir = '/home/zychen/hwproject/my_modeling_phase_1/train.lr_0.0001.bsz_8.step_400000.layer_12-12_36000'
    model = EncoderDecoderModel.from_pretrained(
        f"{checkpoints_dir}/checkpoint-36000").to(device)
    encoder_ckpt_dir = "/home/zychen/hwproject/my_modeling_phase_1/Tokenizer_PretrainedWeights/lilt-roberta-en-base"
    tgt_tokenizer_dir = "/home/zychen/hwproject/my_modeling_phase_1/Tokenizer_PretrainedWeights/bert-base-chinese-tokenizer"

    src_tokenizer, tgt_tokenizer = prepare_tokenizer(
        src_tokenizer_dir=encoder_ckpt_dir,
        tgt_tokenizer_dir=tgt_tokenizer_dir,
    )
    model.eval()

    from model_and_train import MyDataset, prepare_dataset_df, prepare_tokenizer

    dataset_dir = "/home/zychen/hwproject/my_modeling_phase_1/dataset"
    data_file = f"{dataset_dir}/merged.jsonl"
    dataset_df = prepare_dataset_df(data_file=data_file)[:1000]
    print(f"\nnum_instances: {len(dataset_df)}\n")
    print(dataset_df)
    my_dataset = MyDataset(
        df=dataset_df,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_src_length=512,
        max_target_length=512,
    )
    sample = my_dataset[0]
    from transformers import GenerationConfig
    generation_config = GenerationConfig(
        max_length=512,
        early_stopping=True,
        num_beams=1,
        use_cache=True,
        length_penalty=1.0,
    )

    with torch.no_grad():
        generation_config = None
        outputs = model.generate(
            input_ids=sample['input_ids'].unsqueeze(
                0),  # 添加 unsqueeze 以增加 batch 维度
            attention_mask=sample['attention_mask'].unsqueeze(0),
            do_sample=False,
            generation_config=generation_config,
            bos_token_id=0)
        decoded_preds = tgt_tokenizer.batch_decode(outputs,
                                                   skip_special_tokens=True)
        print(decoded_preds)
        print(sample['labels'])
