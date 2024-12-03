# basic imports
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# other external imports
import pandas as pd
import sacrebleu
# torch imports
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
# transformers imports
from tqdm import tqdm
from transformers import (BertConfig, BertTokenizer, EncoderDecoderConfig,
                          EncoderDecoderModel, LayoutLMv3Tokenizer, LiltConfig,
                          LiltModel, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          default_data_collator)

# internal imports



def prepare_tokenizer(src_tokenizer_dir, tgt_tokenizer_dir):
    src_tokenizer = LayoutLMv3Tokenizer.from_pretrained(src_tokenizer_dir)
    tgt_tokenizer = BertTokenizer.from_pretrained(tgt_tokenizer_dir)

    return src_tokenizer, tgt_tokenizer


def prepare_dataset_df(data_file):
    dataset_df = pd.read_json(data_file, lines=True)
    return dataset_df


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(device)
    checkpoints_dir = '/home/zychen/hwproject/my_modeling_phase_1/train.lr_0.0001.bsz_28.step_400000.layer_12-12'
    model = EncoderDecoderModel.from_pretrained(
        f"{checkpoints_dir}/checkpoint-64000").to(device)
    encoder_ckpt_dir = "/home/zychen/hwproject/my_modeling_phase_1/Tokenizer_PretrainedWeights/lilt-roberta-en-base"
    tgt_tokenizer_dir = "/home/zychen/hwproject/my_modeling_phase_1/Tokenizer_PretrainedWeights/bert-base-chinese-tokenizer"

    src_tokenizer, tgt_tokenizer = prepare_tokenizer(
        src_tokenizer_dir=encoder_ckpt_dir,
        tgt_tokenizer_dir=tgt_tokenizer_dir,
    )
    model.eval()

    dataset_dir = "/home/zychen/hwproject/my_modeling_phase_1/dataset"
    data_file = f"{dataset_dir}/merged.jsonl"
    dataset_df = prepare_dataset_df(data_file=data_file)[:5000]
    print(f"\nnum_instances: {len(dataset_df)}\n")
    from model_and_train import (MyDataset, prepare_dataset_df,
                                 prepare_tokenizer)

    my_dataset = MyDataset(
        df=dataset_df,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_src_length=512,
        max_target_length=512,
    )

    dataloader = DataLoader(my_dataset, batch_size=4, shuffle=False)

    references = []
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].tolist()
            outputs = model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     do_sample=True,
                                     max_length=512,
                                     num_beams=1,
                                     use_cache=True,
                                     length_penalty=1.0,
                                     bos_token_id=0)

            decoded_preds = tgt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True)
            decoded_labels = tgt_tokenizer.batch_decode(
                labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend([label.split(' ') for label in decoded_labels])

            predictions_str = ''.join(predictions)
            references_str = ''.join([''.join(ref) for ref in references])

            print(predictions_str, references_str)

    bleu_score = sacrebleu.corpus_bleu(predictions, [references])
    print(f"BLEU score: {bleu_score.score}")
