# basic imports
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# other external imports
import pandas as pd
# torch imports
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
# transformers imports
from transformers import (BertConfig, BertTokenizer, EncoderDecoderConfig,
                          EncoderDecoderModel, LayoutLMv3Tokenizer, LiltConfig,
                          LiltModel, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          default_data_collator)

# internal imports



# prepare tokenizer.
def prepare_tokenizer(src_tokenizer_dir, tgt_tokenizer_dir):
    src_tokenizer = LayoutLMv3Tokenizer.from_pretrained(src_tokenizer_dir)
    tgt_tokenizer = BertTokenizer.from_pretrained(tgt_tokenizer_dir)

    return src_tokenizer, tgt_tokenizer


# read data points.
def prepare_dataset_df(data_file):

    def filter_fn(exam):
        bboxes = exam["layout_src"]
        for box in bboxes:
            x0, y0, x1, y1 = box
            if (x0 > x1) or (y0 > y1):
                print("(x0 > x1) or (y0 > y1)")
                return False
            for cor in box:
                if cor < 0 or cor > 1000:
                    # print("cor < 0 or cor > 1000")
                    # print(exam['img_path'],box)
                    return False
        return True

    dataset = load_dataset("json", data_files=data_file)["train"]
    print()
    print(f"Number of examples: {len(dataset)}")
    print()

    dataset = dataset.filter(filter_fn, num_proc=48)

    dataset_df = dataset.to_pandas()
    # dataset_df = pd.read_json(data_file, lines=True, orient="records")

    # filter the nan data points.
    dataset_df = dataset_df[~dataset_df["tgt_sen_trans"].isna()]
    dataset_df = dataset_df[~dataset_df["text_src"].isna()]
    dataset_df = dataset_df[~dataset_df["layout_src"].isna()]
    # remove entries where "text_src" length is less than 3
    dataset_df = dataset_df[dataset_df["text_src"].str.len() >= 3]
    # reconstruct the idx to avoid index_error.
    dataset_df = dataset_df.reset_index(drop=True)

    print(f"Number of examples after filtered: {len(dataset_df)}")
    return dataset_df


class MyDataset(Dataset):

    def __init__(
        self,
        df,
        src_tokenizer,
        tgt_tokenizer,
        max_src_length,
        max_target_length,
    ):
        self.df = df
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_src_length = max_src_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get text_src + layout_src + tgt_trans.
        text_src = self.df['text_src'][idx]
        layout_src = self.df['layout_src'][idx]
        tgt_trans = self.df['tgt_sen_trans'][idx]

        # read in annotations at word-level (words, word boxes)
        words_ = text_src.split(" ")
        word_boxes_ = layout_src
        # print('words', words_, len(words_), len(word_boxes_))
        assert len(words_) == len(word_boxes_)
        words = []
        word_boxes = []
        for word, word_box in zip(words_, word_boxes_):
            if (word_box[0] >= word_box[2]) or (word_box[1] >= word_box[3]):
                continue

            words.append(word)
            word_boxes.append(word_box)

        assert len(words) == len(word_boxes)

        encoding = self.src_tokenizer(
            words,
            boxes=word_boxes,
            padding="max_length",
            truncation=True,
            max_length=self.max_src_length,
        )

        # construct labels.
        labels = self.tgt_tokenizer(
            tgt_trans,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length)["input_ids"]
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [
            label if label != self.tgt_tokenizer.pad_token_id else -100
            for label in labels
        ]

        encoding["labels"] = labels

        assert len(encoding['input_ids']) == self.max_src_length
        assert len(encoding['attention_mask']) == self.max_src_length
        assert len(encoding['bbox']) == self.max_src_length
        assert len(encoding['labels']) == self.max_target_length

        # finally, convert everything to PyTorch tensors
        for k, v in encoding.items():
            encoding[k] = torch.as_tensor(encoding[k])

        return encoding


def prepare_model(src_tokenizer,
                  tgt_tokenizer,
                  max_src_len,
                  max_tgt_len,
                  num_encoder_hidden_layers,
                  num_decoder_hidden_layers,
                  encoder_ckpt_dir,
                  model_ckpt_dir=None):
    config_encoder = LiltConfig.from_pretrained(
        encoder_ckpt_dir,
        max_position_embeddings=max_src_len + 2,
        num_hidden_layers=num_encoder_hidden_layers)
    config_decoder = BertConfig(vocab_size=tgt_tokenizer.vocab_size,
                                max_position_embeddings=max_tgt_len,
                                num_hidden_layers=num_decoder_hidden_layers)

    model_config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config=config_encoder,
        decoder_config=config_decoder,
    )
    model = EncoderDecoderModel(config=model_config, )

    model.config.decoder_start_token_id = tgt_tokenizer.cls_token_id
    model.config.pad_token_id = tgt_tokenizer.pad_token_id
    model.config.vocab_size = tgt_tokenizer.vocab_size
    model.config.eos_token_id = tgt_tokenizer.pad_token_id

    from safetensors.torch import load_file
    if model_ckpt_dir:
        bin_path = f"{model_ckpt_dir}/pytorch_model.bin"
        safetensors_path = f"{model_ckpt_dir}/model.safetensors"
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path)
        elif os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        else:
            raise FileNotFoundError(
                "Neither pytorch_model.bin nor model.safetensors found in the specified directory."
            )
        model.load_state_dict(state_dict, strict=False)
        model.save_pretrained(
            f"continued_{model_ckpt_dir}")  #save at continued training
    else:
        # Loading the pre-trained params and then save the model, including its configuration.
        tmp_encoder = LiltModel.from_pretrained(
            pretrained_model_name_or_path=encoder_ckpt_dir,
            config=config_encoder,
        )
        # tmp_encoder = LiltModel(config=config_encoder)
        model.encoder = tmp_encoder
        # model.save_pretrained("undertrained_default_safe_true")
        model.save_pretrained("undertrained_safe_serialization_False", safe_serialization=False)
        # model.load_state_dict(torch.load(f"undertrained/pytorch_model.bin"))

        bin_path = "undertrained_safe_serialization_False/pytorch_model.bin"
        safetensors_path = "undertrained_default_safe_true/model.safetensors"
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path)
        elif os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        else:
            raise FileNotFoundError(
                "Neither pytorch_model.bin nor model.safetensors found in the specified directory."
            )
        model.load_state_dict(state_dict, strict=False)

    print(model.config)
    print(model)

    return model


if __name__ == "__main__":

    # hyper-parameters.
    ## for model.
    MAX_TGT_LEN = 512
    MAX_SRC_LEN = 512
    num_encoder_hidden_layers = 12
    num_decoder_hidden_layers = 12

    ## for training.
    num_instances = 500000  #total 620082 ./dataset/merged.jsonl Number of examples after filtered: 547084
    learning_rate = 1e-4
    batch_size = 28
    num_train_steps = 400000 #400000
    output_dir = f"./train.lr_{learning_rate}.bsz_{batch_size}.step_{num_train_steps}.layer_{num_encoder_hidden_layers}-{num_decoder_hidden_layers}"
    save_total_limit = 100
    save_steps = num_train_steps // save_total_limit

    dataset_dir = "/home/zychen/hwproject/my_modeling_phase_1/dataset"
    data_file = f"{dataset_dir}/merged.jsonl"

    # model_ckpt_dir = '/home/zychen/hwproject/my_modeling_phase_1/train.lr_0.0001.bsz_8.step_400000.layer_12-12/checkpoint-32000'
    model_ckpt_dir = '/home/zychen/hwproject/my_modeling_phase_1/train.lr_0.0001.bsz_16.step_500000.layer_12-12_36k+20k/checkpoint-20000'
    encoder_ckpt_dir = "/home/zychen/hwproject/my_modeling_phase_1/Tokenizer_PretrainedWeights/lilt-roberta-en-base"

    tgt_tokenizer_dir = "/home/zychen/hwproject/my_modeling_phase_1/Tokenizer_PretrainedWeights/bert-base-chinese-tokenizer"

    src_tokenizer, tgt_tokenizer = prepare_tokenizer(
        src_tokenizer_dir=encoder_ckpt_dir,
        tgt_tokenizer_dir=tgt_tokenizer_dir,
    )
    dataset_df = prepare_dataset_df(data_file=data_file)[:num_instances]
    print(f"\nnum_instances: {len(dataset_df)}\n")
    print(dataset_df)
    my_dataset = MyDataset(
        df=dataset_df,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_src_length=MAX_SRC_LEN,
        max_target_length=MAX_TGT_LEN,
    )
    model = prepare_model(src_tokenizer=src_tokenizer,
                          tgt_tokenizer=tgt_tokenizer,
                          max_src_len=MAX_SRC_LEN,
                          max_tgt_len=MAX_TGT_LEN,
                          num_encoder_hidden_layers=num_encoder_hidden_layers,
                          num_decoder_hidden_layers=num_decoder_hidden_layers,
                          encoder_ckpt_dir=encoder_ckpt_dir,
                          model_ckpt_dir=model_ckpt_dir)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=False,
        evaluation_strategy="no",
        per_device_train_batch_size=batch_size,
        fp16=True,
        output_dir=output_dir,
        logging_steps=1,
        # save_strategy="epoch",
        learning_rate=learning_rate,
        max_steps=num_train_steps,
        warmup_ratio=0.05,
        save_total_limit=save_total_limit,
        save_steps=save_steps,
        save_safetensors=False,
    )
    # print(training_args)
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=None,
        train_dataset=my_dataset,
        eval_dataset=None,
        data_collator=default_data_collator,
    )

    trainer.train()
