from model_and_train import MyDataset, prepare_tokenizer, prepare_dataset_df
from torch.utils.data import Dataset, DataLoader
import json

dataset_dir = "/home/zychen/hwproject/my_modeling_phase_1/dataset"
data_file = f"{dataset_dir}/testset_10k.jsonl"
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
    # my_dataloader = DataLoader(
    #     my_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    # )
    img_name_list = dataset_df["img_path"].iloc[0:num_test].tolist()
    text_src_list = dataset_df["text_src"].iloc[0:num_test].tolist()
    with open('./mytest/text_src.jsonl', "w") as decoding_res_file:
        for img_name, text_src in zip(img_name_list, text_src_list):
            res_dict = {
                "img_name": img_name,
                "text_src": text_src,
            }

            record = f"{json.dumps(res_dict, ensure_ascii=False)}\n"
            decoding_res_file.write(record)
