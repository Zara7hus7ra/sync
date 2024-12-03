from datasets import load_dataset
import pandas as pd

# text_src_jsonl = '/home/zychen/hwproject/my_modeling_phase_1/mytest/text_src.jsonl'
# dataset = load_dataset("json", data_files=text_src_jsonl)["train"]
# print(f"Number of examples: {len(dataset)}")
# text_src_df = dataset.to_pandas()

# decoding_res = '/home/zychen/hwproject/my_modeling_phase_1/mytest_3600_test5k/decoding_res.json'
decoding_res = '/home/zychen/hwproject/my_modeling_phase_1/mytest_from56k+64k/decoding_res.json'
dataset2 = load_dataset("json", data_files=decoding_res)["train"]
print(f"Number of examples: {len(dataset2)}")
decoding_df = dataset2.to_pandas()

# df_merged = pd.concat([text_src_df, decoding_df], axis=1)

df_merged = decoding_df
print(df_merged.columns.tolist(), df_merged.iloc[4500])


def clean(sentence):
    return ''.join(sentence.split())


df = df_merged
with open('text_src.txt', 'w', encoding='utf-8') as f:
    for text in df['text_src']:
        # cleaned_text = clean(text)
        f.write(text + '\n')

# 将trans_res_seg列的内容写入hyp.txt
with open('hyp.txt', 'w', encoding='utf-8') as f:
    for text in df['trans_res_seg']:
        cleaned_text = clean(text)
        f.write(cleaned_text + '\n')

# 将gt_seg列的内容写入ref.txt
with open('ref.txt', 'w', encoding='utf-8') as f:
    for text in df['gt_seg']:
        cleaned_text = clean(text)
        f.write(cleaned_text + '\n')
