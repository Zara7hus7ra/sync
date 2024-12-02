from datasets import load_dataset
import pandas as pd

decoding_res = '/home/zychen/hwproject/my_modeling_phase_1/mytest_3600_test5k/decoding_res.json'
dataset2 = load_dataset("json", data_files=decoding_res)["train"]
print(f"Number of examples: {len(dataset2)}")
decoding_df = dataset2.to_pandas()

decoding_df