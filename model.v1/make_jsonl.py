import json
import csv
import os
from tqdm import tqdm  # 导入tqdm用于显示进度条
import re
from utils import *
import traceback


def process_json_files(csv_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    json_file = open(os.path.join(output_dir, 'output1.jsonl'),
                     'w',
                     encoding='utf-8')
    try:
        # 读取CSV文件
        with open(csv_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # 跳过标题行

            # 使用tqdm包装csv_reader以显示进度条
            for row in tqdm(csv_reader,
                            desc="Processing JSON files",
                            unit="file"):
                json_path = row[0]  # 获取JSON文件路径
                # print('row', row)
                try:
                    # 读取JSON文件
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    img_path = row[1]
                    shape = cv2.imread(img_path).shape
                    #element -> tuple: (word_text, word_bbox, normed_word_bbox)
                    #  resize_box(text_['src_word_bboxes'][i],shape)
                    doc_triplet = []
                    doc_tgt_sen_trans = []
                    doc_words_boxes_list = []
                    # 处理JSON数据
                    for key, value in json_data.items():
                        if value.get("attribute") == 'text_block':
                            for text_ in value.get('text', []):
                                combined_list = [(
                                    text_['src_words'][i],
                                    text_['src_word_bboxes'][i],
                                ) for i in range(len(text_['src_words']))]
                                doc_words_boxes_list.extend(combined_list)
                                # print(f'combined_list:{combined_list}')
                                doc_tgt_sen_trans.append(
                                    text_['tgt_text.zh-CN'])
                    processed_list = [
                        (src_w, src_w_boxes, resize_box(src_w_boxes, shape))
                        for (src_w, src_w_boxes) in doc_words_boxes_list
                    ]
                    # print(f'processed:{processed_list}')
                    sorted_tuple_list = tblr_reading_order_detector(
                        processed_list)

                    text_src_list = [atuple[0] for atuple in sorted_tuple_list]
                    layout_src_list = [
                        atuple[2] for atuple in sorted_tuple_list
                    ]
                    text_src = ' '.join(text_src_list)
                    tgt_sen_trans = ''.join(doc_tgt_sen_trans)
                    # print('text_src', text_src)
                    data_dict = {
                        "img_path": img_path,
                        "text_src": text_src,
                        "layout_src": layout_src_list,
                        "tgt_sen_trans": tgt_sen_trans
                    }
                    # print(data_dict)
                    json_line = json.dumps(data_dict, ensure_ascii=False)
                    json_file.write(json_line + '\n')

                except FileNotFoundError:
                    print(f"File not found: {json_path}")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {json_path}")
                except KeyError as e:
                    print(f"Missing key {e} in file: {json_path}")
                except Exception as e:
                    print(f"Unexpected error processing {json_path}: {str(e)}")
                    traceback.print_exc()

    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")

    print("Processing completed!")


# csv_path = '/home/zychen/hwproject/my_modeling_phase_1/dataset/output_part2.csv'  # 替换为你的CSV文件路径
csv_path = '/home/zychen/hwproject/my_modeling_phase_1/dataset/output.csv'  # 替换为你的CSV文件路径
output_dir = '/home/zychen/hwproject/my_modeling_phase_1/dataset'  # 输出目录名

process_json_files(csv_path, output_dir)
