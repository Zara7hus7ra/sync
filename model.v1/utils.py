import numpy as np
import cv2
import os
import json
from functools import cmp_to_key

def resize_box(box, ori_img_shape, nomarlized_img_shape=(1000, 1000, 3)):
    """
        box: [x0, y0, x1, y1],
        ori_img_shape: 形如(1560, 1103, 3)
    """

    height_ratio = nomarlized_img_shape[0] / ori_img_shape[0]
    width_ratio = nomarlized_img_shape[1] / ori_img_shape[1]

    x0, y0, x1, y1 = box
    norm_x0, norm_x1 = round(x0 * width_ratio), round(x1 * width_ratio)
    norm_y0, norm_y1 = round(y0 * height_ratio), round(y1 * height_ratio)

    return [norm_x0, norm_y0, norm_x1, norm_y1]

def tblr_reading_order_detector(tuple_list):
    """rule: top-to-bottom, left-to-right

        tuple: (word_text, word_bbox, normed_word_bbox)
        
        return: sorted_tuple_list
    """

    def sort_cmp_fn(word_box1, word_box2):
        """
            sorted function的排序的2个元素的比较准则。
            1. 比较box1和box2的y坐标，如果二者的高重合度达到了二者的50%，则位于同一行，否则位于不同行。
            2. 如果位于同一行，那么比较二者的x0，如果box1_x0 < box2_x0，则返回-1，表示box_1<box_2，否则返回0（表示相等）或者1（box1>box2）。
            3. 如果不位于同一行，那么比较二者的y0，如果box1_y0 < box2_y0，则返回-1，否则返回0或者1.
        """

        x0, y0, x1, y1 = word_box1[1]
        x0_, y0_, x1_, y1_ = word_box2[1]

        
        if y0 < y0_:
            return -1
        elif y0 > y0_:
            return 1
        elif y0 == y0_:
            if x0 <= x0_:
                return -1
            elif x0 > x0_:
                return 1

    sorted_tuple_list = sorted(tuple_list, key=cmp_to_key(sort_cmp_fn))
    # print(sorted_word_box_list)

    return sorted_tuple_list

