#coding=utf-8
import json
import os
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from text_manipulation import extract_sentence_words
from wiki_loader import get_sections
import wiki_utils
result_path = "checkpoints/infer_result"
split_dict_path = "checkpoints/infer_result/split_by_pk.json"
range_lst = ["0.9-1.0","0.8-0.9","0.7-0.8","0.6-0.7","0.5-0.6", "0.4-0.5", "0.3-0.4", "0.2-0.3", "0.1-0.2","0.0-0.1"]
"""
1. 找到pk>=0.5,0.5-0.4(=), 0.4-0.3(=), 0.3-0.2(=), <0.2的样本数、具体样本，分成10个字典保存
2. 写一个函数:
    a.参数是路径名
    b.研究wiki,中间产物是什么
    c.把pred\gold 转化成(.split) 列表
    d.根据中间产物(处理后的文章), 用上面的列表, 可视化结果
"""
def split_by_pk(result_path = result_path):
    split_dict = {}
    for rg in range_lst:
        split_dict[rg] = []
    result_path = os.path.join(result_path, "infer_result.json")
    with open(result_path, "rb") as f:
        result_dict = json.load(f)
    paths = list(result_dict.keys())
    for path in paths:
        pk = float(result_dict[path]["pk"])
        if pk >= 0.9:
            split_dict["0.9-1.0"].append(path)
        elif 0.8 <= pk < 0.9:
            split_dict["0.8-0.9"].append(path)
        elif 0.7 <= pk < 0.8:
            split_dict["0.7-0.8"].append(path)
        elif 0.6 <= pk < 0.7:
            split_dict["0.6-0.7"].append(path)
        elif 0.5 <= pk < 0.6:
            split_dict["0.5-0.6"].append(path)
        elif 0.4 <= pk < 0.5:
            split_dict["0.4-0.5"].append(path)
        elif 0.3 <= pk < 0.4:
            split_dict["0.3-0.4"].append(path)
        elif 0.2 <= pk < 0.3:
            split_dict["0.2-0.3"].append(path)
        elif 0.1 <= pk < 0.2:
            split_dict["0.1-0.2"].append(path)
        else:
            split_dict["0.0-0.1"].append(path)
    with open("checkpoints/infer_result/split_by_pk.json", "w") as f:
        json.dump(split_dict,f)
# split_by_pk()

def count_num(split_dict_path = split_dict_path, visible = True):
    with open(split_dict_path, "rb") as f:
        split_result = json.load(f)
    total_len = 0
    length_lst = []
    for i, rg in enumerate(range_lst):
        length = len(split_result[rg])
        length_lst.append(length)
        total_len += length
        print(rg+":")
        print(length)
    print("total length:",total_len)
    if visible:
        x = [i for i in range(10)]
        y = length_lst
        plt.bar(x, y, facecolor="blue", edgecolor="white")
        for x1, y1 in zip(x, y):
            plt.text(x1, y1, '%.3f' % y1, ha="center", va="bottom", fontsize=7)
        new_xticks = [r"0.9-1.0",r"0.8-0.9",r"0.7-0.8",r"0.6-0.7",r"0.5-0.6", r"0.4-0.5", r"0.3-0.4", r"0.2-0.3", r"0.1-0.2",r"0.0-0.1"]
        plt.xticks(x, new_xticks)
        plt.savefig(os.path.join(path, "count.jpg"))

def show_analysis(document_path,result_dict):
    result = result_dict[document_path]

    pk = result["pk"]
    wd = result["WD"]
    b = result["B"]
    s = result["S"]
    pred = result["pred"]
    golden = result["golden"]

    print("="*3+document_path+" analyzing"+"="*3)
    print("pk值:",pk)
    print("wd值:", wd)
    print("b值:", b)
    print("s值:", s)
    print("pred_seg:")
    print(pred)
    print("golden_seg:")
    print(golden)
    visualize_document(document_path,pred,golden)
    print("如果要查看pred和golden具体的分割效果,请自行访问路径:")
    print("/"+"/".join(document_path.strip("/").split("/")[:-1]))
    print()

def analysis_document(result_path = result_path, split_dict_path = split_dict_path):
    result_path = os.path.join(result_path, "infer_result.json")
    with open(result_path, "rb") as f:
        result_dict = json.load(f)
    rg_choose = input("你想要分析哪一pk范围文章?(0:{0.9-1.0}, 1:{0.8-0.9}, 2:{0.7-0.8}, 3:{0.6-0.7}, 4:{0.5-0.6}, 5:{0.4-0.5}, 6:{0.3-0.4}, 7:{0.2-0.3}, 8:{0.1-0.2}, 9:{0.0-0.1}")
    rg_choose = range_lst[int(rg_choose)]
    with open(split_dict_path, "rb") as f:
        split_result = json.load(f)
    document_path_lst = split_result[rg_choose]
    cot = 0
    while True:
        print("pk值为{}范围内的文章一共有{}篇".format(rg_choose, len(document_path_lst)))
        go_on = input("你要分析第{}篇吗?(y/Enter:要; n:退出分析; 数字:指定哪一篇)".format(cot+1))
        if go_on == "y" or go_on == "":
            if cot < len(document_path_lst):
                show_analysis(document_path_lst[cot], result_dict)
                cot += 1
            else:
                print("不存在,请重新选择")
        elif go_on == "n":
            break
        elif go_on.isdigit():
            if int(go_on) <= len(document_path_lst):
                show_analysis(document_path_lst[int(go_on)-1], result_dict)
                cot = int(go_on)
            else:
                print("不存在,请重新选择")
        else:
            print("不存在,请重新选择")
# visualize_document(document_path,pred,golden)
def visualize_document(path,
                       pred,
                       golden,
                       remove_preface_segment=True,
                       ignore_list=True,
                       remove_special_tokens=True,
                       return_as_sentences=False,
                       high_granularity=False):
    dir_path = "/"+"/".join(path.strip("/").split("/")[:-1])
    # if os.path.exists(os.path.join(dir_path,"pred")) and os.path.exists(os.path.join(dir_path,"golden")):
    #     return
    all_sections = get_sections(path, high_granularity)
    required_sections = all_sections[1:] if remove_preface_segment and len(all_sections) > 0 else all_sections
    required_non_empty_sections = [section for section in required_sections if len(section) > 0 and section != "\n"]
    final_sentence_lst = []
    for section in required_non_empty_sections:
        sentences = section.split('\n')
        if sentences:
            for sentence in sentences:
                is_list_sentence = wiki_utils.get_list_token() + "." == str(sentence)
                if ignore_list and is_list_sentence:
                    continue
                if not return_as_sentences:
                    sentence_words = extract_sentence_words(sentence, remove_special_tokens=remove_special_tokens)
                    if 1 > len(sentence_words):
                        continue
                    else:
                        final_sentence_lst.append(sentence)
    def write_document(sentences, path, seg, name):
        with open(os.path.join(path,name), "w") as f:
            seg_list = seg.split(" ")
            cot = 0
            for i in seg_list:
                if cot == 0:
                    f.write("="*10+"开头"+"\n")
                    cot = 1
                else:
                    if i == "|": # 分割点
                        f.write("="*10+"\n")
                    else:
                        f.write(sentences[int(i)-1]+"\n")
    write_document(final_sentence_lst, dir_path, pred, "pred")
    write_document(final_sentence_lst, dir_path, golden, "golden")
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--choose', help='[split(按照pk值分割结果)/count(统计每个类别的个数)/analysis(分析单独一篇文章)]', type=str)
    parser = parser.parse_args()
    choose = parser.choose
    if choose == "split":
        split_by_pk()
    elif choose == "count":
        count_num()
    elif choose == "analysis":
        analysis_document()