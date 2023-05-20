"""
dataディレクトリの各splitにあるjsons.pklからhigh_descを取り出す。
"""
import json
import pickle
import os


def main():
    data_dir = "/home/initial/workspase/CVPR/DialFRED-Challenge/data/extract_jsons"
    split = "pseudo_valid"
    data_path = os.path.join(data_dir, f"high_descs_{split}.json")
    # print(data_path)
    # with open(data_path, "r") as f:
    #     data = json.load(f)
    
    # result_txt = ""
    # count = 0
    # for key, value in data.items():
    #     for val in value:
    #         result_txt += val + "\n"
    #         count += 1


    # with open(os.path.join(data_dir, f"high_descs_{split}.txt"), "w") as f:
    #     json.dump(result_txt, f)

    # JSONファイルを読み取り、データを変数に格納する
    with open(data_path, 'r') as f:
        data = json.load(f)

    # テキストファイルに書き出す
    with open(os.path.join(data_dir, f"high_descs_{split}.txt"), 'w') as f:
        # 配列の各要素に対して処理を行う
        for key in data:
            # 各要素を改行区切りでテキストファイルに書き込む
            f.write('\n'.join(data[key]) + '\n')


if __name__ == "__main__":
    main()