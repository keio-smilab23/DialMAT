"""
floor_plan_{split}.jsonを読み込み、FloorPlanごとに出現頻度を棒グラフとして表示する.
"""

import matplotlib.pyplot as plt
import json
import numpy as np



def main():
    #floor_plan_{split}.jsonを読み込む
    split = "pseudo_test"
    with open(f"/home/initial/workspase/CVPR/DialFRED-Challenge/data/extract_jsons/floor_plan_{split}.json", "r") as f:
        data = json.load(f)
    
    #FloorPlanごとに出現頻度をカウントする
    if split != "test":
        count_dict = {}
        for key, value in data.items():
            if value not in count_dict:
                count_dict[value] = 1
            else:
                count_dict[value] += 1
    else:
        count_dict = {}
        for key, value in data.items():
            if value["floor_plan"] in count_dict.keys():
                count_dict[value["floor_plan"]] += 1
            else:
                count_dict[value["floor_plan"]] = 1

    #出現頻度を棒グラフとして表示する
    #count_dictのkeyの先頭のFloorPlanという文字を取り除き、int型に変換する
    count_dict = {int(k[9:]):v for k, v in count_dict.items()}
    #x軸はFloorPlanの番号、y軸は出現頻度. ただ、x軸は1から500までの連番ではないので、x軸の値をソートする必要がある.

    count_dict = dict(sorted(count_dict.items(), key=lambda x:x[0]))
    x = np.arange(500)
    #yの値の作成
    y = np.zeros(500)
    for k, v in count_dict.items():
        y[k-1] = v
    # x = np.array(list(count_dict.keys()))
    # y = np.array(list(count_dict.values()))
    print(x, y)
    #棒グラフの棒を太くする
    # plt.bar(x, y)
    plt.ylim(0, 100)
    plt.bar(x, y, width=1.0, align="center")

    #グラフのタイトル、x軸、y軸のラベルを表示
    plt.xlabel("FloorPlan")
    plt.ylabel("Frequency")
    plt.show()
    #図の保存
    plt.savefig(f"/home/initial/workspase/CVPR/DialFRED-Challenge/data/extract_jsons/floor_plan_{split}.png")


    








if __name__ == "__main__":
    main()


