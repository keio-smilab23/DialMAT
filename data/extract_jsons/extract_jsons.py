"""
dataディレクトリの各splitにあるjsons.pklからhigh_descを取り出す。
"""
import json
import pickle
import os


def main():
    data_dir = "/home/initial/workspase/CVPR/DialFRED-Challenge/data/lmdb_augmented_human_subgoal_splited"
    split = "pseudo_test"
    data_path = os.path.join(data_dir, split, "jsons.pkl")
    print(data_path)
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    count = 0 
    result_dict = {}
    for key, value in data.items():
        # for v in value:
        v = value[0]
        # print(len(value)) #22

        #v["turk_annotations"]["anns"][0]["high_descs"]の2つの要素の文字列において、<<が出てくる前の部分を取得.
        # res = [t.split("<<")[0] for t in v["turk_annotations"]["anns"][0]["high_descs"]]
        res = v["scene"]["floor_plan"]
        
        # result_dict[str(count)] = v["task_id"]
        # result_dict[str(count)] = v["turk_annotations"]["anns"][0]["high_descs"]
        result_dict[str(count)] = {"floor_plan" : v["scene"]["floor_plan"], "task_id":v["task_id"], "high_descs":v["turk_annotations"]["anns"][0]["high_descs"], "task_type":v["task_type"]}
        count += 1
        
    print("total_count", count)           

    with open(f"/home/initial/workspase/CVPR/DialFRED-Challenge/data/extract_jsons/task_id_{split}.json", "w") as f:
        json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    main()