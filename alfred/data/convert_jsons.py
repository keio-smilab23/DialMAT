import json
import os
import pickle
from alfred.gen import constants


def main():
    for split in ["train", "valid_seen", "pseudo_valid", "pseudo_test"]:
        #jsons_pklを読み込む. 
        jsons_path = os.path.join(constants.ET_DATA, "lmdb_augmented_human_subgoal", split, "jsons.pkl")
        with open(jsons_path, "rb") as f:
            jsons_list = pickle.load(f)
        

        # print("jsons_list[0].keys()", jsons_list[0].keys())
        c = 0
        for key, value in jsons_list.items():
            c += 1
            if c == 1:
                print(value.keys())
        #     break
        





if __name__ == '__main__':
    main()





















