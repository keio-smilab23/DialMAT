import os
import subprocess
import argparse
import time
import json

def trainModel(epochs):
    # 20epochの学習を実行し、各epochごとにモデルを保存
    EXP_NAME = os.environ["EXP_NAME"]
    SUBGOAL = os.environ["SUBGOAL"]
    command = f"python -m alfred.model.train with exp.model=transformer exp.name=et_{EXP_NAME}_{SUBGOAL}_splited exp.data.train=lmdb_{EXP_NAME}_{SUBGOAL}_splited train.seed=1 train.valid=True train.epochs={epochs}"
    subprocess.run(command, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", dest="epochs", type=int, default=20)
    args = parser.parse_args()

    trainModel(args.epochs)
    #output_dirの設定(./logs/et_augmented_human_subgoal_splited/の中で一番先頭二桁の数値が大きいものを取得)
    output_dir = "./logs/et_augmented_human_subgoal_splited/"
    dirs = os.listdir(output_dir)
    dirs = [dir for dir in dirs if os.path.isdir(output_dir + dir)]
    dirs.sort()
    output_dir = output_dir + dirs[-1]
    print("output_dir:", output_dir)

    # time_ = 0
    # #output_dir/done.txtが作成されているか確認
    # assert os.path.exists(output_dir + "/done.txt"), "output_dir/done.txtが作成されていません"

    # #done.txtを削除
    # os.remove(output_dir + "/done.txt")

    #output_dir/result_of_all_epochs.jsonを取得し、一番SRの高いモデルのパスを取得
    result_json_path = output_dir + "/result_of_all_epochs.json"

    best_epoch = 0
    best_sr = 0
    with open(result_json_path, "r") as f:
        result_json = json.load(f)
        #全ての要素を取り出し、一番SRの高いepochを取得
        for epoch, value in result_json.items():
            if value["sr"] > best_sr:
                best_epoch = int(epoch)
                best_sr = value["sr"]
    #best_epochのモデルのパスを取得
    best_model_path = os.path.join(output_dir,"model_{:02d}.pth".format(best_epoch))
    print("best_model_path:", best_model_path)
    command = f"python /home/initial/workspase/CVPR/DialFRED-Challenge/train_eval_each_epoch.py --output_dir {output_dir} --epoch {str(epoch)} --performer-path {best_model_path} --mode test"
    subprocess.run(command, shell=True)
     
    print("process finished")
    

if __name__ == '__main__':
    main()