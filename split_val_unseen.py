import shutil
import glob
import os


data_dir = os.path.join(os.environ['DATA'], 'generated_2.1.0')
#processed.txtを読み込む
with open(os.path.join(data_dir, 'processed.txt'), 'r') as f:
    lines = f.readlines()

result = []
pseudo_valid_set = set()
pseudo_test_set = set()
for index, line in enumerate(lines):
    if index < 138:
        pseudo_valid_set.add(line.split('/')[1])
        #各lineの先頭のvalid_unseenをpseudo_validに変更
        result.append(line.replace('valid_unseen', 'pseudo_valid',1))
    elif index < 255:
        pseudo_test_set.add(line.split('/')[1])
        #各lineの先頭のvalid_unseenをpseudo_testに変更
        result.append(line.replace('valid_unseen', 'pseudo_test',1))
    else:
        result.append(line)

#./valid_unseen以下のディレクトリ一覧を取得
dirs = glob.glob(os.path.join(data_dir, 'valid_unseen/*'))

#./pseudo_validと./pseudo_testを作成
if os.path.exists(os.path.join(data_dir, 'pseudo_valid')):
    shutil.rmtree(os.path.join(data_dir, 'pseudo_valid'))
if os.path.exists(os.path.join(data_dir, 'pseudo_test')):
    shutil.rmtree(os.path.join(data_dir, 'pseudo_test'))
os.mkdir(os.path.join(data_dir, 'pseudo_valid'))
os.mkdir(os.path.join(data_dir, 'pseudo_test'))


for d in dirs:
    if d.split('/')[-1] in pseudo_valid_set:
        shutil.copytree(d, os.path.join(data_dir, 'pseudo_valid',d.split('/')[-1]))
    elif d.split('/')[-1] in pseudo_test_set:
        shutil.copytree(d, os.path.join(data_dir, 'pseudo_test' ,d.split('/')[-1]))

#processed_new.txtに書き込む
with open(os.path.join(data_dir, 'processed_new.txt'), 'w') as f:
    f.writelines(result)


