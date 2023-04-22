import shutil
import glob
import os


data_dir = os.path.join(os.environ['DATA'], 'generated_2.1.0')
#processed.txtを読み込む
with open(os.path.join(data_dir, 'processed_original.txt'), 'r') as f:
    lines = f.readlines()

result = []
pseudo_valid_list = []
pseudo_test_list = []

result_pseudo_valid = []
result_pseudo_test = []
result_others = []
for index, line in enumerate(lines):
    if index < 255:
        if index % 2 == 0:
            pseudo_valid_list.append(line.split('/')[1] + '/' + line.split('/')[2])
            #各lineの先頭のvalid_unseenをpseudo_validに変更
            result_pseudo_valid.append(line.replace('valid_unseen', 'pseudo_valid',1))
        else:
            pseudo_test_list.append(line.split('/')[1] + '/' + line.split('/')[2])
            #各lineの先頭のvalid_unseenをpseudo_testに変更
            result_pseudo_test.append(line.replace('valid_unseen', 'pseudo_test',1))
    else:
        result_others.append(line)

result = result_pseudo_valid + result_pseudo_test + result_others

#./valid_unseen以下のディレクトリ一覧を取得()
dirs = glob.glob(os.path.join(data_dir, 'valid_unseen/*/*'))

#./pseudo_validと./pseudo_testを作成
if os.path.exists(os.path.join(data_dir, 'pseudo_valid')):
    shutil.rmtree(os.path.join(data_dir, 'pseudo_valid'))
if os.path.exists(os.path.join(data_dir, 'pseudo_test')):
    shutil.rmtree(os.path.join(data_dir, 'pseudo_test'))
os.mkdir(os.path.join(data_dir, 'pseudo_valid'))
os.mkdir(os.path.join(data_dir, 'pseudo_test'))


for d in dirs:
    if d.split('/')[-2] + '/' + d.split('/')[-1] in pseudo_valid_list:
        if os.path.exists(os.path.join(data_dir, 'pseudo_valid', d.split('/')[-2])):
            shutil.copytree(d, os.path.join(data_dir, 'pseudo_valid', d.split('/')[-2], d.split('/')[-1]))
        else:
            os.mkdir(os.path.join(data_dir, 'pseudo_valid', d.split('/')[-2]))
            shutil.copytree(d, os.path.join(data_dir, 'pseudo_valid', d.split('/')[-2], d.split('/')[-1]))
    elif d.split('/')[-2] + '/' +  d.split('/')[-1] in pseudo_test_list:
        if os.path.exists(os.path.join(data_dir, 'pseudo_test', d.split('/')[-2])):
            shutil.copytree(d, os.path.join(data_dir, 'pseudo_test', d.split('/')[-2], d.split('/')[-1]))
        else:
            os.mkdir(os.path.join(data_dir, 'pseudo_test', d.split('/')[-2]))
            shutil.copytree(d, os.path.join(data_dir, 'pseudo_test' , d.split('/')[-2], d.split('/')[-1]))

#processed_new.txtに書き込む
with open(os.path.join(data_dir, 'processed_new.txt'), 'w') as f:
    f.writelines(result)


