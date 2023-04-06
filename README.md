# DialFRED: Dialogue-Enabled Agents for Embodied Instruction Following

Language-guided Embodied AI benchmarks requiring an agent to navigate an environment and manipulate objects typically allow one-way communication: the human user gives a natural language command to the agent, and the agent can only follow the command passively. In this work, we present DialFRED, a dialogue-enabled embodied instruction following benchmark based on the ALFRED benchmark. DialFRED allows an agent to actively ask questions to the human user; the additional information in the user’s response is used by the agent to better complete its task. We release a human-annotated dataset with 53K task-relevant questions and answers and an oracle to answer questions. To solve DialFRED, we propose a questioner-performer framework wherein the questioner is pre-trained with the human-annotated data and fine-tuned with reinforcement learning. Experimental results show that asking the right questions leads to significantly improved task performance.

## Dependency

Inherited from the E.T. repo, the package is depending on:
- numpy
- pandas
- opencv-python
- tqdm
- vocab
- revtok
- numpy
- Pillow
- sacred
- etaprogress
- scikit-video
- lmdb
- gtimer
- filelock
- networkx
- termcolor
- torch==1.7.1?
- torchvision==0.8.2?
- tensorboardX==1.8
- ai2thor==2.1.0
- stanza
- Werkzeug==2.1.1
- E.T. (https://github.com/alexpashevich/E.T.)

## Setup
Set up macro:
```
export DF_ROOT=$(pwd)
export LOGS=$DF_ROOT/logs
export DATA=$DF_ROOT/data
export PYTHONPATH=$PYTHONPATH:$DF_ROOT
```

Install requirements:
```bash
CONFIGURE_OPTS=--enable-shared pyenv install 3.7.1
pyenv virtualenv 3.7.1 df_env
pyenv local df_env

cd $DF_ROOT
pip install --upgrade pip
pip install -r requirements.txt
```

## Downloading data and checkpoints

Download [ALFRED dataset](https://github.com/askforalfred/alfred) (**about 2h?**):

**You can get `nas07/06DialFRED-Challenge/data/json_2.1.0`, `nas07/06DialFRED-Challenge/data/json_feat_2.1.0` and store in `DialFRED-Challenge/data/` instead of the following.**
```bash
cd $DATA
sh download_data.sh json
sh download_data.sh json_feat
```

Copy pretrained checkpoints:
```bash
wget http://pascal.inrialpes.fr/data2/apashevi/et_checkpoints.zip
unzip et_checkpoints.zip
mv pretrained $LOGS/
```

Render images (**about 22h5m**):

**You can get `nas07/06DialFRED-Challenge/data/generated_2.1.0` and store in `DialFRED-Challenge/data/` instead of the following.**
```bash
cd $DF_ROOT
python -m alfred.gen.render_trajs
```

## Prepare dataset

**You can get `nas07/06DialFRED-Challenge/data/lmdb_augmented_human_subgoal` and store in `DialFRED-Challenge/data/` instead of all the following in this section.**

We provide the code to augment the Alfred data by merging low level actions into subgoals and spliting one subgoal into multiple ones. We also created new instructions to improve language variety. 
```bash
python augment_data.py
```

We focus on three types of questions:
1. location clarification question: where is the object?
2. appearance clarification question: what does the object look like?
3. direction clarification question: which direction should I turn to?

To answer these questions, we build an oracle to extract ground-truth information from the virtual environment and generate answers based on templates. Given the offline generated answers, we further modify the data by sampling QA combinations in addition to the instructions. We use the modified data to pre-train the performer.

``` bash
python append_data.py
```

Following the ET pipeline, we can create the lmdb dataset 
``` bash
export EXP_NAME=augmented_human
export SUBGOAL=subgoal
export EVAL_TYPE=valid_unseen

# create lmdb dataset
python -m alfred.data.create_lmdb with args.visual_checkpoint=$LOGS/pretrained/fasterrcnn_model.pth args.data_output=lmdb_${EXP_NAME}_${SUBGOAL} args.vocab_path=$DF_ROOT/files/$EXP_NAME.vocab > ./logs/et_${EXP_NAME}_${SUBGOAL}.log 2>&1 &

```

## Human QA data

We use crowd-sourcing to collect 53K human task-oriented questions and answers. The dataset are available at:
```
./data/dialfred_human_qa.csv
```
Each line contains one annotation for an augmented sub-goal. The definition of each columns are:
```
1   Data splits. Can be training, validation_seen or validation_unseen.
2   Task ID.
3   Trial ID.
4   Room type. 
5   Task type in the original ALFRED dataset.
6   Subgoal start time. Start time of the augmented sub-goal in the corresponding task video.    
7   Subgoal end time. End time of the augmented sub-goal in the corresponding task video.
8   Number of low level actions in the sub-goal. 
9   Sub-goal Index. The index of the sub-goal in the high-level task.
10  Instruction for the augmented subgoal.
11  Verb. The low level actions contained in the sub-goal.
12  Noun1. The first noun involved in the sub-goal.
13  Noun2. The optional second noun involved in the sub-goal.
14  Question type. The type of question asked by the annotator. Can be location, appearance, direction or other types.
15  Question. The question asked by the annotator.
16  Answer. The answer to the question provided by the annotator.
17  Necessary. Whether the annotator thinks the question and answer are necessary for the task completion.
```

## Questioner and performer evaluation

**If the `aws` command is not available ([reference](https://docs.aws.amazon.com/ja_jp/cli/latest/userguide/getting-started-install.html)):**
```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
```

Download checkpoints for finetuned questioner and pretrained performer.
```bash
scripts/fetch_model_checkpt.sh

```
For evaluating the pretrained models (**about 78m**):
```bash
python train_eval.py --mode eval

```

## Train the questioner and performer from scratch

Given the lmdb dataset, we can pre-train the performer (**about 3h38m**).
```bash
export EXP_NAME=augmented_human
export SUBGOAL=subgoal

# train the ET performer
python -m alfred.model.train with exp.model=transformer exp.name=et_${EXP_NAME}_${SUBGOAL} exp.data.train=lmdb_${EXP_NAME}_${SUBGOAL} train.seed=1 > ./logs/et_${EXP_NAME}_${SUBGOAL}.log 2>&1 &

```

Given the human QA data we collected via crowd-sourcing, we can pretrain the questioner model (**about 46m**).
```bash
python seq2seq_questioner_multimodel.py

```

Given the pretrained questioner and performer, we can finetune the questioner model using RL on valid seen (**about 1h2m**).
```bash
# RL anytime: training the questioner to ask questions at anytime of the task
python train_eval.py --mode train --questioner-path ./logs/questioner_rl/pretrained_questioner.pt

```

Given the finetuned questioner and pretrained performer, we can evaluate the models on valid unseen (**about 1h20m**).
```bash
# RL anytime: training the questioner to ask questions at anytime of the task
python train_eval.py --mode eval --questioner-path ./logs/questioner_rl/questioner_anytime_seen1.pt

```

## Citation

If you use our code or data, please consider citing our paper.
```bash
@article{gao2022dialfred,
  title={Dialfred: Dialogue-enabled agents for embodied instruction following},
  author={Gao, Xiaofeng and Gao, Qiaozi and Gong, Ran and Lin, Kaixiang and Thattai, Govind and Sukhatme, Gaurav S},
  journal={arXiv preprint arXiv:2202.13330},
  year={2022}
}
```