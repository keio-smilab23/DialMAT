# DialMAT: Dialogue-Enabled Transformer with Moment-based Adversarial Training

[[paper](https://embodied-ai.org/papers/2023/23.pdf)]

Kanta Kaneda*, Ryosuke Korekata*, Yuiga Wada*, Shunya Nagashima*, Motonari Kambara,
Yui Iioka, Haruka Matsuo, Yuto Imai, Takayuki Nishimura and Komei Sugiura

![model (1)](https://github.com/keio-smilab23/DialMAT/assets/51681991/00fb772b-2590-4269-a05b-24db71e1aef4)


This paper focuses on the DialFRED task, which is the
task of embodied instruction following in a setting where
an agent can actively ask questions to the human user. To
address this task, we propose DialMAT. DialMAT introduces Moment-based Adversarial Training, which incorporates adversarial perturbations into the latent space of
language, image, and action. Additionally, it introduces a
crossmodal parallel feature extraction mechanism that applies foundation models to both language and image. We
evaluated our model using a dataset constructed from the
DialFRED dataset and demonstrated superior performance
compared to the baseline method in terms of success rate
and path weighted success rate. The model secured the top
position in the DialFRED Challenge, which took place at
the CVPR 2023 Embodied AI workshop.

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

**You can get `nas07/06DialFRED-Challenge/data/json_2.1.0.tar.gz`, `nas07/06DialFRED-Challenge/data/json_feat_2.1.0.tar.gz` and extract in `DialFRED-Challenge/data/` instead of the following.**
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

**You can get `nas07/06DialFRED-Challenge/data/generated_2.1.0.tar.gz` and extract in `DialFRED-Challenge/data/` instead of the following.**
```bash
cd $DF_ROOT
python -m alfred.gen.render_trajs
```

## Prepare dataset
**2022-04-21 Update: Since `augment_data.py` and `append_data.py` are already executed and `/traj_data.json` and `files/augmented_human.vocab` are already updated in this repository, start from `export EXP_NAME=augmented_human' (skip augment_data.py and append_data.py)**

<!-- **2022-04-10 You can get `nas07/06DialFRED-Challenge/data/lmdb_augmented_human_subgoal.tar.gz` and extract in `DialFRED-Challenge/data/` instead of all the following in this section.** -->

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
You can split the data of valid_unseen into pseudo_valid and pseudo_test for model evaluation by the following command.

```bash
python split_val_unseen.py
```


Following the ET pipeline, we can create the lmdb dataset 
``` bash
export EXP_NAME=augmented_human
export SUBGOAL=subgoal
export EVAL_TYPE=valid_unseen

# create lmdb dataset
python -m alfred.data.create_lmdb with args.visual_checkpoint=$LOGS/pretrained/fasterrcnn_model.pth args.data_output=lmdb_${EXP_NAME}_${SUBGOAL} args.vocab_path=$DF_ROOT/files/$EXP_NAME.vocab > ./logs/et_${EXP_NAME}_${SUBGOAL}.log 2>&1 &

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

## Create a submission file for CVPR 2023 DialFRED Challenge
The testset contains 1092 tasks. For each task, json data and the oracle answers for the 3 types of questions in the paper are provided in `testset/`
To create a submission file, execute the following command:
```bash
python generate_submission.py --mode eval --questioner-path ./logs/questioner_rl/questioner_anytime_seen1.pt
```

## Evaluate your model from any metadata

As Eval.AI evaluates metrics from metadata, we have prepared a script to compute SR from arbitrary metadata.
This script evaluates the SR from `submission_file/{traj_key_str}.json`.

```bash
python eval_from_metadata.py --mode eval  --performer-path model_09.pth --questioner-path questioner_anytime_seen1.pt  --clip_resnet=true
```


## Citation

If you use our code or data, please consider citing our paper.
```bash

```

## License

Our implementation uses code from the following repositories:

- [DialFRED](https://github.com/xfgao/DialFRED) for experiment pipeline
- [HLSM-MAT](https://github.com/keio-smilab22/HLSM-MAT) for [Moment-based Adversarial Training](https://arxiv.org/abs/2204.00889)

