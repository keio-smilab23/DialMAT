import json
import os
import sys
import copy
from tqdm import tqdm
import time
import openai
import re
import argparse

error_log = {"num_subgoal": [], "no_object": [], "in_hand":[], "cant_open":[], "gpt_output":[], "unknown_action": [] }


def Ask_ChatGPT(message, system_message=""):
    openai.organization = ""
    openai.api_key = "" # 
    # 応答設定
    completion = openai.ChatCompletion.create(
                 model    = "gpt-3.5-turbo",     # モデルを選択
                 messages = [{
                    "role":"system",
                    "content":system_message,   # メッセージ 
                 },{
                    "role":"user",
                    "content":message,   # メッセージ 
                }],

                 max_tokens  = 1024,             # 生成する文章の最大単語数
                 n           = 1,                # いくつの返答を生成するか
                 stop        = None,             # 指定した単語が出現した場合、文章生成を打ち切る
                 temperature = 0,              # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
    )
    
    # 応答
    response = completion.choices[0].message.content
    
    # 応答内容出力
    return response

def add_target(target, target_action, list_of_actions):
    openable_objects = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe', 'Box', 'Laptop']
    if target in openable_objects:
        list_of_actions.append((target, "OpenObject"))
    list_of_actions.append((target, target_action))
    if target in openable_objects:
        list_of_actions.append((target, "CloseObject"))
    return list_of_actions


def convert_to_FILM_with_place(llm_output, filename, num_subgoal, full_instruction):
    action_dict = {'Open': 'OpenObject', 'Close':'CloseObject', 'Put-At/In':'PutObject', 'Toggle-On':'ToggleObjectOn', 'Toggle-Off': 'ToggleObjectOff', 'Move-To':'MoveTo', 'Pickup':'PickupObject', 'Clean':'XXX', 'Heat':'XXX', 'Cool':'XXX', 'Slice':'SliceObject', 'Look-At-In-Light':'XXX'}
    object_action = ['Toggle-On', 'Toggle-Off', 'Pickup', 'Clean', 'Heat', 'Look-At-In-Light']
    open_action = ['Open', 'Close'] # , 'Put-At/In', 'Move-To'
    openable_objects = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe', 'Box', 'Laptop']
    new_dict = {}
    output_env_point = []
   
    sliced_object = "NA"
    categories_in_inst = set()
    prev_action = ("None", "None")
    list_of_actions = []

    current_object_in_hand = "None"

    for k, v in llm_output.items():
        for pair in v:
            object = pair[1]
            action = pair[0]
            place = pair[2]
            # print(action)

            if action == "None":
                continue

            if action not in action_dict.keys():
                if action == "Leave" or action == "Place":
                    action = "Put-At/In"
                elif action == "Grab":
                    action = "Pickup"
                else:
                    log_error(filename, f"Unknown action ### {action} ###")
                    error_log["unknown_action"].append(filename)

            if len(llm_output.keys()) != num_subgoal:
                log_error(filename, "Number of subgoals and llm_output does not match!")
                error_log["num_subgoal"].append(filename)
                return ["ERROR"], ["ERROR"], [], [], "NA", False, []


            if action == "Slice":
                sliced_object = object
                list_of_actions.append( ( object, "SliceObject" ) )
                # caution_pointers.append(len(list_of_highlevel_actions))
                # list_of_actions.append(("SinkBasin", "PutObject"))
                prev_action = list_of_actions[-1]
                continue

            if sliced_object == object:
                object = object + "Sliced"
                categories_in_inst.add(object)
            if "<Object>Sliced" in object:
                object = sliced_object + "Sliced"
            
            # if action == 'Heat':

            

            # if len(list_of_actions) > 1 and list_of_actions[-1] == (place, "CloseObject"):
            #     continue
            
            if action in ['Put-At/In', 'Cool'] and ( (place in ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe', 'Box']) or (object in ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe', 'Box']) ) and prev_action[1] != "OpenObject":
                # if len(list_of_actions) > 1 and list_of_actions[-1] != (place, "OpenObject"):
                list_of_actions.append((place, "OpenObject"))
                list_of_actions.append((place, "PutObject"))
                list_of_actions.append((place, "CloseObject"))
                prev_action = list_of_actions[-1]
                current_object_in_hand = "None"
                continue
            
            if action == "Move-To":
                if place != "None":
                    list_of_actions.append( ( place, action_dict[action] ) )
                else:
                    list_of_actions.append( ( object, action_dict[action] ) )
                prev_action = list_of_actions[-1]
                continue


            # if action_dict[action] == "XXX":
            #     continue

            if action == "Pickup":
                if object != "None":
                    list_of_actions.append( ( object, action_dict[action] ) )
                else:
                    log_error(filename, "No object to be picked up")
                    error_log["no_object"].append(filename)
                    return ["ERROR"], ["ERROR"], [], [], "NA", False, []
                prev_action = list_of_actions[-1]
                if current_object_in_hand == "None":
                    current_object_in_hand = object
                else:
                    log_error(filename, f"Already have {current_object_in_hand} in hand while trying to pick up {object}")
                    error_log["in_hand"].append(filename)
                    # return ["ERROR"], ["ERROR"], [], [], "NA", False, []
                continue
                
            if action == 'Put-At/In':
                list_of_actions.append((place, "PutObject"))
                prev_action = list_of_actions[-1]
                current_object_in_hand = "None"
                continue

            if action in object_action:
                if object != "None":
                    list_of_actions.append( ( object, action_dict[action] ) )
                else:
                    list_of_actions.append( ( place, action_dict[action] ) )
                prev_action = list_of_actions[-1]
                continue
            
            if action in open_action:
                if place in openable_objects:
                    list_of_actions.append( ( place, action_dict[action] ) )
                elif object in openable_objects:
                    list_of_actions.append( ( object, action_dict[action] ) )
                else:
                    log_error(filename, "Tried to open/close object not in openable_objects list")
                    error_log["cant_open"].append(filename)
                    return ["ERROR"], ["ERROR"], [], [], "NA", False, []
                prev_action = list_of_actions[-1]
                continue

            

        output_env_point.append(len(list_of_actions))
        # print(current_object_in_hand)
    
    for pair in list_of_actions:
        object = pair[0]
        categories_in_inst.add(object)

   
    prev_pair = ("None", "None")
    new_list_of_actions = []
    update_pointer = []
    for idx, pair in enumerate(list_of_actions):
        if prev_pair == pair:
            update_pointer.append( (idx, -1)) 
        else:
            new_list_of_actions.append(pair)
        prev_pair = pair
    list_of_actions = new_list_of_actions
    output_env_point = update_output_env_point(output_env_point, update_pointer)
    
    # SliceObject の前に必ず knife を持つ
    has_knife = False
    new_list_of_actions = []
    update_pointer = []
    for idx, pair in enumerate(list_of_actions):
        if pair[1] == "SliceObject" and not has_knife:
            knife_type = "ButterKnife" if "ButterKnife" in categories_in_inst else "Knife"
            new_list_of_actions.append( (knife_type, "PickupObject") )
            # output_env_point = update_output_env_point(output_env_point, idx)
            update_pointer.append( (idx, 1)) 
        elif pair[1] == "PickupObject" and ( pair[0] == "Knife" or pair[0] == "ButterKnife" ):
            has_knife = True
        new_list_of_actions.append(pair)
    output_env_point = update_output_env_point(output_env_point, update_pointer)
    list_of_actions = new_list_of_actions


    # SliceObject の終わったあとに指定がなければ sinkbasin に knife を置く
    prev_pair = ("None", "None")
    new_list_of_actions = []
    update_pointer = []
    for idx, pair in enumerate(list_of_actions):
        if prev_pair[1] == "SliceObject" and pair[1] != "PutObject":
            new_list_of_actions.append( ("SinkBasin", "PlaceObject") )
            # output_env_point = update_output_env_point(output_env_point, idx-1)
            update_pointer.append( (idx, 1)) 
        new_list_of_actions.append(pair)
        prev_pair = pair
    list_of_actions = new_list_of_actions
    output_env_point = update_output_env_point(output_env_point, update_pointer)

    # すでに手に持っている状態で PickupObject を要求される場合
    ### 不要なものを持っている場合
    ### 不要な PickupObject を実行しようとする場合


    # second_object を作成
    second_object = []
    if "two" in full_instruction:
        prev_actions = []
        for pair in list_of_actions:
            if pair[1] == "PickupObject" and pair in prev_actions:
                second_object.append(True)
                second_object.append(False)
                break
            else:
                second_object.append(False)
            prev_actions.append(pair)
    
    return new_list_of_actions, list(categories_in_inst), second_object, [], "NA", False, output_env_point


def update_output_env_point(output_env_point, update_pointer):
    new_output_env_point = output_env_point
    for i, x in enumerate(output_env_point):
        for idx, value in update_pointer:
            if x > idx:
                new_output_env_point[i] += value
        
    return new_output_env_point



def create_new_json(ref_data, output_json_path, appended_data, raw_output, failed):
    output = copy.deepcopy(ref_data)

    if not failed:
        dump = {}
        dump['list_of_highlevel_actions'] = appended_data[0]
        dump['categories_in_inst'] =  appended_data[1]
        dump['second_object'] =  appended_data[2] # ALWAYS []
        dump['caution_pointers'] =  appended_data[3] # 
        dump['task_type'] =  appended_data[4] #ALWAYS "NA"
        dump['sliced'] =  appended_data[5] # ALWAYS FALSE
        dump['raw_output'] = raw_output
        dump['output_env_point'] = appended_data[6]

        output['task_id'] = output_json_path

        output["high_level_actions"] = dump

        with open(output_json_path, 'w') as wf:
            json.dump(output, wf, indent=2)
        wf.close()
    else:
        dump = {}
        dump['list_of_highlevel_actions'] = ["ERROR"]
        dump['categories_in_inst'] =  ["ERROR"]
        dump['second_object'] =   []
        dump['caution_pointers'] =   []
        dump['task_type'] =  "NA"
        dump['sliced'] =  False
        dump['raw_output'] = raw_output
        dump['output_env_point'] = []

        output['task_id'] = output_json_path

        output["high_level_actions"] = dump

        with open(output_json_path, 'w') as wf:
            json.dump(output, wf, indent=2)
        wf.close()



def preproces_instruction(inst):
    inst = inst.lower()
    inst = inst.replace(",", " then")
    inst = inst.replace(".", "")
    inst = inst.rstrip()
    return inst

def output_llm_propmt(json_data, llm_data_dir):
    dump = {}
    for idx, inst in enumerate(json_data["turk_annotations"]["anns"][0]["high_descs"]):
        dump[idx] = preproces_instruction(inst)

    with open(f"{llm_data_dir}/test_prompt.txt", "r") as f:
        prompt = f.read()
    f.close()

    print(json.dumps(dump, indent=1))

    prompt = prompt.replace("XXXXX", json.dumps(dump, indent=1))
    with open(f"{llm_data_dir}/llm_input.txt", "w") as wf:
        wf.write(prompt)
    wf.close()

    return prompt, len(dump.keys())


def get_GPT_output(prompt, filename, raw_output=None):

    # ChatGPT only gives response 3 times / min
    # dump = {}

    if raw_output == None:
        while(1):
            try:
                # llm_data = get_new_json(instruction, prompt_categorize, prompt_extract)
                raw_output = Ask_ChatGPT(prompt)
                # print(output)
                # sys.exit()
                break
            except Exception as e:
                if "<class 'openai.error.RateLimitError'>" in str(type(e)):
                    print(f"Waiting for ChatGPT ...")
                else:
                    print(f"Unknown Error")
                time.sleep(1)
    
    pattern = r'{.*}'
    match = re.search(pattern, raw_output, re.DOTALL)

    try:
        if match:
            # print("a")
            llm_output = match.group(0)
            # print(llm_output)
            llm_output = json.loads(llm_output)
            # print(llm_output)
            failed = False
        else:
            # print("b")
            log_error(filename, "ERROR matching GPT output")
            error_log["gpt_output"].append(filename)
            llm_output = []
            failed = True
    except:
        # print("c")
        log_error(filename, "ERROR matching GPT output (no match)")
        error_log["gpt_output"].append(filename)
        llm_output = []
        failed = True
    
        
    print(llm_output, failed)
    # sys.exit()


    return llm_output, raw_output, failed


def create_new_file(dir_path, output_dir_path, llm_data_dir, use_llm_output = False):
    for filename in tqdm(sorted(os.listdir(dir_path))):
        if filename.endswith(".json"):
            with open(os.path.join(dir_path, filename)) as json_file:
                print(f"=== {filename} ===")

                data = json.load(json_file)
                prompt, num_subgoal = output_llm_propmt(data, llm_data_dir)


                if use_llm_output:
                    llm_output, raw_output, failed = get_GPT_output(prompt, filename, None)
                else:
                    with open(os.path.join(output_dir_path, filename)) as jf:
                       obtained_raw_output = json.load(jf)["high_level_actions"]["raw_output"]
                    jf.close()
                    llm_output, raw_output, failed = get_GPT_output(prompt, filename, obtained_raw_output)
                
                output_path = os.path.join(output_dir_path, filename)
                new_data = []
                if not failed:
                    new_data = convert_to_FILM_with_place(llm_output, filename, num_subgoal, data["turk_annotations"]["anns"][0]["task_desc"])
                create_new_json(data, output_path, new_data, raw_output, failed)

                if not failed:
                    for idx, pair in enumerate(new_data[0]):
                        print(pair, end="")
                        if idx+1 in  new_data[6]:
                            print(" <== ")
                        else:
                            print()
                # sys.exit()

    for error, data in error_log.items():
        print(f"{error}: {len(data)}")






def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-f', '--file')
    parser.add_argument('-m', '--mode')
    parser.add_argument('--llm', action ="store_true")
    args = parser.parse_args()
    

    dir_path = "testset/dialfred_testset_final/"
    output_dir_path = "testset/new_dataset/"
    llm_data_dir = "testset/llm"

    os.makedir(output_dir_path, exist_ok=True)

    if args.mode == "all":
        create_new_file(dir_path, output_dir_path, llm_data_dir, use_llm_output=args.llm)


if __name__ == "__main__":
    main()
