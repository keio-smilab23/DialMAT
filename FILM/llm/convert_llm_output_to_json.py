import json
import copy
import sys
from venv import create

def create_new_json(ref_json_path, output_json_path, appended_data):
    with open(ref_json_path, 'r') as f:
        ref_data = json.load(f)
    f.close()
    
    output = copy.deepcopy(ref_data)

    dump = {}
    dump['list_of_highlevel_actions'] = appended_data[0]
    dump['categories_in_inst'] =  appended_data[1]
    dump['second_object'] =  appended_data[2]
    dump['caution_pointers'] =  appended_data[3]
    dump['task_type'] =  appended_data[4]
    dump['sliced'] =  appended_data[5]

    output['task_id'] = ref_data['task']

    output["high_level_actions"] = dump

    with open(output_json_path, 'w') as wf:
        json.dump(output, wf, indent=2)
    f.close()

def convert_to_FILM(llm_output):
    action_dict = {'Open': 'OpenObject', 'Close':'CloseObject', 'Put-At/In':'PutObject', 'Toggle-On':'ToggleObjectOn', 'Toggle-Off': 'ToggleObjectOff', 'Move-To':'XXX', 'Pickup':'PickupObject', 'Clean':'XXX', 'Heat':'XXX', 'Cool':'XXX', 'Slice':'XXX', 'Look-At-In-Light':'XXX'}
    new_dict = {}

    # {"list_of_highlevel_actions": [('Pickup', 'DishSponge'), ('Open', 'Cabinet')], 
    # "categories_in_inst": [], "second_object": [], "caution_pointers": []}

    sliced_flag = False

    list_of_actions = []
    for k, v in llm_output.items():
        for pair in v:
            object = pair[1]
            action = action_dict[pair[0]]
            if action == "Slice":
                sliced_flag = True
            if action != "XXX":
                list_of_actions.append( ( object, action ) )
    
    categories_in_inst = set()
    for pair in list_of_actions:
        object = pair[0]
        categories_in_inst.add(object)
    
    

    return list_of_actions, list(categories_in_inst), [], [], "task_type=NA", sliced_flag 

def main(): 
    llm_output = {0: [('Move-To', 'CounterTop')],
            1: [('Pickup', 'DishSponge'), ('Move-To', 'Cabinet')],
            2: [('Open', 'Cabinet')],
            3: [('Put-At/In', 'Cabinet')],
            4: [('Close', 'Cabinet')]}
    
    create_new_json("dialfred_testset_final/0001.json", "new_dialfred_testset_final/0001.json", convert_to_FILM(llm_output))
    

if __name__ == "__main__":
    main()