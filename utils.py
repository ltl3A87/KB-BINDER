import json


def process_file(filename):
    data = json.load(open(filename, 'r'), strict=False)
    selected_data = []
    for example in data:
        sele_dict = {}
        sele_dict["id"] = example["qid"]
        sele_dict["question"] = example["question"]
        sele_dict["sparql_query"] = example["sparql_query"]
        sele_dict["s_expression"] = example["s_expression"]
        sele_dict["answer"] = example["answer"]
        selected_data.append(sele_dict)
    return selected_data


def process_file_node(filename):
    data = json.load(open(filename, 'r'), strict=False)
    question_to_mid_dict = {}
    for example in data:
        key = example["question"]
        question_to_mid_dict[key] = {}
        node_list = example["graph_query"]["nodes"]
        if node_list:
            for node in node_list:
                if node["node_type"] == "entity":
                    question_to_mid_dict[key][node["id"]] = node["friendly_name"]
        else:
            topic_mid = example["topic_entity"]
            question_to_mid_dict[key][topic_mid] = example["topic_entity_name"]

    return question_to_mid_dict


def process_file_test(filename):
    data = json.load(open(filename, 'r'), strict=False)
    selected_data = []
    for example in data:
        sele_dict = {}
        sele_dict["id"] = example["qid"]
        sele_dict["question"] = example["question"]
        selected_data.append(sele_dict)
    return selected_data

def process_file_rela(file_path_name, max_index):
    return_dict = {}
    for i in range(max_index):
        filename = file_path_name + str(i)
        data = json.load(open(filename, 'r'), strict=False)
        for key in data:
            if key not in return_dict:
                return_dict[key] = data[key]
            else:
                return_dict[key] = list(set(return_dict[key] + data[key]))
    return return_dict
