import openai
import json
import spacy
from time import sleep
from bm25_trial import BM25_self
import numpy as np
import re


def type_process(file_name):
    test_type_list = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            type = line.strip()
            test_type_list.append(type)
    return test_type_list

def ques_ans_process(file_name):
    return_dict_list = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            return_dict = {}
            question, answers = line.split('\t')
            answer_list = answers.split('|')
            answer_list[-1] = answer_list[-1].strip()
            ent_s_idx = question.index('[')
            ent_e_idx = question.index(']')
            retrieved_ent = question[ent_s_idx+1:ent_e_idx]
            return_dict["question"] = question
            return_dict["retrieved_ent"] = retrieved_ent
            return_dict["answer"] = answer_list
            return_dict_list.append(return_dict)
    return return_dict_list

def type_generator(question):
    prompt = prompt_type
    prompt = prompt + " Question: " + question + "\nQuestion type: "
    got_result = False
    while got_result != True:
        try:
            # print("promt: ", prompt)
            answer_modi = openai.Completion.create(
                engine="code-davinci-002",
                prompt=prompt,
                temperature=0,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["Question: "]
            )
            got_result = True
        except:
            sleep(3)
    gene_exp = answer_modi["choices"][0]["text"].strip()
    return gene_exp



openai.api_key = "" # need to be specified
nlp = spacy.load("en_core_web_sm")

enti_to_fact_dict = {}
with open('data/data/metaQA/kb.txt') as f:
    lines = f.readlines()
    for line in lines:
        s, r, o = line.split('|')
        if s.strip() not in enti_to_fact_dict:
            enti_to_fact_dict[s.strip()] = [line.strip()]
        else:
            enti_to_fact_dict[s.strip()].append(line.strip())
        if o.strip() not in enti_to_fact_dict:
            enti_to_fact_dict[o.strip()] = [line.strip()]
        else:
            enti_to_fact_dict[o.strip()].append(line.strip())
test_type_list_1hop = type_process('data/data/metaQA/qa_test_qtype_1hop.txt')
train_type_list_1hop = type_process('data/data/metaQA/qa_train_qtype_1hop.txt')

test_question_1hop = ques_ans_process('data/data/metaQA/qa_test_1hop.txt')
train_question_1hop = ques_ans_process('data/data/metaQA/qa_train_1hop.txt')
type_to_ques_dict = {}
for type, que in zip(train_type_list_1hop, train_question_1hop):
    if type in type_to_ques_dict:
        type_to_ques_dict[type].append(que["question"])
    else:
        type_to_ques_dict[type] = [que["question"]]

type_to_rela_dict = {"tag_to_movie": "has_tags", "writer_to_movie": "written_by", "movie_to_tags": "has_tags",
                     "movie_to_year": "release_year", "movie_to_writer": "written_by", "movie_to_language": "in_language",
                     "movie_to_genre": "has_genre", "director_to_movie": "directed_by", "movie_to_actor": "starred_actors",
                     "movie_to_director": "directed_by", "actor_to_movie": "starred_actors"
                     }

types_all = list(type_to_rela_dict.keys())
types_all_spl = [type_.split("_") for type_ in types_all]
type_drops_all = []
for i, rela in enumerate(types_all_spl):
    drops_types_all = []
    for word in rela:
        doc = nlp(word)
        if len(doc) > 0:
            drops_types_all.append(doc[0].lemma_)
    type_drops_all.append(" ".join(drops_types_all))
print("type_drops_all: ", type_drops_all)
bm25_all_relas = BM25_self()
bm25_all_relas.fit(type_drops_all)

prompt_type = "Given the following types: actor_to_movie, movie_to_writer, tag_to_movie, writer_to_movie, movie_to_year, director_to_movie, movie_to_language, movie_to_genre, movie_to_director, movie_to_actor, movie_to_tags\nQuestion: what movies are about [ginger rogers] \nQuestion type: tag_to_movie\nQuestion: what movies was [Erik Matti] the writer of\nQuestion type: writer_to_movie\nQuestion: what topics is [Bad Timing] about\nQuestion type: movie_to_tags\nQuestion: [True Romance], when was it released\nQuestion type: movie_to_year\nQuestion: who wrote the screenplay for [True Romance]\nQuestion type: movie_to_writer\nQuestion: what language is [Cabeza de Vaca] in\nQuestion type: movie_to_language\nQuestion: what kind of film is [True Romance]\nQuestion type: movie_to_genre\nQuestion: can you name a film directed by [William Cameron Menzies]\nQuestion type: director_to_movie\nQuestion: who acted in [Terminal Velocity]\nQuestion type: movie_to_actor\nQuestion: who's the director of [True Romance]\nQuestion type: movie_to_director\nQuestion: what does [Sacha Baron Cohen] appear in\nQuestion type: actor_to_movie\n"
total = 0
correct = 0
for ques_dict in test_question_1hop:
    print("total: ", total)
    question = ques_dict["question"]
    print("question: ", question)
    question_type = type_generator(question)
    print("question_type: ", question_type)
    if question_type not in type_to_rela_dict:
        tokenized_query = re.split('_', question_type)
        tokenized_ques = question.split()
        tokenized_query = tokenized_query + tokenized_ques
        drops_query = []
        for word in tokenized_query:
            doc = nlp(word)
            if len(doc) > 0:
                drops_query.append(doc[0].lemma_)
        drops_query = " ".join(drops_query)
        scores = list(bm25_all_relas.transform(drops_query, [i for i in range(11)]))
        sorted_score_index = list(np.argsort(scores))
        sorted_score_index.reverse()
        bound_type = sorted_score_index[0]
        question_type = types_all[bound_type]
    relas = type_to_rela_dict[question_type]
    print("relas: ", relas)
    ent = ques_dict["retrieved_ent"]
    found_relas = enti_to_fact_dict[ent]
    print("found_relas: ", found_relas)
    rela_to_ans_dict = {}
    for fact in found_relas:
        s, r, o = fact.split("|")
        if s == ent:
            if r in rela_to_ans_dict:
                rela_to_ans_dict[r].append(o)
            else:
                rela_to_ans_dict[r] = [o]
        if o == ent:
            if r in rela_to_ans_dict:
                rela_to_ans_dict[r].append(s)
            else:
                rela_to_ans_dict[r] = [s]
    print("rela_to_ans_dict: ", rela_to_ans_dict)
    print("answer: ", ques_dict["answer"])
    if relas in rela_to_ans_dict:
        pred = rela_to_ans_dict[relas]
    else:
        pred = []
    print("pred: ", pred)
    if set(pred) == set(ques_dict["answer"]):
        correct += 1
    total += 1
    print("accuracy: ", correct/total)







