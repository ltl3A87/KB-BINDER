import openai
import json
import spacy
from time import sleep
nlp = spacy.load("en_core_web_sm")
from rank_bm25 import BM25Okapi
from bm25_trial import BM25_self
import random
import numpy as np
import re

type_to_rela_dict = {"tag_to_movie": "has_tags", "writer_to_movie": "written_by", "movie_to_tags": "has_tags",
                         "movie_to_year": "release_year", "movie_to_writer": "written_by", "movie_to_language": "in_language",
                         "movie_to_genre": "has_genre", "director_to_movie": "directed_by", "movie_to_actor": "starred_actors",
                         "movie_to_director": "directed_by", "actor_to_movie": "starred_actors"
                         }

def type_process(file_name):
    test_type_list = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            type = line.strip()
            test_type_list.append(type)
    return test_type_list
# test_type_list_1hop = type_process('data/data/metaQA/qa_test_qtype_1hop.txt')
train_type_list_2hop = type_process('data/data/metaQA/qa_train_qtype_2hop.txt')


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


def convert_que_to_logical_form(question):
    qt = train_ques_to_type_dict[question]
    qt_seg = qt.split('_')
    rela_1 = "_".join(qt_seg[:3])
    rela_2 = "_".join(qt_seg[2:5])
    ent_s_idx = question.index("[")
    ent_e_idx = question.index("]")
    ent = question[ent_s_idx:ent_e_idx+1]
    logical_form = rela_2 + "(" + rela_1 + "(" + ent + "))"
    # print("logical form: ", logical_form)
    return logical_form, rela_1, rela_2

# prompt_type = "Given the following types: actor_to_movie, movie_to_writer, tag_to_movie, writer_to_movie, movie_to_year, director_to_movie, movie_to_language, movie_to_genre, movie_to_director, movie_to_actor, movie_to_tags\nQuestion: what movies are about [ginger rogers] \nQuestion type: tag_to_movie\nQuestion: what movies was [Erik Matti] the writer of\nQuestion type: writer_to_movie\nQuestion: what topics is [Bad Timing] about\nQuestion type: movie_to_tags\nQuestion: [True Romance], when was it released\nQuestion type: movie_to_year\nQuestion: who wrote the screenplay for [True Romance]\nQuestion type: movie_to_writer\nQuestion: what language is [Cabeza de Vaca] in\nQuestion type: movie_to_language\nQuestion: what kind of film is [True Romance]\nQuestion type: movie_to_genre\nQuestion: can you name a film directed by [William Cameron Menzies]\nQuestion type: director_to_movie\nQuestion: who acted in [Terminal Velocity]\nQuestion type: movie_to_actor\nQuestion: who's the director of [True Romance]\nQuestion type: movie_to_director\nQuestion: what does [Sacha Baron Cohen] appear in\nQuestion type: actor_to_movie\n"
def two_hop_type_generator(question):
    prompt = "Given the following operations: actor_to_movie, movie_to_writer, tag_to_movie, writer_to_movie, movie_to_year, director_to_movie, movie_to_language, movie_to_genre, movie_to_director, movie_to_actor, movie_to_tags\n"
    # tokenized_query = nlp(question)
    tokenized_query = question.split()
    s_index = -1
    e_index = -1
    for i, seg in enumerate(tokenized_query):
        if "[" in seg:
            s_index = i
        if "]" in seg:
            e_index = i
    if e_index != len(tokenized_query) - 1:
        tokenized_query = tokenized_query[:s_index] + tokenized_query[e_index + 1:]
    else:
        tokenized_query = tokenized_query[:s_index]
    top3_ques = bm25_train_full.get_top_n(tokenized_query, corpus, n=5)
    doc_scores = bm25_train_full.get_scores(tokenized_query)
    top_score = max(doc_scores)
    print("top_score: ", top_score)
    print("top3 questions: ", top3_ques[:3])
    selected_examples_cur = top3_ques
    for que in selected_examples_cur:
        logical_form, rela_1, rela_2 = convert_que_to_logical_form(que)
        prompt = prompt + "Question: " + que + "\nLogical Form: " + logical_form + "\nTwo operations: " + rela_2 + ", " + rela_1 + "\n"
    # prompt = prompt + " Question: " + question + "\nQuestion type: "
    # prompt = "Given the following operations: actor_to_movie, movie_to_writer, tag_to_movie, writer_to_movie, movie_to_year, director_to_movie, movie_to_language, movie_to_genre, movie_to_director, movie_to_actor, movie_to_tags\nQuestion: which person wrote the films directed by [Yuriy Norshteyn]\nLogical Form: movie_to_writer(director_to_movie([Yuriy Norshteyn]))\nTwo operations: movie_to_writer, director_to_movie\nQuestion: which movies have the same director of [Just Cause]\nLogical Form: director_to_movie(movie_to_director([Yuriy Norshteyn]))\nTwo operations: director_to_movie, movie_to_director\nQuestion: what genres do the movies written by [Maureen Medved]\nLogical Form: movie_to_genre(writer_to_movie([Maureen Medved]))\nTwo operations: movie_to_genre, writer_to_movie\nQuestion: what were the release years of the movies acted by [Todd Field]\nLogical Form: movie_to_year(actor_to_movie([Todd Field]))\nTwo operations: movie_to_year, actor_to_movie\nQuestion: the films written by [Babaloo Mandel] starred which actors\nLogical Form: movie_to_actor(writer_to_movie([Babaloo Mandel]))\nTwo operations: movie_to_actor, writer_to_movie\n"
    prompt = prompt + "Question: " + question + "\nLogical Form: "
    got_result = False
    while got_result != True:
        try:
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


def retrieve_answer(found_type, found_ent):
    found_relas = enti_to_fact_dict[found_ent]
    # print("found_ent: ", found_ent)
    # print("found_relas: ", found_relas)
    rela_to_ans_dict = {}
    for fact in found_relas:
        s, r, o = fact.split("|")
        if s == found_ent:
            if r in rela_to_ans_dict:
                rela_to_ans_dict[r].append(o)
            else:
                rela_to_ans_dict[r] = [o]
        if o == found_ent:
            if r in rela_to_ans_dict:
                rela_to_ans_dict[r].append(s)
            else:
                rela_to_ans_dict[r] = [s]
    rela = type_to_rela_dict[found_type]
    # print("rela_to_ans_dict: ", rela_to_ans_dict)
    if rela in rela_to_ans_dict:
        return rela_to_ans_dict[rela]
    else:
        return []



if __name__=="__main__":
    openai.api_key = ""
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

    test_question_2hop = ques_ans_process('data/data/metaQA/qa_test_2hop.txt')
    train_question_2hop = ques_ans_process('data/data/metaQA/qa_train_2hop.txt')
    train_ques_to_type_dict = {}
    for type, que in zip(train_type_list_2hop, train_question_2hop):
        train_ques_to_type_dict[que["question"]] = type

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

    corpus = [data["question"] for data in train_question_2hop]
    tokenized_train_data = []
    for doc in corpus:
        tokenized_train_data.append(doc.split())
    bm25_train_full = BM25Okapi(tokenized_train_data)

    total = 0
    correct = 0
    for ques_dict in test_question_2hop:
        question = ques_dict["question"]
        print("question: ", question)
        got_result = False
        while got_result is not True:
            try:
                question_type = two_hop_type_generator(question)
                # print("question_type: ", question_type)
                question_type = question_type.split("operations: ")[1]
                question_type = question_type.split(", ")
                for idx, type_ in enumerate(question_type):
                    if type_ not in type_to_rela_dict:
                        tokenized_query = re.split('_', type_)
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
                        question_type[idx] = types_all[bound_type]
                got_result = True
            except:
                sleep(3)
        question_type.reverse()
        print("question_type: ", question_type)
        # relas = type_to_rela_dict[question_type]
        # print("relas: ", relas)
        ent = ques_dict["retrieved_ent"]
        first_step_ans = retrieve_answer(question_type[0], ent)
        print("first_step_ans: ", first_step_ans)
        pred = []
        for ent_mid in first_step_ans:
            pred = pred + retrieve_answer(question_type[1], ent_mid)
        print("answer: ", ques_dict["answer"])
        print("pred: ", list(set(pred)))
        set_pred = set(pred)
        if ent in set_pred:
            set_pred.remove(ent)
        if set_pred == set(ques_dict["answer"]):
            correct += 1
        total += 1
        print("total: ", total)
        print("correct: ", correct)
        print("accuracy: ", correct/total)
