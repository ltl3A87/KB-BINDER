# KB-BINDER
The implementation for paper [Few-shot In-context Learning for Knowledge Base Question Answering](http://arxiv.org/abs/2305.01750)

## Set up

1. Set up the knowledge base server: Follow [Freebase Setup](https://github.com/dki-lab/Freebase-Setup) to set up a Virtuoso triplestore service. After starting your virtuoso service, replace the url in `utils/sparql_executer.py` with your own.
2. Download GrailQA dataset and other required files from the [link](https://drive.google.com/drive/folders/1g8ZpMLSw95KwjisXEw07rVVC3TJ1LZdn?usp=sharing) and put them under `data/`.
3. Install all required libraries:
```
$ pip install -r requirements.txt
```

## Run Experiments
```
$ python3 import few_shot_kbqa.py --shot_num 40 --temperature 0.3
 --api_key [your api key] --engine [engine model name]
--train_data_path [your train data path] --eva_data_path [your eva data path]
--fb_roles_path [your freebase roles file path] --surface_map_path [your surface map file path]
```
