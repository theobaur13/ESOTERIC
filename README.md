# ESOTERIC

ESOTERIC - Elasticsearch Semantic Optimized Text Extraction Retrieval from Information Corpus

A command line application that uses natural language processing tools to retrieve relevant evidence sentences for a given input claim from the FEVER dataset.

  ## Prerequisites
To run this system [Elasticsearch 8.12.2 x64](https://www.elastic.co/downloads/past-releases/elasticsearch-8-12-2) needs to be installed and an Elasticsearch database needs to be active. A detailed guide on how to set up Elasticsearch can be found [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html#_start_elasticsearch). 

To load embeddings  into the Elasticsearch database it is highly recommended that a [Google Colab](https://colab.research.google.com/) instance with a V100 GPU is used as this process is computationally expensive. Once the Elasticsearch database is loaded the retrieval process can be executed using only CPU power.

## Installation

Clone repository.

  

```bash
git clone https://github.com/theobaur13/ESOTERIC
```

Set up virtual environment.

  

```python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Create a `.env` file inside the `ESOTERIC` root directory and fill with the following information:
```
ES_HOST_URL={ELASTICSEARCH DB URL}
ES_USER={ELASTICSEARCH DB USERNAME}
ES_PASS={ELASTICSEARCH DB PASSWORD}
ES_PORT={ELASTICSEARCH DB PORT}
ES_SCHEME={ELASTICSEARCH DB HTTP SCHEME}
```

Create `data` directory inside main `ESOTERIC` directory.

```
mkdir data
```

Create `wiki-pages` directory inside `data` directory.

  

```
mkdir data\wiki-pages
```

### Elasticsearch Setup  

Download `wiki-pages.zip` from [FEVER](https://fever.ai/download/fever/wiki-pages.zip).

  

Extract `wiki-pages.zip` into `\data\wiki-pages` so that the `wiki-pages` directory appears as follows:

```
ESOTERIC
│
└───data
│
└───wiki-pages
	│ wiki-001.jsonl
	│ wiki-002.jsonl
	│ ...
	│ wiki-109.jsonl
```

Run `elasticsearch_loader.py`. The `batch_limit` argument specifies up to which wiki file to load up to (a `batch_limit` of 50 will load data from `wiki-001.jsonl` to `wiki-050.jsonl`). If `batch_limit` is left blank all wiki files will be loaded. 

__WARNING__: Be aware that loading all 109 files can mean that embeddings take upwards of 35 hours on a V100 GPU to load.

```python
py elasticsearch_loader.py --batch_limit 109
```
### Loading Embeddings
Either run `DPR_Embedding_Loader.ipynb` in a Google Colab instance or run `dpr_embedding_loader.py` locally (if you have a good GPU). If using Google Colab but hosting the Elasticsearch database locally, you will need to send the embeddings from the Google Colab instance to your machine, we did this by setting up a tunnel to our local Elasticsearch server using [ngrok](https://ngrok.com/).

__NOTE__: Expect this process to take around 20 minutes for each `wiki-XXX.jsonl` file loaded.
### Passage Retrieval Model
The model can either be trained or downloaded. First create the `models` directory inside the root directory:
```
mkdir models
```
Create a `relevancy_classification` model inside the models directory.
```
mkdir models\relevancy_classification
```
#### Downloading Premade Model
If you do not want to train the model, simply download the model [files](https://huggingface.co/theobaur/relevancy_classification_FEVER) and paste the model files into the `relevancy_classification` directory.
The directory structure should be as follows:
```
ESOTERIC
│
└───data
│
└───wiki-pages
│
└───models
		└───relevancy_classification
				│ config.json
				│ model.safetensors
				│ relevancy_classification_20000.json
				│ special_tokens_map.json
				│ tokenizer_config.json
				│ tokenizer.json
				│ vocab.txt
```

#### Training Passage Retrieval Model

Create `claims` directory inside `data` directory.
```
mkdir data\claims
```
Download [`shared_task_dev.jsonl`](https://fever.ai/download/fever/shared_task_dev.jsonl) and [`train.jsonl`](https://fever.ai/download/fever/train.jsonl) and copy into `claims` directory, which should appear as follows:
```
ESOTERIC
│
└───data
├───claims
│   │ shared_task_dev.jsonl
│   │ train.jsonl
│
└───wiki-pages
```
Run `analysis_db_loader.py` to build the analysis database:
```
py analysis_db_loader.py
```

Run `train.py`:
```
py train.py
```
Create a new dataset:
```
Do you want to create a new dataset? (y/n): 
y

Enter the number of claims to use: 
10000
```
Train a new model:
```
Do you want to train a new model? (y/n): 
y
```

## Usage

Activate virtual environment.

  

```python
venv\Scripts\activate
```

Run main script.

  

```python
py main.py
```

Enter a claim.

  

```
Enter claim: Telemundo is an English-language television network.
```
Results:

  

```
Base claim: Telemundo is an English-language television network.

Doc ID: List_of_Telemundo_affiliates_-LRB-by_U.S._state-RRB-
Document Score: 0.6735334292361913
        Start: 0 End: 154
        Sentence: Telemundo is an American Spanish language broadcast television television network owned by NBCUniversal which was launched in 1984 under the name NetSpan.
        Sentence Score: 0.8857951164245605

Doc ID: Telemundo
Document Score: 1
        Start: 0 End: 178
        Sentence: Telemundo ( [ teleˈmundo ] ) is an American Spanish-language terrestrial television network owned by Comcast through the NBCUniversal division NBCUniversal Telemundo Enterprises.
        Sentence Score: 0.870830774307251

Doc ID: List_of_Telemundo_affiliates_-LRB-table-RRB-
Document Score: 0.6741218730589448
        Start: 0 End: 169
        Sentence: Telemundo is an American broadcast television television network owned by the Telemundo Television Group division of NBCUniversal, which was launched in 1984 as NetSpan.
        Sentence Score: 0.8659212589263916

Doc ID: Telemundo_Internacional
Document Score: 0.6709062024044125
        Start: 0 End: 183
        Sentence: Telemundo Internacional is a Latin American basic cable and satellite television network that is owned by the NBCUniversal Hispanic Enterprises and Content subsidiary of NBCUniversal.
        Sentence Score: 0.8393296003341675

Doc ID: Noticias_Telemundo
Document Score: 0.6699378299834778
        Start: 0 End: 296
        Sentence: Noticias Telemundo ( [ noˈtisjas teleˈmundo ], Telemundo News ) is the news division of Telemundo, an American Spanish language broadcast television network that is owned by NBCUniversal Hispanic Enterprises and Content, a subsidiary of the NBCUniversal Television Group division of NBCUniversal.
        Sentence Score: 0.8276209831237793
        Start: 838 End: 1201
        Sentence: Noticias Telemundo maintains bureaus located at many of the network 's television stations across the United States ( particularly those owned by parent subsidiary Telemundo Station Group, that are owned-and-operated stations of the network ) and throughout Latin America, and uses video content from English language sister network NBC 's news division NBC News.
        Sentence Score: 0.6684804558753967
```

### Current System Analysis
You can analyse the recall, precision, and Oracle FEVER scores for both the document and passage retriever of the current version. In order to analyse the results you first need to run the system, producing a JSON file.

Run `analysis.py`

```
py analysis.py
```

  

Select option *r*

```
Would you like to run system or analyse results?
(r) Run, (a) Analyse
r
```
To analyse the results, do the following:

Run `analysis.py`

```
py analysis.py
```

Select option *a* to analyse results
```
Would you like to run system or analyse results?
(r) Run, (a) Analyse
r
```
### Legacy Analysis
You can analyse the performance of legacy systems using the legacy analysis.

Run `legacy.py`

```
py legacy.py
```
Choose a system version to analyse:
```
Enter the system name code 
1 - ESOTERIC 1
3 - ESOTERIC 3
3.1 - ESOTERIC 3.1
3.2 - ESOTERIC 3.2
3.3 - ESOTERIC 3.3
3.4 - ESOTERIC 3.4
3.5 - ESOTERIC 3.5
3.6 - ESOTERIC 3.6
3.7 - ESOTERIC 3.7
final - Final version
```
Ensure to load the necessary data first before running each legacy system by entering *1*:
```
Enter the action you want to perform
1 - Build 
2 - Run analysis
3 - Show Stats
1
```
Ensure to run the legacy system before showing any stats by entering *2*
```
Enter the action you want to perform
1 - Build 
2 - Run analysis
3 - Show Stats
2
```
Observe the recall, precision, F1, and Oracle FEVER scores by entering *3*
```
Enter the action you want to perform
1 - Build 
2 - Run analysis
3 - Show Stats
3
```
Results:
```
Document recall: X%
Document precision: X%
Document F1 score: X%
Passage recall: X%
Passage F1 score: X%
Combined recall: X%
Combined F1 score: X%
FEVER doc score: X%
FEVER passage score: X%
FEVER combined score: X%
Execution average: Xs
```
## License

  

[MIT](https://choosealicense.com/licenses/mit/)
