# nlp-automated-fact-checker
A command line application that uses natural language processing tools to retrieve relevant evidence sentences for a given input claim from the FEVER dataset.

This application unfinished and is still in development as of 21/02/2024.

## Installation
Clone repository.

```bash
git clone https://github.com/theobaur13/nlp-automated-fact-checker
```
Set up virtual environment.

```python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Create `data` directory inside main `nlp-automated-fact-checker` directory.
```
mkdir data
```
Create `wiki-pages` directory inside `data` directory.

```
mkdir \data\wiki-pages
```

Download `wiki-pages.zip` from [FEVER](https://fever.ai/download/fever/wiki-pages.zip).

Extract `wiki-pages.zip` into `\data\wiki-pages` so that the `wiki-pages` directory appears as follows:
```
nlp-automated-fact-checker  
│
└───data
    │
    └───wiki-pages
        │   wiki-001.jsonl
        │   wiki-002.jsonl
        │   ...
        │   wiki-109.jsonl

```
Run `db_loader.py`. The `batch_limit` argument specifies up to which wiki file to load up to (a `batch_limit` of 50 will load data from `wiki-001.jsonl` to `wiki-050.jsonl`). If `batch_limit` is left blank all wiki files will be loaded.

```python
py db_loader.py --batch_limit 109
```
Run `FAISS_loader.py`. The `batch_size` arguements specifies how many individual documents to load from the database and encode in one block. The maximum `batch_size` is 999, which takes the quickest to load, however is the most CPU intensive. If `batch_size` is left blank it will automatically be set to 999.

```python
py FAISS_loader.py --batch_size 999
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

```python
Enter claim: Luton Town won the English Premier League in 1990
```
Results:

```python
Base claim: luton town won the english premier league in 1990
Extracted question: What team won the english premier league in 1990?
Extracted question: What league did luton town win in 1990?
Extracted question: When did luton town win the english premier league?

Claim: luton town won the english premier league in 1990

Evidence Sentence: None
Evidence Document: The 1974 -- 75 season was the 89th season in the history of Luton Town Football Club . It was Luton Town 's 55th consecutive season in the Football League , and their 58th overall . It was also the team 's first season in the First Division since 1959 -- 60 , and their sixth overall . The season saw Luton narrowly relegated back to Division Two .   This article covers the period from 1 July 1974 to 30 June 1975 .
Score: 20.390697
Doc ID: 27505
Wiki URL: https://en.wikipedia.org/wiki/1974–75_Luton_Town_F.C._season

Claim: What team won the english premier league in 1990?

Evidence Sentence: None
Evidence Document: Statistics of Belgian League in season 1990/1991 .
Score: 28.267803
Doc ID: 34399
Wiki URL: https://en.wikipedia.org/wiki/1990–91_Belgian_First_Division

Evidence Sentence: None
Evidence Document: Events from 1995 in England
Score: 28.250763
Doc ID: 49984
Wiki URL: https://en.wikipedia.org/wiki/1995_in_England

Claim: What league did luton town win in 1990?

Evidence Sentence: None
Evidence Document: Statistics of Belgian League in season 1990/1991 .
Score: 27.751915
Doc ID: 34399
Wiki URL: https://en.wikipedia.org/wiki/1990–91_Belgian_First_Division

Evidence Sentence: None
Evidence Document: Statistics of Maltese Premier League in season 1970/1971 .
Score: 26.235252
Doc ID: 17795
Wiki URL: https://en.wikipedia.org/wiki/1970–71_Maltese_Premier_League

Claim: When did luton town win the english premier league?

Evidence Sentence: None
Evidence Document: The 1974 -- 75 season was the 89th season in the history of Luton Town Football Club . It was Luton Town 's 55th consecutive season in the Football League , and their 58th overall . It was also the team 's first season in the First Division since 1959 -- 60 , and their sixth overall . The season saw Luton narrowly relegated back to Division Two .   This article covers the period from 1 July 1974 to 30 June 1975 .
Score: 26.466125
Doc ID: 27505
Wiki URL: https://en.wikipedia.org/wiki/1974–75_Luton_Town_F.C._season
```
## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
