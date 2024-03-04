# FEVERISH
FEVERISH - Fact Extraction and VERification with Intelligent Search Heuristics
A command line application that uses natural language processing tools to retrieve relevant evidence sentences for a given input claim from the FEVER dataset.

This application unfinished and is still in development as of 21/02/2024.

## Installation
Clone repository.

```bash
git clone https://github.com/theobaur13/FEVERISH
```
Set up virtual environment.

```python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Create `data` directory inside main `FEVERISH` directory.
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
FEVERISH  
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
Enter claim: Savages was exclusively a German film.
```
Results:

```
Base claim: Savages was exclusively a German film.

Doc ID: Savages_-LRB-2012_film-RRB-
Evidence Document: For the 2007 film , see The Savages   Savages is a 2012 American crime thriller film directed by Oliver Stone . It is based on the novel of the same name by Don Winslow . The screenplay was written by Shane Salerno , Stone , and Winslow . The film was released on July 6 , 2012 , and stars Taylor Kitsch , Blake Lively , Aaron Taylor-Johnson , Demian Bichir , Benicio del Toro , Salma Hayek , John Travolta and Emile Hirsch .
Document Score: 0.3672808750132442
Sentence: It is based on the novel of the same name by Don Winslow .
Sentence Score: 0.44645547532495655
Wiki URL: https://en.wikipedia.org/wiki/Savages_-LRB-2012_film-RRB-

Doc ID: 1976_in_Germany
Evidence Document: Events in the year 1976 in Germany .
Document Score: 3.8814392
Sentence: Events in the year 1976 in Germany .
Sentence Score: 0.4450946417141637
Wiki URL: https://en.wikipedia.org/wiki/1976_in_Germany

Doc ID: 1990_Deutsche_Tourenwagen_Meisterschaft
Evidence Document: The 1990 Deutsche Tourenwagen Meisterschaft was the seventh season of the Deutsche Tourenwagen Meisterschaft -LRB- German Touring Car Championship -RRB- . The season had twelve rounds with two races each .
Document Score: 4.102498
Sentence: The season had twelve rounds with two races each .
Sentence Score: 0.4379587061003252
Wiki URL: https://en.wikipedia.org/wiki/1990_Deutsche_Tourenwagen_Meisterschaft

Doc ID: 1979_in_Germany
Evidence Document: Events in the year 1979 in Germany .
Document Score: 4.8555717
Sentence: Events in the year 1979 in Germany .
Sentence Score: 0.4374303623229342
Wiki URL: https://en.wikipedia.org/wiki/1979_in_Germany
```
## Analysis
### Setup
Create `claims` directory inside `data` directory.

```
mkdir \data\claims
```

Download [`shared_task_dev.jsonl`](https://fever.ai/download/fever/shared_task_dev.jsonl) and [`train.jsonl`](https://fever.ai/download/fever/train.jsonl) and copy into `claims` directory, which should appear as follows:

```
FEVERISH  
│
└───data
    ├───claims
    │   │   shared_task_dev.jsonl
    │   │   train.jsonl
    │
    └───wiki-pages
        │   wiki-001.jsonl
        │   wiki-002.jsonl
        │   ...
        │   wiki-109.jsonl
```

Run `analysis.py`
```
py analysis.py
```

Select option *l*
```
Would you like to load claims into the database, run system, or analyse results?
(l) Load, (r) Run, (a) Analyse
l
```

### Usage
Run `analysis.py`
```
py analysis.py
```
Select option *r* to run system against database of prelabelled FEVER claims or selection option *a* to plot results or display statistics.


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
