# NLP_semantic_similarity
Natural Language Processing course assignment, project 16. Semantic Similarity of small text documents.

This repository contains three different semantic similarity scripts and a main script which functions as the CLI for running the scripts with custom data.

## Usage of main.py

### Test usage
```bash
python main.py test True
```
This runs all three scripts with the STSS-1311-Dataset.csv and reports the pearsons correlation of the result with the STSS-131 human judgement values

### Use usage
```bash
usage: main.py use [-h] (-c CSV | -s SENTENCES) -m {hierarchical,wu-palmer,idf} [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -c CSV, --csv CSV     Input csv file, must have sentences as the first and second element of each row
  -s SENTENCES, --sentences SENTENCES
                        semi-colon separated sentences, e.g. 'The weather is sunny;The moon is not present'
  -m {hierarchical,wu-palmer,idf}, --mode {hierarchical,wu-palmer,idf}
                        Choose the mode of similarity measurement
  -o OUTPUT, --output OUTPUT
                        Name of the output file, if empty results will be printed and returned
```
For example running a pair of sentences and saving the results to test_output.csv
```bash
python main.py use -s "this is an example;No samples are here" -o "test_output.csv" -m "idf"
```

## For different project specification requirements:

For the Inverse Document Frequency based script see:
```
wnSemSim.py
```
For the Noun Derivation based Wu-Palmer Semantic similarity script see:
```
wnNounSim.py
```
For the Hierarchical reasoning based script see:
```
wnhier.py
```