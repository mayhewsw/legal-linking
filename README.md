# legal-linking
Linking of legal documents to other legal documents.

Requirements:
* allennlp (0.8.1)
* python (3.6+)

[Allennlp](https://github.com/allenai/allennlp/) is easy to install (preferably in a conda environment):
```bash
$ pip install allennlp
```

## Running the Code

```bash
$ allennlp train legal.json --include-package mylib -s tmp
```
Where `-s` refers to the serialization directory where trained model will be stored. 
Probably this should be something more interesting that `tmp`

Then, when you want to make predictions, run:
```bash
$ python interactive.py 
```

As of writing, it is not actually interactive.


## The Model
As of writing, the model is extremely simple. The input vector consists of a bag of 
words, where only those words in the intersection of the query paragraph and constitutional
paragraph are included. 

This bag of words vector is passed through a linear transform that ends up in two dimensions. 
These are the predictions of yes/no match.

The model is scored with F1, with the positive label as the label of interest. 

The interactive model runs a given query against all constitution paragraphs
and prints out those that have prediction=1. 

The data reader allows 3 times as many negative instances as positive.

## Data structure

### Cases
For the US Supreme Court (USSC) files, data are organized as a jsonlines file, with one JSON object (representing a single Supreme Court case) per line. The data structure for each individual case is as follows:

```
[
  {
    'text': "In 2008, the California Supreme Court held that...",
    'meta': {
             'doc_type': "opinion",
             'id': 0
             'source_url': "https://www.law.cornell.edu/supremecourt/text/12-144"
            }
     'matches': [["Fourteenth Amendment", "https://www.law.cornell.edu/constitution/amendmentxiv", '55']]
  }, 
  ...
]
```

Each line in this data structure corresponds to a paragraph in the source case. Variables are as follows:
* `text`: the text of the corresponding paragraph
* `meta`: metadata corresponding to the case. Currently, the implemented metadata fields are:
 * document `id` (integer; 0-indexed and sequentially organized) 
 * document type (possible fields are majority `opinion`, `dissent`ing opinion, `concur`ring opinion, and `per curiam` ("opinion of the court"))
 * the `source_url` from which the document was drawn
* `matches`: a list of hyperlinked matches to constitutional texts, if any are present in the given paragraph. Matches are formatted as a list with three items. The first item contains the in-text string match, and the second item contains the hyperlink that points to the constitutional text corresponding to that match. The third item is an index identifier which links to the corresponding constitutional text in the constitution file (see below for details).

For case data, two files are included: a `full` version, which contains the unedited text of each case, and a `stripped` version, which strips out the text strings corresponding to constitutional named entities. In the `stripped` version, removed text strings are replaced with `@@@` as a placeholder string. Each file is split into chunks of approximately 100

### Constitutional text
For the Constitutional text, data are organized as a single json file. The data structure is as follows:

```
{
  '55': {'text': 'Amendment XIV\nSection 1.\n\nAll persons born or naturalized in the United States...',
         'i': 55,
         'link': 'https://www.law.cornell.edu/constitution/articlexiv'}
  ...
}
```

Each key in this file corresponds to a hyperlinked entity match scraped from the Cornell website. These keys match the indices from the `matches` field in the case files.

## Writing

Overleaf is here: https://www.overleaf.com/9129219286mfwdpctnyfzz
