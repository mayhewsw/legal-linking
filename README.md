# Legal Linking

The repository for [Legal Linking: Citation Resolution and Suggestion in Constitutional Law](https://cogcomp.org/page/publication_view/873), published at the [Natural Legal Language Processing workshop](https://sites.google.com/view/nllp/) at NAACL 2019.

Our trained models are available in the releases section, and the data is available in this repository.

Requirements:
* allennlp (0.8.1)
* python (3.6+)
* scikit-learn

[AllenNLP](https://github.com/allenai/allennlp/) is easy to install (preferably in a conda environment):
```bash
$ pip install allennlp
```

## Running the Code

```bash
$ allennlp train legal.json --include-package mylib -s tmp
```
Where `-s` refers to the serialization directory where trained model will be stored. 
Probably this should be something more interesting that `tmp`

NOTE: for some reason, this creates a `bert.txt` vocab file under the `model/vocabulary` folder. This 
will cause issues with OOV tokens (I haven't figured why this happens). To get around this, just go into
that folder and do `cp bert.txt .bert.txt`. The vocabulary reader will ignore hidden files. You will then
have to recreate the model file (in the serialization directory):

```bash
$ cp best.th weights.th
$ tar czvf model.tar.gz weights.th vocabulary/ config.json
```


## Preparing Data

All data is originally stored in JSON files, and the mkdata.sh script converts into
the lines format. The format that the datareader expects is `line<TAB>label1,label2`.

You can give `mkdata.sh` a number as argument which will limit the number of examples (usually
you don't want to do this, probably only for testing).

It takes a while (15min) to create a file called `all_lines_labeled`, so if this already exists, `mkdata.sh`
won't make it again. Be careful though: perhaps you need to update the file. Consider deleting
it first.

The `data/stats.py` script will give some stats on the data.

## Get Rule-based Results
Look at `score_all.sh` for an example. You want to use `tag_data.py` but with the `-d` flag (destructive)
which removes all prior annotations before adding new ones. If you don't use this flag, you 
will get an extremely high score (close to 100%).

## Get Linear Model Results
Make sure to set the training data and output parameters in the file, then run

```bash
$ python linear_model.py
```

## Get NN Model Results
Run this:

```bash
$ allennlp predict legal-bert-model.tar.gz data/validation/all_validation --include-package mylib --cuda-device 0 --use-dataset-reader --output bert.txt --predictor legal_predictor --silent
```

## How to score result files

To score a result file against a gold file, run `score.py` on the predictions and 
gold line files.

```bash
$ python score.py --gold data/validation/all_validation --pred results/linear.txt
```


## Running the Demo
Make sure the model is in the right place (if trained with GPU, you need to ask for CUDA), and 
run `./demo.sh`. Very easy.

Static files for the demo are kept in `demo_files`.

## The Models
See the paper for description of the models.

## Training on the GPU
To train on the GPU, just change the `device` parameter in `legal.json` to 0. You probably also want to change the `batch_size`. I typically copy `legal.json` to `legal_gpu.json` and
modify the `legal_gpu.json` file. 

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
