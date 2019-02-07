# legal-linking
Linking of legal documents to other legal documents.

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
