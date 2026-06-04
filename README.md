Project code for the LLM slang stagnation project. Divided into three main chunks
TODO: This needs a big update
1) CCAnalysis: Scripts used for in depth analysis of common crawl data to try and grasp slang representation in model training data. Almost everything here is no longer being used since switching to FineWeb, the exception being bert_slang_filter.py.
2) DataProcessingTools: Scripts for creating annotations used for fine tuning and validatio, as well as model eval and some visualization.
3) DatasetAnalysis: Scripts used during inital evaluation of datasets. Gets some metadata in a nice comparable way.
4) FineWebAnalysis: Script for pulling target word samples from FineWeb on HuggingFace, finetuned models, and list of actual target words pulled from FineWeb.
3) PromptingSlang: Scripts for prompting models trying to establish a dataset representitive of their bias toward using different parts of slang vocabulary.


target_words.txt is a list of target slang that has been manually curated from Urban Dictionary. Due to computational and practical (linguistic) limitations, we operate on a subset of all slang available on the internet. The manual curation attempted to create a set of mixed millenial (2010-2020) and gen-z (2020+) internet slang. The definition being used here is slang as an indicator of in-group belonging to a generation (See Citera 2020 - "Differences in emotional word use across generations in the united states", Earl - 1972 "In semantic influence and concept attainment of slang and its effects on parents' and teenagers' linguistic interaction", Barbieri 2008 - "Patterns of age based linguistic variation in american English").
Note that the manual curation has also avoided hate speech or other highly offensive "slang", a lot of internet slang is mildly offensive though...
