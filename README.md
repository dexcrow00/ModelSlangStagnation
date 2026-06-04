Project code for the LLM slang stagnation project. Divided into three main chunks
TODO: This needs a big update
1) CCAnalysis: Scripts used for in depth analysis of common crawl data to try and grasp slang representation in model training data. Almost everything here is no longer being used since switching to FineWeb, the exception being bert_slang_filter.py.
2) DataProcessingTools: Scripts for creating annotations used for fine tuning and validatio, as well as model eval and some visualization.
3) DatasetAnalysis: Scripts used during inital evaluation of datasets. Gets some metadata in a nice comparable way.
4) FineWebAnalysis: Script for pulling target word samples from FineWeb on HuggingFace, finetuned models, and list of actual target words pulled from FineWeb.
3) PromptingSlang: Scripts for prompting models trying to establish a dataset representitive of their bias toward using different parts of slang vocabulary.
