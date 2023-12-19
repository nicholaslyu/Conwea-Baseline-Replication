# Conwea-Baseline-Replication
In this Q1 project, we aim to replicate two baseline results from the [ConWea](#conwea) paper: IR-TF-IDF and [Word2Vec](#word2vec).

# Packages
Please install the following dependencies
* Pickle
* Sklrean
* Pandas
* Numpy
* gensim
* nltk

# How to run?
* `python tf-idf.py` would run all tf-idf models (wouldn't take long), and the results would be saved.
* `python w2v.py` would run all w2v models with all parameters tuning process(take very long), and results would be saved.





### Citations
#### ConWea
```
@inproceedings{mekala-shang-2020-contextualized,
    title = "Contextualized Weak Supervision for Text Classification",
    author = "Mekala, Dheeraj  and
      Shang, Jingbo",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.30",
    pages = "323--333",
    abstract = "Weakly supervised text classification based on a few user-provided seed words has recently attracted much attention from researchers. Existing methods mainly generate pseudo-labels in a context-free manner (e.g., string matching), therefore, the ambiguous, context-dependent nature of human language has been long overlooked. In this paper, we propose a novel framework ConWea, providing contextualized weak supervision for text classification. Specifically, we leverage contextualized representations of word occurrences and seed word information to automatically differentiate multiple interpretations of the same word, and thus create a contextualized corpus. This contextualized corpus is further utilized to train the classifier and expand seed words in an iterative manner. This process not only adds new contextualized, highly label-indicative keywords but also disambiguates initial seed words, making our weak supervision fully contextualized. Extensive experiments and case studies on real-world datasets demonstrate the necessity and significant advantages of using contextualized weak supervision, especially when the class labels are fine-grained.",
}
```

#### Word2Vec
```
@article{word2vec,
    title={Efficient estimation of word representations in vector space},
    author={Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
    journal={arXiv preprint arXiv:1301.3781},
    year={2013}
}
```
