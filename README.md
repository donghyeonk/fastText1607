# Bag of Tricks for Efficient Text Classification, fastText
Unofficial PyTorch Implementation of "Bag of Tricks for Efficient Text Classification", 2016, A. Joulin et al. (https://arxiv.org/pdf/1607.01759.pdf)

* The original model
    * fastText, h=10, bigram (See Table 1 of the paper)
* Dataset
    * AG's news
        * More information can be found at [here](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)        
* Experiment
    * Download ag_news_csv.tar.gz from [here (Xiang Zhang's folder)](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
    * Make __data__ dir
    * Extract train.csv and test.csv to the __data__ directory

```
# Download spacy en_core_web_lg model
$ python3 -m spacy download en_core_web_lg --user

# Create a pickle file: data/ag.pkl
$ python3 dataset.py

# Run
$ python3 main.py
```
    
* Performance

| Model                                | Dataset | Accuracy (%) | Training time for an epoch (CPU) |
|:------------------------------------:|:-------:|:------------:|:--------------------------------:|
| The original: fastText, h=10, bigram | AG      | 92.5         | 1 s                              |
| This repo's model                    | AG      | 92.2         | 11 s (Intel i7 8th gen.)               |


* Embeddings are used instead of binary encoding (=multi-hot)
* Diff. from the paper
    * Adam optimizer instead of SGD (stochastic gradient descent)
    * No Hashing trick because the vocabulary size is less than 10M (1.4M)
    * No Hierarchical softmax because the number of classes is only 4
* Reference
    * https://github.com/poliglot/fasttext
    * https://github.com/bentrevett/pytorch-sentiment-analysis
    