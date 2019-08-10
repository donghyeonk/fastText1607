# Bag of Tricks for Efficient Text Classification, fastText
Unofficial PyTorch Implementation of "Bag of Tricks for Efficient Text Classification", 2016, A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov (https://arxiv.org/pdf/1607.01759.pdf)

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
    
* Performance (accuracy %)

| Model                        | AG     | DBP |                          |
|:----------------------------:|:------:|:------:|:--------------------------------:|
| fastText, h=10, bigram       | 92.5   |  98.6  |                              |
| My implementation of fastText| 92.7   |  98.6  |        |


* Training time for an epoch (CPU)

|        | fastText | My implementation of fastText (Intel i7 8th gen.) | 
|:------:|:--------:|:----------:|
| AG     | 1s       |  11s       |
| DBP    | 7s       | 117s       |

* Embeddings are used instead of binary encoding (=multi-hot).
* Diff. from the paper
    * Adam optimizer instead of SGD (stochastic gradient descent)
    * No Hashing trick because the vocabulary size (1.4M) is less than 10M
    * No Hierarchical softmax because the number of classes is only 4
* Reference
    * https://github.com/poliglot/fasttext
    * https://github.com/bentrevett/pytorch-sentiment-analysis
