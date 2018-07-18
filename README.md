# Bag of Tricks, fastText
Unofficial Implementation of "Bag of Tricks for Efficient Text Classification", 2016, Armand Joulin et al. (https://arxiv.org/pdf/1607.01759.pdf)

* Model
    * fastText, h=10, bigram (See Table 1 of the paper)
* Dataset
    * AG's news
        * Download ag_news_csv.tar.gz from [here (Xiang Zhang's folder)](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
        * Extract train.csv and test.csv to data/
        * More information can be found at [here](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)        
* Learning
```
# Create a pickle file: data/ag.pkl
$ python3 dataset.py

# Run
$ python3 main.py
```
    
* Performance

| Model                  | Dataset | Accuracy (%) |
|:----------------------:|:-------:|:------------:|
| fastText, h=10, bigram | AG      | 92.5         |
| This repo.             | AG      | 91.0         |

* Diff. from the paper
    * Embeddings instead of 10M bins
    * Adam optimizer instead of SGD (stochastic gradient descent)
* Reference
    * https://github.com/poliglot/fasttext
    * https://github.com/bentrevett/pytorch-sentiment-analysis
    