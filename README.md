# fastText
(Work in progress) Unofficial Implementation of "Bag of Tricks for Efficient Text Classification", 2016, Armand Joulin et al. (https://arxiv.org/pdf/1607.01759.pdf)

* Model
    * fastText, h=10, bigram (See Table 1 of the paper)
* Dataset
    * AG's news
        * Download [AG's corpus file (118 MB)](https://www.di.unipi.it/~gulli/newsSpace.bz2) and decompress it to data/
        * More information can be found at [here](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)        
* Learning
```
# Create a pickle file: data/ag.pkl
$ python3 dataset.py

# Run
$ python3 main.py
```