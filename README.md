# fastText
(Work in progress) Unofficial Implementation of "Bag of Tricks for Efficient Text Classification", 2016, Armand Joulin et al. (https://arxiv.org/pdf/1607.01759.pdf)

* Model
    * fastText, h=10, bigram (See Table 1 of the paper)
* Dataset
    * AG
        * Decompress [AG's corpus file (118 MB)](https://www.di.unipi.it/~gulli/newsSpace.bz2) to data/        
        * Known issues
            * A line breaker (\N)
* Execution
```
// create a pickle file: data/ag.pkl
$ python3 dataset.py

// run
$ python3 main.py
```