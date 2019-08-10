# Bag of Tricks for Efficient Text Classification, fastText
Unofficial PyTorch Implementation of "Bag of Tricks for Efficient Text Classification", 2016, A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov (https://arxiv.org/pdf/1607.01759.pdf)

* The original model
    * fastText, h=10, bigram (See Table 1 of the paper)
* Dataset
    * We use [preprocessed data (See Xiang Zhang's folder)](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
        * [AG's news](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
        * DBPedia
        * Sogou
        * ...
* Experiment

    ```
    # Download spacy en_core_web_lg model
    $ python3 -m spacy download en_core_web_lg --user
    
    # Download datasets (select mac or ubuntu)
    $ sh download_datasets_mac.sh
    ```

    * AG
    ```
    # Create a pickle file: data/ag_news_csv/ag.pkl
    $ python3 dataset.py --data_dir ./data/ag_news_csv --pickle_name ag.pkl --num_classes 4 --max_len 467
    
    # Run
    $ python3 main.py --data_path ./data/ag_news_csv/ag.pkl
    ```

    * DBPedia
    ```
    # Create a pickle file: data/dbpedia_csv/dbp.pkl
    $ python3 dataset.py --data_dir ./data/dbpedia_csv --pickle_name dbp.pkl --num_classes 14 --max_len 3013
    
    # Run
    $ python3 main.py --data_path ./data/dbpedia_csv/dbp.pkl --lr 0.1
    ```

    * Sogou
    ```
    # Create a pickle file: data/sogou_news_csv/sogou.pkl
    $ python3 dataset.py --data_dir ./data/sogou_news_csv --pickle_name sogou.pkl --num_classes 5 --max_len 500
    
    # Run
    $ python3 main.py --data_path ./data/sogou_news_csv/sogou.pkl --lr 0.1
    ```
    
* Performance (accuracy %)

| Model                        | AG     | DBP    | Sogou  |
|:----------------------------:|:------:|:------:|:------:|
| fastText, h=10, bigram       | 92.5   |  98.6  |        |
| My implementation of fastText| 92.6   |  98.6  |        |


* Training time for an epoch (CPU)

|        | fastText | My implementation of fastText (Intel i7 8th gen.) | 
|:------:|:--------:|:----------:|
| AG     | 1s       |  11s       |
| DBP    | 7s       | 117s       |
| Sogou  |          |            |

* Embeddings are used instead of binary encoding (=multi-hot).
* Diff. from the paper
    * Adam optimizer instead of SGD (stochastic gradient descent)
    * No Hashing trick because the vocabulary size (1.4M, 6.7M) is less than 10M
    * No Hierarchical softmax because the number of classes is only 4, 14
* Reference
    * https://github.com/poliglot/fasttext
    * https://github.com/bentrevett/pytorch-sentiment-analysis
