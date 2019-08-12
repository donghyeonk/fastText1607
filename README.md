# Bag of Tricks for Efficient Text Classification, fastText
Unofficial PyTorch Implementation of "Bag of Tricks for Efficient Text Classification", 2016, A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov (https://arxiv.org/pdf/1607.01759.pdf)

* The original model
    * fastText, h=10, bigram (See Table 1 of the paper)
* Dataset
    * We use [preprocessed data (See Xiang Zhang's folder)](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
        * [AG's news](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
        * Sogou, DBPedia, Yelp P., Yelp F. 
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
    $ python3 main.py --data_path ./data/dbpedia_csv/dbp.pkl
    ```

    * Sogou
    ```
    # Create a pickle file: data/sogou_news_csv/sogou.pkl
    $ python3 dataset.py --data_dir ./data/sogou_news_csv --pickle_name sogou.pkl --num_classes 5 --max_len 90064
    
    # Run
    $ python3 main.py --data_path ./data/sogou_news_csv/sogou.pkl --batch_size 1024 --lr 0.25
    ```
  
    * Yelp P.
    ```
    # Create a pickle file: data/yelp_review_polarity_csv/yelp_p.pkl
    $ python3 dataset.py --data_dir ./data/yelp_review_polarity_csv --pickle_name yelp_p.pkl --num_classes 2 --max_len 2955
    
    # Run
    $ python3 main.py --data_path ./data/yelp_review_polarity_csv/yelp_p.pkl --batch_size 1024 --lr 0.25
    ```
  
    * Yelp F.
    ```
    # Create a pickle file: data/yelp_review_full_csv/yelp_f.pkl
    $ python3 dataset.py --data_dir ./data/yelp_review_full_csv --pickle_name yelp_f.pkl --num_classes 5 --max_len 
    
    # Run
    $ python3 main.py --data_path ./data/yelp_review_full_csv/yelp_f.pkl
    ```
    
* Performance (accuracy %)

| Model                         | AG           | Sogou        | DBP          | Yelp P.      | Yelp F.      |
|:-----------------------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| fastText, h=10, bigram        | 92.5         | 96.8         | 98.6         | 95.7         | 63.9         |
| My implementation of fastText | 92.4 (Ep. 5) | 97.0 (Ep. 2) | 98.5 (Ep. 3) | 95.8 (Ep. 1) |              |


* Training time for an epoch (CPU)

|        | fastText | My implementation of fastText (Intel i7 8th gen.) | 
|:------:|:--------:|:----------:|
| AG     | 1s       |   11s      |
| Sogou  | 7s       | 1800s      |
| DBP    | 2s       |  100s      |
| Yelp P.| 3s       |  378s      |
| Yelp F.| 4s       |        |

* Embeddings are used instead of binary encoding (=multi-hot).
* Diff. from the paper
    * Adam optimizer instead of SGD (stochastic gradient descent)
    * No Hashing trick because the vocabulary size (1.4M, 6.7M, 3.4M) is less than 10M
    * No Hierarchical softmax because the number of classes is only 4, 14, 5
* Reference
    * https://github.com/poliglot/fasttext
    * https://github.com/bentrevett/pytorch-sentiment-analysis
