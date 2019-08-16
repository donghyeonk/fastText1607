# Bag of Tricks for Efficient Text Classification, fastText
Unofficial PyTorch Implementation of "Bag of Tricks for Efficient Text Classification", 2016, A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov (https://arxiv.org/pdf/1607.01759.pdf)

* The original model
    * fastText, h=10, bigram (See Table 1 of the paper)
* Dataset
    * We use [preprocessed data (See Xiang Zhang's folder)](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
        * [AG's news](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html), Sogou, DBpedia, Yelp P., Yelp F., Yahoo A., Amazon F., Amazon P.
* Experiment
    ```
    # Download a spacy "en_core_web_lg" model
    $ python3 -m spacy download en_core_web_lg --user
    
    # Download datasets (select your os (mac or ubuntu))
    $ sh download_datasets_mac.sh
    ```

    * AG
    ```
    # Create a pickle file: data/ag_news_csv/ag.pkl
    $ python3 dataset.py --data_dir ./data/ag_news_csv --pickle_name ag.pkl --num_classes 4 --max_len 467
    
    # Run
    $ python3 main.py --data_path ./data/ag_news_csv/ag.pkl --batch_size 2048 --lr 0.5 --log_interval 20
    ```
  
    * Sogou
    ```
    # Create a pickle file: data/sogou_news_csv/sogou.pkl
    $ python3 dataset.py --data_dir ./data/sogou_news_csv --pickle_name sogou.pkl --num_classes 5 --max_len 90064
    
    # Run
    $ python3 main.py --data_path ./data/sogou_news_csv/sogou.pkl --batch_size 1024 --lr 0.1 --log_interval 40
    ```

    * DBpedia
    ```
    # Create a pickle file: data/dbpedia_csv/dbp.pkl
    $ python3 dataset.py --data_dir ./data/dbpedia_csv --pickle_name dbp.pkl --num_classes 14 --max_len 3013
    
    # Run
    $ python3 main.py --data_path ./data/dbpedia_csv/dbp.pkl --batch_size 2048 --lr 0.1 --log_interval 20
    ```

    * Yelp P.
    ```
    # Create a pickle file: data/yelp_review_polarity_csv/yelp_p.pkl
    $ python3 dataset.py --data_dir ./data/yelp_review_polarity_csv --pickle_name yelp_p.pkl --num_classes 2 --max_len 2955
    
    # Run
    $ python3 main.py --data_path ./data/yelp_review_polarity_csv/yelp_p.pkl --batch_size 1024 --lr 0.1 --log_interval 40
    ```

    * Yelp F.
    ```
    # Create a pickle file: data/yelp_review_full_csv/yelp_f.pkl
    $ python3 dataset.py --data_dir ./data/yelp_review_full_csv --pickle_name yelp_f.pkl --num_classes 5 --max_len 2955
    
    # Run
    $ python3 main.py --data_path ./data/yelp_review_full_csv/yelp_f.pkl --batch_size 1024 --lr 0.05 --log_interval 40
    ```

    * Yahoo A.
    ```
    # Create a pickle file: data/yahoo_answers_csv/yahoo_a.pkl
    $ python3 dataset.py --data_dir ./data/yahoo_answers_csv --pickle_name yahoo_a.pkl --num_classes 10 --max_len 8024
    
    # Run
    $ python3 main.py --data_path ./data/yahoo_answers_csv/yahoo_a.pkl --batch_size 1024 --lr 0.05 --log_interval 40
    ```

    * Amazon F.
    ```
    # Create a pickle file: data/amazon_review_full_csv/amazon_f.pkl
    $ python3 dataset.py --data_dir ./data/amazon_review_full_csv --pickle_name amazon_f.pkl --num_classes 5 --max_len 1214
    
    # Run
    $ python3 main.py --data_path ./data/amazon_review_full_csv/amazon_f.pkl --batch_size 4096 --lr 0.25 --log_interval 10
    ```

    * Amazon P.
    ```
    # Create a pickle file: data/amazon_review_polarity_csv/amazon_p.pkl
    $ python3 dataset.py --data_dir ./data/amazon_review_polarity_csv --pickle_name amazon_p.pkl --num_classes 2 --max_len 1318
    
    # Run
    $ python3 main.py --data_path ./data/amazon_review_polarity_csv/yahoo_a.pkl --batch_size 4096 --lr 0.25 --log_interval 10
    ```

* Performance (accuracy %)

| Model                         | AG           | Sogou        | DBpedia      | Yelp P.      | Yelp F.      | Yahoo A.      | Amazon F.      | Amazon P.      |
|:-----------------------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:-------------:|:--------------:|:--------------:|
| fastText, h=10, bigram        | 92.5         | 96.8         | __98.6__         | 95.7         | __63.9__         | 72.3         | __60.2__         | __94.6__         |
| My implementation of fastText | __92.6__ (Ep. 3) | __97.1__ (Ep. 5) | 98.1 (Ep. 4) | 95.7 (Ep. 1) | 63.5 (Ep. 1) | __72.5__ (Ep. 1) | 57.7 (Ep. 1) | 94.3 (Ep. 1) |


* Training time for an epoch (CPU)

|        | fastText | My implementation of fastText (Intel i7 8th gen.) | 
|:------:|:--------:|:----------:|
| AG     | 1s       |   12s      |
| Sogou  | 7s       | 30m      |
| DBpedia| 2s       |  3m      |
| Yelp P.| 3s       |  7m      |
| Yelp F.| 4s       |  8m      |
| Yahoo A.| 5s       |  24m      |
| Amazon F.| 9s       |  14m      |
| Amazon P.| 10s       |  15m      |

* Dictionary size & data size

|Dataset   | Size      | Is Hashing Trick needed? | # train | # test | # classes |
|:--------:|:---------:|:---------:|:----:|:---:|:---:| 
| AG       |  1.4M     | No        | 120K |7.6K | 4| 
| Sogou    |  3.4M     | No        | 450K | 60K | 5|
| DBP      |  6.6M     | No        | 560K | 70K |14|
| Yelp P.  |  6.4M     | No        | 560K | 38K | 2|
| Yelp F.  |  7.1M     | No        | 650K | 50K | 5|
| Yahoo A. | 17.9M     | Yes       | 1.4M | 60K |10|
| Amazon F.| 21.7M     | Yes       | 3M   |650K | 5|
| Amazon P.| 24.3M     | Yes       | 3.6M |400K | 2|


* Embeddings are used instead of binary encoding (=multi-hot).
* Diff. from the paper
    * Adam optimizer instead of SGD (stochastic gradient descent)
    * No Hashing Trick for AG, Sogou, DBP, Yelp P. and Yelp F. because the vocabulary size (1.4M ~ 7.1M) is less than 10M
    * No Hierarchical softmax because the number of classes is only 2 ~ 14
* Reference
    * https://github.com/poliglot/fasttext
    * https://github.com/bentrevett/pytorch-sentiment-analysis
