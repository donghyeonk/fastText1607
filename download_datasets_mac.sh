#!/bin/bash

mkdir data
cd data

# brew install gnu-sed

echo "Downloading AG ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms" -O ag_news_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz ag_news_csv.tar.gz

echo "Downloading Sogou ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE" -O sogou_news_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz sogou_news_csv.tar.gz

echo "Downloading DBP ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k" -O dbpedia_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz dbpedia_csv.tar.gz

echo "Downloading Yelp Polarity ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg" -O yelp_review_polarity_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz yelp_review_polarity_csv.tar.gz

echo "Downloading Yelp Full ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0" -O yelp_review_full_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz yelp_review_full_csv.tar.gz

echo "Downloading Yahoo Answers ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU" -O yahoo_answers_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz yahoo_answers_csv.tar.gz

echo "Downloading Amazon Polarity ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM" -O amazon_review_polarity_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz amazon_review_polarity_csv.tar.gz

echo "Downloading Amazon Full ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA" -O amazon_review_full_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz amazon_review_full_csv.tar.gz
