#!/bin/bash

mkdir data
cd data

echo "Downloading AG ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms" -O ag_news_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz ag_news_csv.tar.gz

echo "Downloading DBP ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k" -O dbpedia_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz dbpedia_csv.tar.gz

echo "Downloading Yelp P. ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg" -O yelp_review_polarity_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz yelp_review_polarity_csv.tar.gz

echo "Downloading Sogou ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE" -O sogou_news_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz sogou_news_csv.tar.gz