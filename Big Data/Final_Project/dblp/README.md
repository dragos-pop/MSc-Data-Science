
## Project - DBLP 

The goal in this project is to train a binary classifier to identify duplicate entries in a bibliography.

Submissions for this project will be shown on the [DBLP Leaderboard](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/dblp).


#### Training Data

The primary files contain detailed bibliographic data about research papers.

`dblp-*.csv`

|pauthor | peditor | ptitle | pyear | paddress | ppublisher | pseries | pid | pkey | ptype_id | pjournal_id | pbooktitle_id | pjournalfull_id | pbooktitlefull_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Additional details are contained in the following .json fiels.

`pbooktitle.json`

| pbooktitle_id | name |
|---|---|

`pbooktitlefull.json`

| pbooktitlefull_id | name |
|---|---|

`pjournal.json`

| pjournal_id | name |
|---|---|

`pjournalfull.json`

| pjournalfull_id | name |
|---|---|


`train.csv`

| key1 | key2 | label |
|---|---|---|

In addition, there are two files that contain information about directors and writers of the movies.


#### Validation & Test Data

We provide validation and test data as input for the submissions. This data has the same format as the training data, but does not contain the corresponding label.

`validation_hidden.csv` `test_hidden.csv`


| key1 | key2 |
|---|---|
