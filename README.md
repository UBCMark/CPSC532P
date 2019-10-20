## Downloading the preprocessed dataset

```
cd data/
wget https://drive.google.com/open?id=1WyEOyG-tj5vJhNvgRT6Cb_feKdej45HC
mv open\?id\=1WyEOyG-tj5vJhNvgRT6Cb_feKdej45HC finished.zip
unzip finished.zip
```

### Create embedding (GloVe) and mapping files

```
python word2vec.py finished/vocab
```
