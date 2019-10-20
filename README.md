## Downloading the preprocessed dataset

```
git clone https://github.com/UBCMark/CPSC532P.git
cd data/
wget https://drive.google.com/open?id=1WyEOyG-tj5vJhNvgRT6Cb_feKdej45HC
unzip finished.zip
```

### Create embedding (GloVe) and mapping files

```
python word2vec.py finished/vocab
```
