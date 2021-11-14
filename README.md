# Hierarchical Classification

This repository implements a proof of concept of using `k-means-clustering` for hierarchical multiclass classification and comparing
its results to a standard artificial neural network implementation.

## Dataset considerations

The original dataset was downloaded from [this](https://www.kaggle.com/kashnitsky/hierarchical-text-classification) Kaggle link and was divided
between test and validation sets. The only contents which were not present there were the parsed versions of the 40k and 10k test and validation
sets.

Both implementations for design reasons, can only output categories which were present in the train dataset (I know, I know...)
so I evaluated the impact this could take in the final metric calculation using this procedure: 

```python
df1 = pd.read_csv("data/amazon/train_parsed_40k.csv")
df2 = pd.read_csv("data/amazon/val_parsed_10k.csv")
cats1 = set(df1['CATEGORY'].unique())
cats2 = set(df2['CATEGORY'].unique())
missing = cats2 - cats1
missing
>> {'coatings batters', 'hydrometers', 'chocolate covered nuts', 'breeding tanks', 'flying toys', 'dried fruit', 'exercise wheels', 'shampoo', 'lamb'}
len(df2[df2['CATEGORY'].isin(missing)])
>> 11
```

Only 11 elements from a total of 10k rows, which is not a problem.

## Dependencies

This project uses `poetry` as a dependency manager. Although not mandatory, I think it's better to use Anaconda to create
an isolated virtual environment.

You can install `poetry` using `pip`, with `pip install poetry`.

After that you can run `poetry install`.

## How to run.

You can run this project by:

1. `python -m entrypoint parse` - To re-parse the validation and test dataset csv files.
2. `python -m entrypoint kmeans` - To run the KNN model and predict the categories, calculating the hit ratio.
3. `python -m entrypoint ann` - To run the ANN model and predict the categories, calculating the hit ratio.
