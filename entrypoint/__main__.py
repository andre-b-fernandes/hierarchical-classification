from argparse import ArgumentParser
from logging import info, getLogger, INFO
from lib.data import (
    load_training_data,
    parse_train_file,
    parse_validation_file,
    load_validation_data,
    CATEGORY_COL
)
from lib.ann import train as train_ann, predict as predict_ann
from lib.clustering import train as train_kmeans, predict as predict_means
from lib.metrics import hit_ratio

def main():

    getLogger().setLevel(INFO)

    parser = ArgumentParser(
        prog="Hierarchical Classification",
        description="Attempts to run a PoC doing a hierarchical classification of a public kaggle dataset"
    )
    parser.add_argument(
        "action",
        default="parse",
        help="Which action to take.",
        choices=["parse", "kmeans", "ann"]
    )
    args = parser.parse_args()

    if args.action == "parse":
        parse_validation_file()
        parse_train_file()
    else:
        train_df, train_embeddings, n_categories = load_training_data()
        validation_df, validation_embeddings,_ = load_validation_data()
        if args.action == "kmeans":
            kmeans = train_kmeans(matrix=train_embeddings, n_categories=n_categories)
            validation_df = predict_means(k_means=kmeans, df=validation_df, embeddings=validation_embeddings)
        elif args.action == "ann":
            categories = train_df[CATEGORY_COL].values
            ann, mapper = train_ann(embeddings=train_embeddings, categories=categories)
            validation_df = predict_ann(ann=ann, embeddings=validation_embeddings, df=validation_df, categories=mapper)

        ratio = hit_ratio(df=validation_df)
        info(f"Ratio: {ratio} %")


if __name__ == "__main__":
    main()
