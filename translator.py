from deep_translator import GoogleTranslator
import pandas as pd
import sys


def translate_reviews(dataset):
    df = pd.read_csv(dataset)
    df.head()

    df["News"] = df["News"].apply(
        lambda new: GoogleTranslator(source="es", target="en").translate(new)
    )
    df.head()

    df.to_csv("./datasets/dataset_translated.csv", index=False)


if __name__ == "__main__":
    translate_reviews(sys.argv[1])
