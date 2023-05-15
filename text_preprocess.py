import logging
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')

# debugging purpose
# i = 0


def preprocess(text: str, remove_stopwords: bool) -> str:
    """This utility function sanitizes a string by:
    - removing links
    - removing special characters
    - removing numbers
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text
    """

    # remove links
    text = re.sub(r"https?://\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower()
                  in stopwords.words("spanish")]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()

    # # debugging purpose
    # global i
    # i += 1
    # if i % 1000 == 0:
    #    logging.debug(f'{i} rows of the dataset preprocessed')

    return text


if __name__ == "__main__":
    df = pd.read_csv('./datasets/dataset.csv')
    # df = pd.read_csv('./datasets/test.csv')

    df["News"] = df["News"].apply(
        lambda x: preprocess(x, remove_stopwords=True))

    logging.info(df.head())

    df.to_csv('./datasets/dataset_cleaned.csv', index=False)
    # df.to_csv('./datasets/test_cleaned.csv', index = False)
