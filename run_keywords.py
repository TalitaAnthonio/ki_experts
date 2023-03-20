"""
Usage: python3 run_keywords.py --PathToDir data 
"""

import pandas as pd 
from load_data import process_data, remove_html_tags, remove_code
from models import KeywordExtractor
from evaluation import compute_precision_at_k
import pdb 
import numpy as np 
import en_core_web_sm
from sklearn.model_selection import train_test_split  
import argparse
from typing import NoReturn


ap = argparse.ArgumentParser()
ap.add_argument("--PathToDir", help="Path To directory with questions.csv and tags.csv")
ap.parse_args() 
args = vars(ap.parse_args())
args = vars(ap.parse_args())

PathToDir = args["PathToDir"]

def run_baseline(questions_dataframe:pd.DataFrame,tags_dataframe:pd.DataFrame, dataset_size=50000, k=4) -> NoReturn: 
    """Runs the baseline for keyword extraction 
    :param questions_dataframe: pd.Dataframe with Body, Title
    :param tags_dataframe: pd.Dataframe with Ids and Tags. 
    :param dataset_size: int indicating the number of instances that should be used. Default = 50000
    :param k: int indicating the k value for precision at k."""
    # clean text body 
    questions_dataframe["Body"] = questions_dataframe["Body"].apply((lambda x: remove_code(x)))
    questions_dataframe["Body"] = questions_dataframe["Body"].apply((lambda x: remove_html_tags(x)))
   
    # split data into a train and development. 
    print("processing data ...") 
    dataframe = process_data(questions_dataframe.head(dataset_size), tags_dataframe)
    train, dev = train_test_split(dataframe, test_size=0.2)
    print("train size {0}".format(len(train)))
    print("development size {0}".format(len(dev)))
    
    print("computing keywords ..... ")
    predicted_keywords_for_questions = KeywordExtractor(k, train["Title"].tolist(), dev["Title"].tolist()).get_top_words()

    print("results on the development set .... ")
    precisions_at_k = []
    for question, predicted_keywords_for_question, golden_keywords in zip(dev["Title"].tolist(), predicted_keywords_for_questions, dev["Tags"].tolist()): 
        print("Question: {0}".format(question))
        # if there are no words predicted (rare case, but it can happen), just use the ones that are most commonly used in the dataset. 
        if predicted_keywords_for_question == []: 
            predicted_keywords_for_question = ["python", "django", "pandas", "numpy"]

        print("predicted words:{0}".format(predicted_keywords_for_question))
        print("golden keywords: {0}".format(golden_keywords))
        precision_at_k = compute_precision_at_k(predicted_keywords_for_question, golden_keywords,k=4)
        print("precision_at_k: {0}".format(precision_at_k))
        precisions_at_k.append(precision_at_k)
        print("---------------------------------")

    print("average precision at k {0}".format(np.mean(precisions_at_k)))


def main(): 
    questions_dataframe = pd.read_csv("./{0}/questions.csv".format(PathToDir), encoding="ISO-8859-1")
    tags_dataframe = pd.read_csv("./{0}/tags.csv".format(PathToDir), encoding="ISO-8859-1")

    # run the model 
    run_baseline(questions_dataframe, tags_dataframe)
    # add other models in the future. 



main()