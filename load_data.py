""""
This script is used for preprocessing the data. 
"""

import pandas as pd 
import pdb 
import re
import spacy 

nlp = spacy.load("en_core_web_sm")



def remove_html_tags(text:str) -> str:
    """Removes the html tags from a string 
    From: https://stackoverflow.com/questions/3398852/using-python-remove-html-tags-formatting-from-a-string
    :param text: that needs to be cleaned 
    :returns: cleaned string without tags  
    """
    compiler = re.compile(r'<.*?>')
    return compiler.sub('', text)

def remove_code(text:str) -> str : 
    """Removes the text inside the <pre><code> tags
    :param text: string that needs to be cleaned 
    :returns: cleaned string without code and code tags. 
    """
    if "<pre><code>" in text: 
        cleaned_document = re.sub("<[pre][^>]<[code][^>] *>(.+?)</[code]></[pre]>", '', text)
        return cleaned_document
    else: 
        return text


def merge_title_and_body(list_with_titles, list_with_body): 
    articles = []
    for title, body in zip(list_with_titles, list_with_body):
        # lets only used the firstb couple of sentences.  
        # # tokenize the body 
        tokenized_body = [sent for sent in body.split("\n") if sent]
        first_sentences = tokenized_body[0:3]
        content = "{0} {1}".format(title, " ".join(first_sentences))
        # lemmatize the data here. 
        articles.append(content)
    return articles 
       
def process_data(questions_dataframe:pd.DataFrame, tags_dataframe:pd.DataFrame) -> pd.DataFrame:
    """Make a dataframe with: 
    - the questions and their respective tags
    - an additional column with title and body into one string
    :param questions_dataframe: pd.Dataframe with questions.
    :param tags_dataframe: pd.Dataframe with tags
    :returns merged_dataframe: pd.Dataframe with Title, Body, Title_and_Body as columns """ 
    ids = questions_dataframe["Id"].tolist()  
    tags_for_questions = []
    for question_id in ids: 
        question_tags = tags_dataframe[tags_dataframe["Id"]==question_id]
        tags_for_questions.append(question_tags["Tag"].tolist())

    
    merged_dataframe = pd.DataFrame.from_dict({"Tags": tags_for_questions, 
    "Title": questions_dataframe["Title"].tolist(), 
    "Body": questions_dataframe["Body"].tolist(),
    "Title_and_Body": merge_title_and_body(questions_dataframe["Title"].tolist(), questions_dataframe["Body"].tolist())})
    return merged_dataframe 
