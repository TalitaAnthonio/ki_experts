from sklearn.feature_extraction.text import TfidfVectorizer
import pdb 
import pandas as pd 


def load_stopwords(): 
    """Get a list of stopwords downloaded from the internet
    :returns: list with stopwords """
    with open("./data/stopwords.txt", "r") as file_in: 
        stopwords = file_in.readlines()
    return [stopword.strip() for stopword in stopwords]


def get_top_words(matrix:list, vocab:list, k:int) -> list: 
    """Get the top k words from the document_term_matrix
    :param matrix: list with document_term matrix. Length is equal to the number of documents and each element is of length vocab. 
    :param vocab: list with extracted features 
    :param k: int indicating the top predictions we have to consider. 
    :returns top_words_per_document: list with the top-k words."""
    top_words_per_document = []
    for row in matrix: 
        # get the indices of the ones that are not zero. 
        indexes = [index for index, value in enumerate(row,0) if value > 0.0]
        scores = [row[pos] for pos in indexes]
        vocabulary = [vocab[pos] for pos in indexes]

        # get the top words 
        df = pd.DataFrame.from_dict({"Scores": scores, "Words": vocabulary})
        df = df.sort_values("Scores", ascending=False)
        top_words = df.head(k)
        top_words_per_document.append(top_words["Words"].tolist())
    return top_words_per_document
        

class KeywordExtractor: 

    def __init__(self, k, train_docs, dev_docs):
        self.k = k 
        self.train_docs = train_docs 
        self.dev_docs = dev_docs 
        self.stopwords = load_stopwords()
        self.vectorizer = TfidfVectorizer(stop_words=self.stopwords, max_df=0.9, use_idf=True, lowercase=True,max_features=500)
        

        self.fitted_vector = self.vectorizer.fit_transform(train_docs)
        self.doc_matrix_dev = self.vectorizer.transform(dev_docs).toarray().tolist()
        
        self.feature_names = self.vectorizer.get_feature_names_out()

    def get_top_words(self): 
        top_words = get_top_words(self.doc_matrix_dev,self.feature_names, self.k)
        return top_words 

