Prototype AI model for keyword extraction from questions. The current model is a baseline and is used as a 
starting point to explore the feasability, as indicated in the case study. For future work, I suggest to experiment with a BERT-based model and to experiment with different evaluation metrics. 

# Installation 

1. Install packages using pip. 

```
pip install requirements
```

# Datasets and resources 
The dataset is a collection of questions/titles, explanations and their keywords. 
1. Download the dataset using the dropbox link. 
2. Unzip dataset and put the files in a folder. 

Comments: at the moment I am only using a part of the dataset, given my limited computational resources. 

# Running 

Use the following command to run the script: 

```
python3 run_keywords --DataDir 
```

--DataDir is the directory with the data files. 

Comment: In a use-case without having keywords available, the input to the model would be a file with questions/texts. 

# Approach 
This section describes how I tried to approach the task. 

## Step 1: understand the task 
- The task is to develop a propotype AI model that performs keyword prediction based on the headings and/or questions. 
- The goal is to investigate the feasability using an available dataset. 
- Use git and work in the agile development team: code should be readable.  

## Step 2: explore the dataset 
1. explore the structure: The dataset contains two tsv-files: ``questions.tsv`` and ``tags.tsv``. The tags and texts are connected through an ID. 
2. explore the size: The size of the dataset is 199,51745. Given the computational capacity, I will work with a subset of the dataset. 
3. explore the content: 

Inspect the keywords using simple linux commands: 

````
cat tags.csv| cut -f2 -d',' | sort | uniq -c | sort -nr  | head -n20

````

Which shows the following: 

```` 
607283 python
62818 django
34616 python-2.7
26854 pandas
26814 python-3.x
25848 numpy
18951 list
16521 matplotlib
14047 regex
13413 dictionary
10766 tkinter
10616 string
10488 flask
10286 google-app-engine
9323 csv
9170 arrays
8023 json
7529 mysql
7121 linux
7118 html
````

This shows that: 
- the keywords are unigrams 
- majority of the keywords is python. 

- Looking at the questions.tsv and tags.tsv file also indicates that:
1. the title is not always a question 
2. the body text needs to be cleaned: programming code, html-tags  
3. not all the rows are necessary for this task 
4. there are keywords that do not come from the question, so using the body seems a good idea. 
5. the exact number of keywords that needs to be predicted differs per question. 

### Step 3: Brainstorming
1. My experience with keyword extraction: did not work on the task, but did experiments: 
- with tf-idf 
- computed the semantic similarity/difference between words using BERT embeddings + cosine similarity 
- used k-means clustering to find a set of diverse words for a given document. 

2. What would state-of-the-art models be for keyword extraction? 
Despite the popularity of BERT, tf-idf still seems to be widely used for keyword extraction. 

## Step 4: develop a baseline 
Given the available time, I decide to start with implementing a baseline. 

### Decide on Input 
- Since some keywords do not occur in the question, we need to use the text. Using the full texts might lead to having many words that are irrelevant. I know from text analysis that typically the first sentences are an explanation to the question. I would predict that these sentences are also interesting for determining keytwords. 

### Decide on baseline model: TF-IDF  
Since there is a uge amount of data available, I decide to use a method that can make use of this data, instead of 
using a method that compute the similarity of a question directly. From my experience, tf-idf is still a good 
method for keyword extraction. 

#### TF-IDF 
TF-IDF is a statistical measure that can be used to represent documents in a vector space. It was invented for information retrieval and can be used to 
find important words in a document. 
- term frequency: count of a word in a document. 
- inverse document frequency: each word is divided by the number of documents it appears in. The more common a word is in a set of documents, the closer the number will approach 0. The log is taken to 
- The score is computed by multiplied tf * idf. 

To compute the inverse document frequency, it is important to have a big collection of documents, which we have. 


### Evaluation 
- Extract the top k keywords from the vectorizer. I used k=4, given the average number of 
keywords that needs to be predicted. 
- Identify if the predicted keywords are in the golden keywords. 
- Current precision at k (for 50,000 instances): 0.35


### completed that did not work 
- Tried to use only lemmas, but the procedure is extremely slow and the precision was barely increased (still 0.31.)

## Step 5: future plans 
My future plans would be to: 
1. Use BERT to compute the cosine similarity between candidate keywords (for instance, just nouns). Compare the results to the baseline quantitatively, qualitatively and preferably also in a use-case. 
-   I did not use BERT for computing similarities, because I wanted to make use of the data that is provided and with tf-idf I could make use of the data. 
2. Experiment with different evaluation methods: exact match is not the best way to evaluate the capacity of a model. 
3. Extrinsic Evaluation: how fast is this model? How effective is it? 
4. Explore the usage of other datasets, if available. 