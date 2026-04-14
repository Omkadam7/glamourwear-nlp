#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing

# 
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: Below are the main libraries I used in my assignment:
# * pandas
# * re
# * nltk
# * nltk.tokenize
# * nltk.FreqDist
# * itertools.chain
# 
# ## Introduction
# 
# In this assignment, I worked on pre-processing a dataset related to women's clothing reviews from an e-commerce platform. The dataset contains multiple features, including the Clothing ID, Age, Review Title, and Review Text. Additionally, the dataset includes customer ratings, recommendations, and various other attributes related to product feedback. The main goal of this task was to prepare the Review Text column for further text analysis, such as sentiment analysis or modeling.
# 
# My primary focus was on text cleaning and preparation, where I applied several pre-processing techniques to the review text. This process included tokenizing the text, converting it into lowercase, and removing short words, stopwords, and words with minimal frequency. I used Python's `apply()` function to apply these transformations across the dataset, as mentioned in `[2] Amani, “How can I use the apply() function for a single column?” from Stack Overflow.` Each step was carefully executed and documented to ensure the dataset was cleaned properly for future analysis. I created a DataFrame at every step of the process to visually compare the changes and see how the data evolved after each cleaning stage.
# 
# Finally, I created a cleaned version of the dataset and saved it as `processed.csv`. I also built a vocabulary of the cleaned review text, ensuring that it only contained meaningful and frequently occurring words. The vocabulary was saved as `vocab.txt`, sorted alphabetically and indexed from zero, to facilitate further text analysis.

# ## Importing libraries 

# In[1]:


import pandas as pd                            # for data manipulation
import re                                      # for regular expressions
import nltk                                    # for natural language processing tasks
from nltk.tokenize import RegexpTokenizer      # tokenizer for splitting text based on a specific pattern
from nltk import FreqDist                      # for calculating frequency distributions of words
from itertools import chain                    # lists into a single list


# ## 1.1 Examining and loading data
# 
# #### **Dataset**
# 
# The dataset that we are going to use in this exercise comes from a `Women’s Clothing E-Commerce platform`. It contains `19662 rows` of real customer reviews. To maintain privacy, the data has been anonymized, and any direct references to the retailer in the reviews have been replaced with the word “retailer.”
# 
# Before we start, we will double-check that the `assignment3.csv` file is in the same directory as this Jupyter Notebook. This will ensure that we can load and work with the data smoothly.
# 
# The dataset includes the following features:
# - `Clothing ID`
# - `Age`
# - `Title`
# - `Review Text`
# - `Rating`
# - `Recommended IND`
# - `Positive Feedback Count`
# - `Division Name`
# - `Department Name`
# - `Class Name`
# 
#     We will be using just the `Review Text` column from this dataset to perform text analysis.
# #### **Loading Data**
# Before doing any pre-processing, we first need to load the data into a usable format. The reviews data is stored in a CSV file, and the best way to handle this type of file is by using the `pandas` library in Python. Pandas allows us to load the data into a `DataFrame`, which is a table-like structure where each row represents a review, and each column represents a feature like the review text or the rating.
# 
# We will use the `pandas.read_csv()` function to load the CSV file and inspect the first few rows to understand the structure of the data. This will allow us to see important details like the review text, the rating provided by customers, and whether they recommend the product or not.

# In[2]:


data_df = pd.read_csv('assignment3.csv')   # loading the data into a pandas DataFrame

data_df.head()                            


# ## 1.2 Pre-processing data
# Once the clothing review data has been loaded, we need to perform several pre-processing steps to prepare the data for analysis. These pre-processing steps are designed to clean the text and make it ready for further tasks like sentiment analysis or building machine learning models.
# 
# Here are the sub-tasks that we will perform one by one:
# 1. Extract Information About the Reviews
# 2. Tokenization
# 3. Convert All Words to Lowercase
# 4. Remove Words Shorter Than 2 Characters
# 5. Remove Stopwords
# 6. Remove Words That Appear Only Once
# 7. Remove the Top 20 Most Frequent Words
# 8. Save the Processed Data
# 9. Build a Vocabulary
# 
# ### 1.2.1 Extract the Review Text
# We are working with a dataset that includes columns such as `Clothing ID`, `Age`, `Title`, `Review Text`, `Rating`, `Recommended IND`, `Positive Feedback Count`, `Division Name`, `Department Name`, and `Class Name`. Our focus will be on the `Review Text` column, as this contains the actual text data that we will analyze. The information in this column needs to be processed step by step.

# In[3]:


reviews = data_df['Review Text'] # extracting the 'Review Text' column here


# ### 1.2.2 Tokenization
# Tokenization is a fundamental step in text processing where we break down a string of text (in this case, customer reviews) into smaller parts called "tokens." These tokens usually represent individual words or meaningful pieces of text.
# 
# Example:
# 
# If we have the review: `"I love this dress,"` tokenization will give us: `['I', 'love', 'this', 'dress']`
# 
# We are using a regular expression (regex) for tokenization because it gives us more control over how we split the text. For example, we want to handle situations like:
# - `Hyphenated words` (e.g., "well-known") should be kept together.
# - `Words with apostrophes` (e.g., "it's", "don't") should also be kept together.
# 
# By using the `RegexpTokenizer` from NLTK library, we can define a custom pattern to specify how to split the text. The pattern we use is `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"`. Let's break this down:
# - `[a-zA-Z]+`: Matches one or more letters (both uppercase and lowercase).
# - `(?:[-'][a-zA-Z]+)?`: This part handles words with hyphens or apostrophes. It ensures that words like "it's" and "well-known" are treated as single tokens.
# - The `apply()` function applies the `tokenize_review()` function to each row in the `Review Text` column, creating a new column called `Tokenized Review`.
# 
# By using this approach, we can ensure that the text is broken down into manageable, meaningful tokens for further analysis.

# In[4]:


tokenizer = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")  # initializing the tokenizer with a custom regex pattern to handle words, hyphens, and apostrophes

def tokenize_review(review):                                 # defining a function to tokenize the review text
    return tokenizer.tokenize(str(review)) 

data_df['Tokenized Review'] = reviews.apply(tokenize_review) # applying the tokenization function to the 'Review Text' column

data_df[['Review Text', 'Tokenized Review']].head()        


# ### 1.2.3 Converting All Words to Lowercase
# In the next step, we convert all tokens to lowercase. This is important because it ensures that words like "Dress" and "dress" are treated as the same word. Without this step, capitalized words would be treated differently, which could lead to redundant or fragmented results in our analysis.
# 
# Here, we create a function `convert_to_lowercase` to ensure that all tokens in each review are in lowercase. This creates a new column, `Lowercased Review`, where all words are in lowercase. This step is critical in text pre-processing to avoid case-sensitive duplicates.

# In[5]:


def convert_to_lowercase(tokens):  # defining a function to convert all tokens to lowercase
    return [token.lower() for token in tokens]

data_df['Lowercased Review'] = data_df['Tokenized Review'].apply(convert_to_lowercase) # applying the function to each tokenized review

data_df[['Tokenized Review', 'Lowercased Review']].head()


# ### 1.2.4 Removing Words Shorter Than 2 Characters
# 
# Now, we remove any words that are shorter than 2 characters. Typically, words like "a" and "i" do not provide much meaning in text analysis, so removing them helps in reducing noise in the dataset. The function `remove_short_words()` filters out words that are shorter than 2 characters. The result is stored in the `Filtered Review` column, where these shorter words are removed.
# 
# In this step, we focus on removing less meaningful tokens (words shorter than 2 characters). This makes our dataset cleaner and more useful for further analysis.

# In[6]:


def remove_short_words(tokens): # defining a function to remove words shorter than 2 characters
    return [token for token in tokens if len(token) > 1] 

data_df['Filtered Review'] = data_df['Lowercased Review'].apply(remove_short_words) # applying the function to each review

data_df[['Lowercased Review', 'Filtered Review']].head()


# ### 1.2.5 Removing Stopwords
# 
# Next, we remove stopwords. Stopwords are common words like "the", "and", "is" that appear frequently but do not carry much significance for text analysis. We use the provided stopword list and filter these words out.
# 
# Here, we load the `stopwords_en.txt` and create a function `remove_stopwords` to eliminate these words from each review and the cleaned data is saved in the `Stopword Removed Review` column. This step reduces unnecessary words that can dilute the importance of other meaningful words.

# In[7]:


with open('stopwords_en.txt', 'r') as file: # loading the stopwords from the provided file
    stopwords_list = [line.strip() for line in file]

def remove_stopwords(tokens): # defining a function to remove stopwords
    return [token for token in tokens if token not in stopwords_list]

data_df['Stopword Removed Review'] = data_df['Filtered Review'].apply(remove_stopwords) # applying the function to each review

data_df[['Filtered Review', 'Stopword Removed Review']].head()


# ### 1.2.6 Remove Words That Appear Only Once (Using Term Frequency)
# 
# In this step, we focus on removing words that appear only once across the entire dataset, using the concept of `Term Frequency (TF)`. Term Frequency refers to how often a word occurs in the dataset. Words that occur only once generally do not provide meaningful insights and may introduce noise. To handle this, we use `FreqDist` from `NLTK` to compute the term frequency for all words in the dataset. After computing the frequencies, we define a simple function `remove_rare_words` that filters out words with a frequency of 1 and the results are stored in `Once Removed Review`. 
# 
# We then apply this function to each review, leaving only the words that appear more than once, ensuring that our dataset focuses on more meaningful terms.

# In[8]:


all_words = [word for review in data_df['Stopword Removed Review'] for word in review] # the list of tokens across all reviews to compute term frequency

term_freq = FreqDist(all_words) # computing term frequency using FreqDist

def remove_rare_words(tokens): # defining a function to remove words that appear only once
    return [word for word in tokens if term_freq[word] > 1]

data_df['Once Removed Review'] = data_df['Stopword Removed Review'].apply(remove_rare_words) # applying the function to remove rare words from each review

data_df[['Stopword Removed Review', 'Once Removed Review']].head()


# ### 1.2.7 Remove the Top 20 Most Frequent Words (Using Document Frequency)
# 
# In this step, we apply `Document Frequency (DF)` to remove the top 20 most frequent words. Document Frequency counts how many different reviews a word appears in, rather than how often it appears overall. Words that appear in most reviews are typically too common and may not add much value to our analysis. To handle this, we first transform each review into a set of unique words, ensuring that each word is only counted once per review. 
# 
# We then compute the document frequency using `FreqDist`, identify the 20 most frequent words, and remove them using a function and the final cleaned reviews are saved in `Final Cleaned Review`. By removing these overly common words, we ensure that the reviews retain more specific and meaningful terms, making our data more informative for future analysis.

# In[9]:


unique_words_per_review = [set(review) for review in data_df['Once Removed Review']] # converting each review into a set of unique words

unique_words = list(chain.from_iterable(unique_words_per_review)) # list of unique words across all reviews to compute document frequency

doc_freq = FreqDist(unique_words)                                 #  computing document frequency using FreqDist


top_20_words = {word for word, freq in doc_freq.most_common(20)}  # identifying the top 20 most frequent words

def remove_top_20_words(tokens):                                  # defining a function to remove the top 20 most frequent words
    return [word for word in tokens if word not in top_20_words]

data_df['Final Cleaned Review'] = data_df['Once Removed Review'].apply(remove_top_20_words) # applying the function to remove the top 20 frequent words from each review

data_df[['Once Removed Review', 'Final Cleaned Review']].head()


# In[10]:


df = data_df['Final Cleaned Review'] # storing the token in a df for step 1.2.9


# ### 1.2.8 Save the Processed Data as `processed.csv`
# 
# In this step, we clean up the DataFrame by dropping the unnecessary columns that were used during intermediate pre-processing steps, such as `Tokenized Review`, `Lowercased Review`, `Filtered Review`, `Stopword Removed Review`, and `Once Removed Review`. Afterward, we add the `Final Cleaned Review` column to the data frame as an extra column here. Additionally, since the cleaned review data is stored as lists of words, we use the `join()` function to convert each list of tokens back into a single string. Lastly, we rearrange the columns to match the original format with an extra column of `Final Cleaned Review` and inspect the first few rows of the cleaned DataFrame. The final processed data will later be saved as `processed.csv` under the **Saving Required Outputs** section.

# In[11]:


data_df.columns


# In[12]:


# dropping the unnecessary columns using data_df.drop
data_df.drop(columns=['Tokenized Review', 'Lowercased Review', 'Filtered Review', 'Stopword Removed Review', 'Once Removed Review'], inplace=True)

data_df['Final Cleaned Review'] = data_df['Final Cleaned Review'].apply(lambda x: ' '.join(' '.join(x).split())) # removes extra whitespaces as well as join the words with only one space also
data_df=data_df[['Clothing ID', 'Age', 'Title', 'Review Text', 'Rating', 'Recommended IND', 'Positive Feedback Count', 'Division Name', 'Department Name', 'Class Name', 'Final Cleaned Review']]


# In[13]:


data_df.head()


# ### 1.2.9 Build a Vocabulary and Save it as `vocab.txt`
# 
# In this task, we extract all the unique words from the cleaned reviews in the `Final Cleaned Review` column. First, we gather all the tokens from the cleaned reviews into a list. Then, we use Python’s `set()` function to ensure that only unique words are included. Once we have the unique words, we sort them alphabetically to ensure proper order in the output. This vocabulary, which contains each unique word from the reviews, will be saved in the format `word:index`, where the index starts from 0. The resulting vocabulary will be stored as `vocab.txt` under the **Saving Required Outputs** section.

# In[14]:


all_cleaned_words = [word for review in df for word in review] # the list of tokens in the cleaned reviews

unique_words = sorted(set(all_cleaned_words)) # sorted alphabetically


# ## Saving required outputs
# This section saves the outputs as per the assignment's requirements. Specifically, we are saving two key outputs:
# 
# 1. `processed.csv` – This file contains the entire dataset, but with the `Final Cleaned Review` column replacing the original `Review Text`.
# 2. `vocab.txt` – This file contains the vocabulary of unique words from the cleaned reviews, sorted alphabetically, with each word assigned a unique integer index starting from 0.

# ### Saving `processed.csv`
# Here, we save the final processed dataset where the `Final Cleaned Review` column is the cleaned and tokenized version of the original `Review Text`. The following columns will be included in the `processed.csv` file:
# 
# - `Clothing ID`
# - `Age`
# - `Title`
# - `Review Text` (`Final Cleaned Review` renamed as `Review Text`)
# - `Rating`
# - `Recommended IND`
# - `Positive Feedback Count`
# - `Division Name`
# - `Department Name`
# - `Class Name`
# - `Final Cleaned Review`
# Here’s the code to save `processed.csv`:

# In[15]:


data_df.to_csv('processed.csv', index=False) # saving into processed.csv


# ### Saving `vocab.txt`
# In the previous step `1.2.9`, we generated the vocabulary list from the cleaned reviews. Now, we will save this vocabulary in the required format: `word_string:word_integer_index`, where each unique word is indexed starting from 0.
# 
# Here’s the code to save `vocab.txt`:

# In[16]:


with open('vocab.txt', 'w') as vocab_file: # saving into vocab.txt
    for index, word in enumerate(unique_words):
        vocab_file.write(f"{word}:{index}\n")


# This ensures that both the vocabulary and the processed dataset are saved as required for further analysis or modeling. The files `processed.csv` and `vocab.txt` are now ready for use.

# ## Summary
# In this assignment, I worked on cleaning and preparing customer reviews from a women’s clothing e-commerce dataset. I went through several steps to clean the text, like breaking down the reviews into individual words (tokenization), making all the words lowercase, and removing short words, common stopwords, and words that appeared only once or too often. After finishing these steps, I saved the cleaned data into a file called `processed.csv` and created a vocabulary file, `vocab.txt`, that lists all the unique words from the reviews.
# 
# This process helped me understand how to handle and clean raw text data, making it ready for more advanced tasks, like analyzing customer sentiments or building models. I was able to compare the dataset after each step by printing the DataFrame, allowing me to clearly see how the reviews changed as I applied each transformation. Now, the data is fully processed and ready for deeper analysis.

# ## References
# 
# [1] M. Lotfinejad, “Learn How to Use the Apply Method in Pandas With This Tutorial,” Dataquest, Feb. 18, 2022. https://www.dataquest.io/blog/tutorial-how-to-use-the-apply-method-in-pandas/
# 
# [2] Amani, “How can I use the apply() function for a single column?,” Stack Overflow, 2024. https://stackoverflow.com/questions/34962104/how-can-i-use-the-apply-function-for-a-single-column (accessed Sep. 26, 2024).
# 
# [3] “Python String join() Method,” GeeksforGeeks, Jan. 02, 2018. https://www.geeksforgeeks.org/python-string-join-method/
# 
# [4] leo, “python combine split and join into 1 line of code,” Stack Overflow, 2024. https://stackoverflow.com/questions/53502754/python-combine-split-and-join-into-1-line-of-code (accessed Sep. 26, 2024).
