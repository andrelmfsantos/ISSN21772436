#!/usr/bin/env python
# coding: utf-8

# # Literature Review Program with Python and Topic Modeling

# <p>Program to develop a literature exploratory analysis from articles with Latent Dirichlet Allocation (LDA) algorithm.</p>
# <p>The database contains the following characteristic:</p>
# 
# <ul>
#   <li>Source: Web of Science & Scopus</li>
#   <li>Range: 2003 to 2021/feb</li>
#     <li>Articles: 694</li>
#   <li>Variables:
#     <ul>
#       <li>AU: authors</li>
#       <li>TI: title</li>
#       <li>PY: publish year</li>
#       <li>AB: abstract</li>
#       <li>AB_Clean: abstract preprocessed</li>
#       <li>DI: doi</li>
#       <li>Country: first author affiliation </li>
#       <li>Field: research field</li>
#       <li>Subject: subject area</li>
#       <li>Quartile: scientific journal rankings</li>
#     </ul>
#   </li>
#   </ul>
#   
# **This program is divided into three sections:**
# 1. Reading database; 
# 2. Pre-processing corpus; 
# 3. Identify dominant topics.
# 
# **Reports:**
# 
# * Report01: coherence scores
# * Report02: terms and topics 
# * Report03: most important topics by article
# * Report04: master corpus dataframe
# * Report05: dominant topics and total docs
# * Report06: relevant papers
# 
# **Benchmark(computer system):**
# * Estimated processing time: 2h
# * Processor: Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz 2.90GHz
# * RAM: 8,00 GB
# * System: Windows 10 Home 64 bits

# **Citation:**
# 
# [Santos, A.L.M.F](http://lattes.cnpq.br/1030756025619809); [Silva,I.C.](http://lattes.cnpq.br/4922228816252524); 
# [Nunes,K.J.F.S.](http://lattes.cnpq.br/8701754979998018); [Silva, J. E. P.](http://lattes.cnpq.br/5464670038922392). (2021). Literature Review Program with Python and Topic Modeling. [DOI:10.5281/zenodo.4588060](https://github.com/andrelmfsantos/ISSN21772436/tree/v1.0.0)

# # 1. Reading database

# In[1]:


# Libraries required
# Need in the same folder "contractions.py"; "text_normalizer.py"; "model_evaluation_utils.py"

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib as plt
pd.options.display.max_colwidth - 200
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')

import keras.backend as K
from keras.layers import Dense, Embedding, Lambda
from keras.layers import Dot
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams
from keras.utils import np_utils
from keras.utils import plot_model

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import gutenberg

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import spacy
import text_normalizer as tn

import warnings
warnings.filterwarnings('ignore')

from string import punctuation
from tqdm import tqdm
from operator import itemgetter
from collections import Counter

import model_evaluation_utils as meu
import itertools
import gensim
import pickle
import seaborn as sns
import copy


# In[2]:


# Dataset from Github

dataset = pd.read_csv('https://raw.githubusercontent.com/andrelmfsantos/ISSN21772436/main/dataset/openinnovation.csv', 
                      sep = ";")
print('Rows = ',len(dataset))
dataset.head(5)


# In[3]:


# building a corpus of documents - pg.203

corpus = list(dataset['AB_Clean'])
labels = list(dataset['Subject'])

corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus, 'Category': labels})
corpus_df = corpus_df[['Document', 'Category']]
corpus_df


# # 2. Pre-processing corpus

# In[4]:


# preprocessing our text corpus - pg.205

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lowercase and remove special characters\whitespace
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)


# In[5]:


# apply it to our sample corpus - pg.207

norm_corpus = normalize_corpus(corpus)
norm_corpus


# In[7]:


# Bag of words models - pg.208

#from sklearn.feature_extraction.text import CountVectorizer

#get bag of words features in sparse format
cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)
cv_matrix


# In[8]:


# view dense representation

# warining might give a memory error if data is too big
cv_matrix = cv_matrix.toarray()
cv_matrix


# # 3. Identify dominant topics

# In[9]:


# Document-topic feature matrix from our LDA model - pg.230

#from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=3, max_iter=10000, random_state=0)
dt_matrix = lda.fit_transform(cv_matrix)
features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3'])
features


# In[10]:


# Loading the bible corpus - pg.233

#from nltk.corpus import gutenberg
#from string import punctuation

bible = corpus_df['Document']
bible =  [i.split(' ') for i in bible]
remove_terms = punctuation + '0123456789'
norm_bible = [[word.lower() for word in sent if word not in remove_terms] for sent in bible]
norm_bible = [' '.join(tok_sent) for tok_sent in norm_bible]
norm_bible = filter(None, normalize_corpus(norm_bible))
norm_bible = [tok_sent for tok_sent in norm_bible if len(tok_sent.split()) > 2]
print('Total lines:', len(bible))
print('\nSample line:', bible[10])
print('\nProcessed line:', norm_bible[10])


# In[11]:


# Implementing the Continuous Bag of Words (CBOW) Model - pg.236
# build the corpus vocabulary

#from keras.preprocessing import text
#from keras.utils import np_utils
#from keras.preprocessing import sequence

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(norm_bible)
word2id = tokenizer.word_index

# build vocabulary of unique words
word2id['PAD'] = 0
id2word = {v:k for k, v in word2id.items()}
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in norm_bible]

vocab_size = len(word2id)
embed_size = 100
window_size = 2 # context window size

print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])


# In[12]:


# Build a CBOW (Context, Target) Generator - pg.237

def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = window_size*2
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word = []
            start = index - window_size
            end = index + window_size + 1
            
            context_words.append([words[i] for i in range(start, end) if 0 <= i < sentence_length and i != index])
            label_word.append(word)
            
            x = sequence.pad_sequences(context_words, maxlen=context_length)
            y = np_utils.to_categorical(label_word, vocab_size)
            yield (x, y)
            
# Test this out for some samples
i = 0
for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
    if 0 not in x[0]:
        print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[np.argwhere(y[0])[0][0]])
        if i == 10:
            break
        i += 1


# In[13]:


# CBOW model summary and architecture - pg.239

#import keras.backend as K
#from keras.models import Sequential
#from keras.layers import Dense, Embedding, Lambda
#from keras.utils import plot_model

# build CBOW architecture
cbow = Sequential()
cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(vocab_size, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#view model summary
print(cbow.summary())


# ### The Skip-Gram Model - pg 244

# In[14]:


# Implementing the Skip-Gram Model - pg.246
# Build the Corpus Vocabulary

#from keras.preprocessing import text

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(norm_bible)

word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}

vocab_size = len(word2id) + 1
embed_size = 100

wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in norm_bible]

print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])


# In[15]:


# Build a Skip-Gram [(target, context), relevancy] Generator - pg.247

#from keras.preprocessing.sequence import skipgrams

# generate skip-grams
skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=10)
for wid in wids]

# view sample skip-grams
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(10):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(id2word[pairs[i][0]], pairs[i][0],
                                                      id2word[pairs[i][1]], pairs[i][1],
                                                      labels[i]))


# In[16]:


# Build the Skip-Gram Model Architecture - pg.248

#from keras.layers import Dot
#from keras.layers.core import Dense, Reshape
#from keras.layers.embeddings import Embedding
#from keras.models import Sequential
#from keras.models import Model

# build skip-gram architecture
word_model = Sequential()
word_model.add(Embedding(vocab_size, embed_size, embeddings_initializer="glorot_uniform", input_length=1))
word_model.add(Reshape((embed_size, )))

context_model = Sequential()
context_model.add(Embedding(vocab_size, embed_size, embeddings_initializer="glorot_uniform", input_length=1))
context_model.add(Reshape((embed_size,)))
model_arch = Dot(axes=1)([word_model.output, context_model.output])
model_arch = Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid")(model_arch)
model = Model([word_model.input,context_model.input], model_arch)
model.compile(loss="mean_squared_error", optimizer="rmsprop")

# view model summary
print(model.summary())


# In[17]:


# Train the Model - pg.251

for epoch in range(1, 6):
    loss = 0
    for i, elem in enumerate(skip_grams):
        pair_first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        pair_second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [pair_first_elem, pair_second_elem]
        Y = labels
        if i % 10000 == 0:
            print('Processed {} (skip_first, skip_second, relevance) pairs'.format(i))
        loss += model.train_on_batch(X,Y)
    print('Epoch:', epoch, 'Loss:', loss)


# In[18]:


# Word embeddings for our vocabulary based on the Skip-Gram model - pg.252

word_embed_layer = model.layers[2]
weights = word_embed_layer.get_weights()[0][1:]

print(weights.shape)
pd.DataFrame(weights, index=id2word.values()).head()


# In[19]:


# building the dataframe - pg.286

data_df = pd.DataFrame({'Article':dataset['AB_Clean'], 'TargetLabel':dataset['Field'], 'TargetName':dataset['Subject']})
print(data_df.shape)
data_df.head(10)


# In[20]:


# Data Preprocessing and Normalization - pg.287

total_nulls = data_df[data_df.Article.str.strip() == ''].shape[0]
print("Empty documents:", total_nulls)


# In[21]:


# Remove all the records with no textual content in the article - pg.287

data_df = data_df[~(data_df.Article.str.strip() == '')]
data_df.shape


# In[22]:


# normalize our corpus - pg.290 (adapted from pg.203)

corpus = list(data_df['Article'])
corpus = np.array(corpus)
norm_corpus = normalize_corpus(corpus)
data_df['CleanArticle'] = norm_corpus

# view sample data

data_df = data_df[['Article', 'CleanArticle', 'TargetLabel', 'TargetName']]
data_df.head(10)


# In[23]:


# Test if have some documents that, after preprocessing, might end up being empty or null - pg.291

data_df = data_df.replace(r'^(\s?)+$', np.nan, regex=True)
data_df.info()


# In[24]:


# Remove null documents

data_df = data_df.dropna().reset_index(drop=True)
data_df.info()


# In[25]:


# Building Train and Test Datasets

#from sklearn.model_selection import train_test_split

train_corpus, test_corpus, train_label_nums, test_label_nums, train_label_names, test_label_names =                                 train_test_split(np.array(data_df['CleanArticle']), np.array(data_df['TargetLabel']),
                                                       np.array(data_df['TargetName']), test_size=0.7, random_state=42)

train_corpus.shape, test_corpus.shape


# ### TF-IDF Features with Classification Models - pg.319

# In[26]:


# Performance models - pg.319

#from sklearn.feature_extraction.text import TfidfVectorizer

# build BOW features on train articles
tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0)
tv_train_features = tv.fit_transform(train_corpus)

# transform test articles into features
tv_test_features = tv.transform(test_corpus)

print('TFIDF model:> Train features shape:', tv_train_features.shape, ' Test features shape:', tv_test_features.shape)


# ### Topic Modeling - pg.362

# In[27]:


papers = data_df.Article
print(papers)


# In[28]:


get_ipython().run_cell_magic('time', '', "#import nltk\n\nstop_words = nltk.corpus.stopwords.words('english')\nwtk = nltk.tokenize.RegexpTokenizer(r'\\w+')\nwnl = nltk.stem.wordnet.WordNetLemmatizer()\n\ndef normalize_corpus(papers):\n    norm_papers = []\n    for paper in papers:\n        paper = paper.lower()\n        paper_tokens = [token.strip() for token in wtk.tokenize(paper)]\n        paper_tokens = [wnl.lemmatize(token) for token in paper_tokens if not token.isnumeric()]\n        paper_tokens = [token for token in paper_tokens if len(token) > 1]\n        paper_tokens = [token for token in paper_tokens if token not in stop_words]\n        paper_tokens = list(filter(None, paper_tokens))\n        if paper_tokens:\n            norm_papers.append(paper_tokens)\n            \n    return norm_papers\n    \nnorm_papers = normalize_corpus(papers)\nprint(len(norm_papers))")


# In[29]:


norm_papers = normalize_corpus(corpus)
print(norm_papers[0][:512])


# In[30]:


# Topic Models with Gensim - pg.368

#import gensim

bigram = gensim.models.Phrases(norm_papers, min_count=5, threshold=100, delimiter=b'_') # higher threshold fewer phrases.
bigram_model = gensim.models.phrases.Phraser(bigram)
# sample demonstration
print(bigram_model[norm_papers[0]][:50])


# In[31]:


# Let’s generate phrases for all our tokenized research papers and build a vocabulary - pg.370

norm_corpus_bigrams = [bigram_model[doc] for doc in norm_papers]

# Create a dictionary representation of the documents.
dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
print('Sample word to number mappings:', list(dictionary.items())[:15])
print('Total Vocabulary Size:', len(dictionary))


# In[32]:


# Prune our vocabulary and start removing terms - pg.370
# Github Ch06b - Topic Modeling with gensim.ipynb

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.6)
print('Total Vocabulary Size:', len(dictionary))


# In[33]:


# # Transforming corpus into bag of words vectors - pg.371
# Github Ch06b - Topic Modeling with gensim.ipynb

bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
print(bow_corpus[1][:50])


# In[34]:


# viewing actual terms and their counts - pg.371

print([(dictionary[idx] , freq) for idx, freq in bow_corpus[1][:50]])


# In[35]:


# total papers in the corpus - pg.371

print('Total number of papers:', len(bow_corpus))


# ### LDA Tuning: Finding the Optimal Number of Topics - pg.402

# In[36]:


# LDA Tuning: Finding the Optimal Number of Topics - pg.402

#from tqdm import tqdm

def topic_model_coherence_generator(corpus, texts, dictionary, 
                                    start_topic_count=2, end_topic_count=10, step=1,
                                    cpus=1):
    
    models = []
    coherence_scores = []
    for topic_nums in tqdm(range(start_topic_count, end_topic_count+1, step)):
        lda_model = gensim.models.LdaModel(corpus=bow_corpus, id2word=dictionary, chunksize=1740, 
                                           alpha='auto', eta='auto', random_state=42,
                                           iterations=500, num_topics=30, 
                                           passes=20, eval_every=None)
        cv_coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, corpus=corpus, 
                                                              texts=texts, dictionary=dictionary, 
                                                              coherence='c_v')
        coherence_score = cv_coherence_model_lda.get_coherence()
        coherence_scores.append(coherence_score)
        models.append(lda_model)
    
    return models, coherence_scores


# In[37]:


# Finding the optimal number of topics in a topic model - part 1

#from tqdm import tqdm

def topic_model_coherence_generator(corpus, texts, dictionary,
                          start_topic_count=2, end_topic_count=10, step=1,
                                    cpus=1):
    models = []
    coherence_scores = []
    for topic_nums in tqdm(range(start_topic_count, end_topic_count+1, step)):
        gensim_lda_model = gensim.models.LdaModel(
                                              corpus=corpus,
                                              num_topics=topic_nums,
                                              id2word=dictionary,
                                              chunksize=1740,
                                              iterations=500,
                                              alpha="auto",
                                              eta="auto",
                                              passes=20
                                                )

        cv_coherence_model_gensim_lda = gensim.models.CoherenceModel(model=gensim_lda_model,
                                                    corpus=corpus,
                                                    texts=texts,
                                                    dictionary=dictionary,
                                                    coherence='c_v')

        coherence_score = cv_coherence_model_gensim_lda.get_coherence()
        coherence_scores.append(coherence_score)
        models.append(gensim_lda_model)
        
        ### saving each model
        gensim_lda_model.save(''+str(topic_nums)+'.gensim')

    return models, coherence_scores


# In[38]:


# Finding the optimal number of topics in a topic model - part 2


lda_models, coherence_scores = topic_model_coherence_generator(corpus=bow_corpus,
                                                texts=norm_corpus_bigrams,
                                                dictionary=dictionary,
                                                start_topic_count=2,
                                                end_topic_count=30, step=1, cpus=16)


# In[39]:


# Finding the optimal number of topics in a topic model - part 3

coherence_df = pd.DataFrame({'Number of Topics': range(2, 31, 1),
                        'Coherence Score': np.round(coherence_scores, 4)})
coherence_df.sort_values(by=['Coherence Score'], ascending=False).head(10)


# In[40]:


#------------------------------------------------------------------------
# Report 1 - save coherence score df and coherence score list 

coherence_df.to_csv('Report01_coherence_df.csv', index=False)
#------------------------------------------------------------------------


# In[41]:


# Finding the optimal number of topics in a topic model - part 4

#import pickle
with open("coherence_scores.txt", "wb") as fp:   #Pickling
    pickle.dump(coherence_scores, fp)


# In[42]:


# Topic model tuning the number of topics vs. coherence score - pg.404

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

x_ax = range(2, 31, 1)
y_ax = coherence_scores
plt.figure(figsize=(12, 6))
plt.plot(x_ax, y_ax, c='r')
plt.axhline(y=round(np.mean(coherence_scores),2), c='k', linestyle='--', linewidth=2)
plt.rcParams['figure.facecolor'] = 'white'
xl = plt.xlabel('Number of Topics')
yl = plt.ylabel('Coherence Score')


# In[43]:


# We can retrieve the best model - pg.405

best_model_idx = coherence_df[coherence_df['Number of Topics'] == 6].index[0]
best_lda_model = lda_models[best_model_idx]
best_lda_model.num_topics


# In[44]:


# Let’s view all the 20 topics generated by our selected best model - pg.405

topics = [[(term, round(wt, 3)) 
               for term, wt in best_lda_model.show_topic(n, topn=20)] 
                   for n in range(0, best_lda_model.num_topics)]

for idx, topic in enumerate(topics):
    print('Topic #'+str(idx+1)+':')
    print([term for term, wt in topic])
    print()


# In[45]:


# A better way of visualizing the topics is to build a term-topic dataframe - 407

topics_df = pd.DataFrame([[term for term, wt in topic] 
                              for topic in topics], 
                         columns = ['Term'+str(i) for i in range(1, 21)],
                         index=['Topic '+str(t) for t in range(1, best_lda_model.num_topics+1)]).T
topics_df


# In[46]:


#------------------------------------------------------------------------
# Report 2 - terms and topics

topics_df.to_csv("Report02_TermsTopics.csv")
#------------------------------------------------------------------------


# In[47]:


# Viewing all the topics of our LDA topic model - pg.408

topics_df = pd.DataFrame([', '.join([term for term, wt in topic])  
                              for topic in topics],
                         columns = ['Terms per Topic'],
                         index=['Topic'+str(t) for t in range(1, best_lda_model.num_topics+1)]
                         )
topics_df


# ### Latent Semantic Indexing - pg.372

# In[48]:


get_ipython().run_cell_magic('time', '', '\nTOTAL_TOPICS = best_lda_model.num_topics\nlsi_bow = gensim.models.LsiModel(bow_corpus, id2word=dictionary, num_topics=TOTAL_TOPICS,\n                                 onepass=True, chunksize=1740, power_iters=1000)')


# In[49]:


# major topics or themes in our corpus - pg.372

for topic_id, topic in lsi_bow.print_topics(num_topics=TOTAL_TOPICS, num_words=20):
    print('Topic #'+str(topic_id+1)+':')
    print(topic)
    print()


# In[50]:


# Separate different sub-themes based on the sign or orientation of terms - pg.375

for n in range(TOTAL_TOPICS):
    print('Topic #'+str(n+1)+':')
    print('='*50)
    d1 = []
    d2 = []
    for term, wt in lsi_bow.show_topic(n, topn=20):
        if wt >= 0:
            d1.append((term, round(wt, 3)))
        else:
            d2.append((term, round(wt, 3)))

    print('Direction 1:', d1)
    print('-'*50)
    print('Direction 2:', d2)
    print('-'*50)
    print()


# In[51]:


# SVD - pg.379

term_topic = lsi_bow.projection.u
singular_values = lsi_bow.projection.s
topic_document = (gensim.matutils.corpus2dense(lsi_bow[bow_corpus], len(singular_values)).T / singular_values).T
term_topic.shape, singular_values.shape, topic_document.shape


# In[52]:


# Output shows - pg.379

document_topics = pd.DataFrame(np.round(topic_document.T, 3), 
                               columns=['T'+str(i) for i in range(1, TOTAL_TOPICS+1)])
document_topics.head(15)


# In[53]:


# Find out the most important topics for a few sample papers - pg.380

document_numbers = range(0,9)

for document_number in document_numbers:
    top_topics = list(document_topics.columns[np.argsort(-np.absolute(document_topics.iloc[document_number].values))[:3]])
    print('Document #'+str(document_number)+':')
    print('Dominant Topics (top 3):', top_topics)
    print('Paper Summary:')
    print(papers[document_number][:500])
    print()


# In[54]:


#------------------------------------------------------------------------
# Report 3 - most important topics by article
import_topics = pd.concat([corpus_df, document_topics], axis=1)
import_topics.to_csv("Report03_MostImportantTopics.csv")
import_topics
#------------------------------------------------------------------------


# ### Interpreting Topic Model Results - pg.409

# In[55]:


# Predict the distribution of topics in each research paper - pg.409

tm_results = best_lda_model[bow_corpus]


# In[56]:


# Most dominant topic per research paper with some intelligent sorting and indexing - pg.410

corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] 
                     for topics in tm_results]
corpus_topics[:5]


# In[57]:


# Master dataframe that will hold the base statistics - pg.410

corpus_topic_df = pd.DataFrame()
corpus_topic_df['Document'] = range(0, len(corpus_topics))
corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
corpus_topic_df['Topic Desc'] = [topics_df.iloc[t[0]]['Terms per Topic'] for t in corpus_topics]
corpus_topic_df['Paper'] = corpus_df.Document
corpus_topic_df = pd.concat([corpus_topic_df,dataset.DI],axis=1)
corpus_topic_df


# In[58]:


#------------------------------------------------------------------------
# Report 4 - master corpus dataframe
corpus_topic_df.to_csv("Report04_Master_Corpus_Topic.csv")
#------------------------------------------------------------------------


# ### Dominant Topics Distribution Across Corpus - pg.410

# In[59]:


# Viewing the distribution of dominant topics - pg.410

pd.set_option('display.max_colwidth', 100)

topic_stats_df = pd.DataFrame(corpus_topic_df['Dominant Topic'],columns=['Dominant Topic'])
topic_stats_df['Doc Count'] = topic_stats_df.groupby(by='Dominant Topic')['Dominant Topic'].transform('count')
topic_stats_df['% Total Docs'] = round(topic_stats_df['Doc Count']/len(corpus_topic_df)*100,2)
paper_df = corpus_topic_df[['Dominant Topic','Topic Desc']]
topic_stats_df = pd.merge(topic_stats_df,paper_df, on="Dominant Topic", how='inner')
topic_stats_df = topic_stats_df.drop_duplicates(subset=['Dominant Topic'])
topic_stats_df = topic_stats_df.sort_values(by=['Dominant Topic'], ascending=True)
topic_stats_df = topic_stats_df.reset_index(drop=True)
topic_stats_df


# In[60]:


#------------------------------------------------------------------------
# Report 5 - dominant topics and total docs
#topic_stats_df.to_csv("Report05_DominantTopic-ToTalDocs.csv")
#------------------------------------------------------------------------


# ### Relevant Research Papers per Topic Based on Dominance - pg.413

# In[61]:


# Viewing each topic and corresponding paper with its maximum contribution - pg.413

relevant_papers_df = corpus_topic_df.groupby('Dominant Topic').apply(lambda topic_set:
                                                                     (topic_set.sort_values(by=['Contribution %'],
                                                                                            ascending=False).iloc[0]))
relevant_papers_df


# In[62]:


#------------------------------------------------------------------------
# Report 6 - relevant papers
relevant_papers_df.to_csv("Report06_RelevantPapers.csv")
#------------------------------------------------------------------------


# ### Reference
# * Sarkar, D. (2019). Text Analytics with Python. Apress. doi:10.1007/978-1-4842-4354-1
