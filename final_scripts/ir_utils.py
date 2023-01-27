import os
import sys
import time
import nltk
import string
import requests
import numpy as np
import pandas as pd
import pyterrier as pt
import gensim.downloader as api

from tqdm import tqdm
from ir_utils import *
from pymed import PubMed
from gensim import models
from langdetect import detect
from unidecode import unidecode
from pyterrier.measures import *
from gensim.models import Word2Vec
from krovetzstemmer import Stemmer
from gensim.models import KeyedVectors
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from semanticscholar import SemanticScholar


pubmed = PubMed(tool="Trec-Covid-Local-QE-Tool", email="tim_colin.pack1@smail.th-koeln.de")

krovetz_stemmer = Stemmer()
regex_tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
semantic_scholar = SemanticScholar()

# Path to the data folder
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"))
INDICES_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../indices"))

# Helper function to get the pyterrier instance
def get_pyterrier_instance():
    if not pt.started():
        pt.init()
    return pt


# Helper function to intialize multiple indices
# Prepares the index path and avoid errors with already existing indices
# Takes the index name and the index path as input
# Returns the index path
index_count = 0
def prepare_index_path(indexName, index_path):
    global index_count
    index_count = index_count + 1
    index_path = index_path + '/' + indexName + str(index_count)

    if os.path.exists(index_path) & os.path.isdir(index_path):
        files = os.listdir(index_path)
        for file in files:
            file_name = index_path + '/' + file
            os.remove(file_name)
        os.rmdir(index_path)
    elif os.path.exists(index_path) & (not os.path.isdir(index_path)):
        os.rmove(index_path)

    return os.path.abspath(index_path)

# Helper function to build an index
# Takes the index name and the dataset as input
# Returns the index
def build_index(indexName, dataset):
    index_path = prepare_index_path(indexName, INDICES_PATH)
    indexer = pt.IterDictIndexer(
        index_path, overwrite=True, blocks=True)
    indexer.setProperty(
        "stopwords.filename", os.path.abspath("en.txt"))
    index_created_ref = indexer.index(dataset.get_corpus_iter(),
                                  fields=['title', 'doi', 'abstract'],
                                  meta=('docno',))
    index_created = pt.IndexFactory.of(index_created_ref)
    return index_created


# Helper function to run an experiment with bm25,tfidf and inL2
# Takes the index, the query and the qrels as input
# Returns the experiment
def run_experiment(docs, query, qrels):
    tfidf = pt.BatchRetrieve(docs, wmodel="TF_IDF")
    bm25 = pt.BatchRetrieve(docs, wmodel="BM25")
    inL2 = pt.BatchRetrieve(docs, wmodel="InL2")

    bo1 = pt.rewrite.Bo1QueryExpansion(docs)

    return pt.Experiment(
        [tfidf, bm25, inL2, bm25 >> bo1 >> bm25],
        query,
        qrels,
        eval_metrics=[P@5, P@10, nDCG@10, nDCG, RR(rel=2), "map"],
        names=["TF_IDF", "BM25", "InL2", "bm25 >> bo1 >> bm25"]
    )


# Use SMART Stopwordlist to remove stopwords
# Preprocess the given text with SMART Stopword list, regex-Tokenizer and lowercase conversion
with open(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "en.txt"))) as f:
    stopword_list_lines = [line.rstrip('\n') for line in f]
stop_words = set(stopword_list_lines)
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])


def remove_special_chars(text):
    punctuation = string.punctuation + ".,:;–—‐“”″„’‘•′·«»§¶‘"
    return "".join([i for i in text if i not in punctuation])

# tokenize sentences with nltk sent_tokenize
# lowercase sentences
# remove stopwords with SMART Stopword list
# remove special characters
# tokenize words with regex tokenizer
# stem words with krovetz stemmer
# return the preprocessed text as a list of tokens
# if the token cannot be stemmed, use unidecode to convert it to ascii
def preprocess(text,useLowercase=True,useStopwords=True,useSpecialChars=True,userRegexTokenizer=True,useStemmer=True):
    tokens = []
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        return []
    for sentence in sentences:
        if useLowercase:
            sentence = sentence.lower()
        if useStopwords:
            sentence = remove_stopwords(sentence)
        if useSpecialChars:
            sentence = remove_special_chars(sentence)
        if userRegexTokenizer:
            tokens = tokens + regex_tokenizer.tokenize(sentence)
        else:
            tokens = tokens + nltk.word_tokenize(sentence)
    result = []
    for token in tokens:
        try:
            if useStemmer:
                result.append(krovetz_stemmer.stem(token))
            else:
                result.append(token)
        except Exception as e:
            result.append(krovetz_stemmer.stem(unidecode(token)))
    return result

# Also return the original tokens
def preprocess_sentence(sentence,useLowercase=True,useStopwords=True,useSpecialChars=True,userRegexTokenizer=True,useStemmer=True):
    if useLowercase:
        sentence = sentence.lower()
    if useStopwords:
        sentence = remove_stopwords(sentence)
    if useSpecialChars:
        sentence = remove_special_chars(sentence)
    if userRegexTokenizer:
        tokens = regex_tokenizer.tokenize(sentence)
    else:
       tokens = nltk.word_tokenize(sentence)
    result = []
    for token in tokens:
        try:
            if useStemmer:
                result.append([krovetz_stemmer.stem(token),tokens])
            else:
                result.append([token,token])
        except Exception as e:
            result.append([unidecode(token),token])
    return result


# We can only use english because of english stopword list so we have to detect languages
# And filter for english docs
# Takes the docs, the language to return and the field to detect the language from as input
# Returns the docs
# If the language is not specified, all languages are returned
def detect_language(docs,return_specific_language="en",field="abstract"):
    languages = []
    for i, doc in docs.iterrows():
        text = doc[field]
        # If no field is not present take the title for language detection
        # If the detection fails or there is not 'title' or 'abstract' set language to 'unkown'
        if text is pd.NA or text == "":
            text = doc['title']
        try:
            lang = detect(text)
        except Exception as e:
            lang = 'unknown'
        languages.append(lang)

    # convert the languages list to a numpy array
    languages = np.asarray(languages)
    if return_specific_language:
        docs = docs.loc[languages == return_specific_language]
    return docs


# Avoid nltk.sent_tokenize to crash the whole program
# If nltk.sent_tokenize crashes, return an empty list
def caught_sent_tokenize(text):
    try:
        return nltk.sent_tokenize(text)
    except Exception as e:
        return []


# format the data to be able to use it with word2vec
# nltk is used to split the text into sentences
def prepare_docs_for_word2vec(docs):
    docs_abstracts = docs.abstract
    docs_abstracts_sentences = docs_abstracts.apply(
        lambda input_text: [t.split() for t in caught_sent_tokenize(input_text)])
    return docs_abstracts_sentences.sum()


# Helper function to preprocess the data
# Takes the dataset as input
# Returns the preprocessed dataset
def preprocess_data(irds_trec_covid_dataset_docs):
    # Check if data/en_docs_preprocessed.pkl exists
    # If it does not exist, preprocess the data and save it to the file
    # If it exists, load the data from the file
    if not os.path.exists(os.path.join(DATA_PATH, "en_docs_preprocessed.pkl")):
        print("Preprocessing data...")
        # Check if file data/en_docs.pkl exists
        # If it does not exist, preprocess the data and save it to the file
        # If it exists, load the data from the file
        if not os.path.exists(os.path.join(DATA_PATH, "en_docs.pkl")):
            print("Detecting languages...")
            irds_trec_covid_dataset_en_docs = detect_language(irds_trec_covid_dataset_docs)
            irds_trec_covid_dataset_en_docs.to_pickle(os.path.join(DATA_PATH, "en_docs.pkl"))
        else:
            print("Loading language data...")
            irds_trec_covid_dataset_en_docs = pd.read_pickle(os.path.join(DATA_PATH, "en_docs.pkl"))

        irds_trec_covid_dataset_en_docs_preprocessed = irds_trec_covid_dataset_en_docs.copy()
        for index, row in tqdm(irds_trec_covid_dataset_en_docs_preprocessed.iterrows(), total=len(irds_trec_covid_dataset_en_docs_preprocessed)):
            if row["title"] is not pd.NA and not row["title"] == "":
                title_tokens = preprocess(row["title"])
                irds_trec_covid_dataset_en_docs_preprocessed.loc[index, "title"] = " ".join(title_tokens)
            if row["abstract"] is not pd.NA and not row["abstract"] == "":
                abstract_tokens = preprocess(row["abstract"])
                irds_trec_covid_dataset_en_docs_preprocessed.loc[index, "abstract"] = " ".join(abstract_tokens)
        irds_trec_covid_dataset_en_docs_preprocessed.to_pickle(os.path.join(DATA_PATH, "en_docs_preprocessed.pkl"))
    else:
        print("Loading preprocessed data...")
        irds_trec_covid_dataset_en_docs_preprocessed = pd.read_pickle(os.path.join(DATA_PATH, "en_docs_preprocessed.pkl"))
    return irds_trec_covid_dataset_en_docs_preprocessed


def get_initial_word2vec_model(preprocessed_data):
    # Check if data/word2vec_abstracts.model exists
    # If it does not exist, train the model and save it to the file
    # If it exists, load the model from the file
    if not os.path.exists(os.path.join(DATA_PATH, "word2vec_abstracts.model")):
        print("Training word2vec model...")
        word2vec_model = Word2Vec(sentences=preprocessed_data, window=10, min_count=1, sg=1, seed=1)
        word2vec_model.save(os.path.join(DATA_PATH, "word2vec_abstracts.model"))
    else:
        print("Loading word2vec model...")
        word2vec_model = Word2Vec.load(os.path.join(DATA_PATH, "word2vec_abstracts.model"))
    return word2vec_model


# Helper function to get the global word2vec model
# Returns the global word2vec model
def get_global_word2vec_model():
    print("Downloading word2vec model...")
    word2vec_model = api.load("glove-wiki-gigaword-300")
    return word2vec_model


# Helper function to get the top k documents for a query
# Takes the index, the number of documents to return, the query and a boolean to sort the results by score
# Returns the top k documents for the query
def get_top_k_docs_bm25(index,queries,k,isSorted=True):
    bm25 = pt.BatchRetrieve(index, wmodel='BM25', num_results=k)
    res = bm25.transform(queries)
    if isSorted:
        return res.sort_values(by="score", ascending=False)[:3]
    else:
        return res[:3]


# Helper function to get the a dataframe of the docs for a field
# Takes the docs and the field
# Returns a dataframe with the qid and the field as query
def get_query_qid_df_tokens_from_docs_by_field(docs,docs_to_get_tokens_for,field="title"):
    tokens = []
    i = 0
    for index, row in docs_to_get_tokens_for.iterrows():
        row = docs.loc[docs["docno"] == row.docno].iloc[0]
        if row[field] is not pd.NA and not row[field] == "":
            tokens.append([str(i),row[field]])
            i = i + 1
    return pd.DataFrame(tokens, columns=["qid", "query"])


# Helper function to rank the tokens from a dataframe with BM25 and Bo1
# Takes the index, the dataframe with the tokens, the number of tokens to return and a boolean to sort the results by score
# Returns the top k tokens for the query
def get_top_k_ranked_tokens_from_dataframe_with_bm25_and_bo1(index, tokens_df,k,isSorted=True):
    bm25 = pt.BatchRetrieve(index, wmodel='BM25')
    bo1 = pt.rewrite.Bo1QueryExpansion(index)
    pipelineQE = bm25 >> bo1 >> bm25
    if not isSorted:
        return res[:k]["query_0"].values
    res = pipelineQE.transform(tokens_df)
    res = res.sort_values(by="score", ascending=False)
    res = res.drop_duplicates(subset=['qid'], keep='first')
    res = res.drop_duplicates(subset=['query_0'], keep='first')
    # ATTENTION MIGHT BE HORRIBLY WRONG. IF YOU READ THIS AND KNOW A BETTER WAY TO DO IT PLEASE TELL ME
    if isSorted:
        return res[:k]["query_0"].values


# takes docno and returns list of titles
# searches docs and metadata for doi
# if there is no doi looks for the 'title' in pubmed and takes the first result
# gets the paper by doi from semantic scholar
def get_references_titles_by_docno_from_sm_and_pubmed(metadata,docs,docno):
    current_doc_metadata = metadata.loc[metadata["cord_uid"] == docno]
    current_doc_metadata_doi = current_doc_metadata.doi.item()
    current_doc_docs_doi = docs.loc[docs["docno"] == docno].doi.item()

    if not current_doc_metadata_doi and not docs.loc[docs["docno"] == docno]:
        print("Paper not found in metadata and docs!")
        return []

    paper = semantic_scholar.get_paper(current_doc_metadata_doi or current_doc_docs_doi)
    if not paper or paper == {} or not paper.references:

        results = pubmed.query(current_doc_metadata.title.item(), max_results=1)
        if not results:
            print("Paper not found in semanticscholar and pubmed!")
            return []
        try:
            paper_obj = next(results)
            paper_pubmed_id = paper_obj.pubmed_id.split("\n")[0]
            paper = semantic_scholar.get_paper("PMID:" + paper_pubmed_id)
        except StopIteration as e:
            print(e)
            return []
    if not paper or not paper.references:
        print("No references found for paper!")
        return []
    else:
        return list(map(lambda x:x.title, paper.references))


# takes a dataframe with docnos and returns a list of titles
# searches docs and metadata for doi
# if there is no doi looks for the 'title' in pubmed and takes the first result
# gets the paper by doi from semantic scholar
def get_all_titles_from_references_of_docs_from_sm_and_pubmed(metadata,docs,docs_to_get_titles_for):
    titles = []
    titlesAmount = 0
    for index, row in docs_to_get_titles_for.iterrows():
        tmpTitles = get_references_titles_by_docno_from_sm_and_pubmed(metadata,docs,row["docno"])
        titles.append(tmpTitles)
        titlesAmount = titlesAmount + len(tmpTitles)
        sys.stdout.write("\r" + str(titlesAmount) + " titles found")
    print("\n")
    return titles


# Could do more with all the parameters but for now this should do
# Takes a list of authors and a query string
# Returns a list of papers
def semantic_scholar_search_author_and_query_string(authors_name_array,query_string,paper_count):
    r = requests.post("https://www.semanticscholar.org/api/1/search", json={"queryString": query_string, "page": 1, "pageSize": 10, "sort": "relevance", "authors": authors_name_array, "coAuthors": [], "venues": [], "yearFilter": None, "requireViewablePdf": False, "fieldsOfStudy": [
    ], "useFallbackRankerService": False, "useFallbackSearchCluster": False, "hydrateWithDdb": True, "includeTldrs": True, "performTitleMatch": True, "includeBadges": True, "tldrModelVersion": "v2.0.0", "getQuerySuggestions": False, "useS2FosFields": True})
    if r.status_code == 200:
        results = r.json()['results']
        return results[:paper_count]
    else:
        print(r.status_code)
        return []

# Take docs and metadata and a field
# Return all relevant papers of the authors of the docs
# Relevant papers are ranked by semantic sholar API
# k is the number of relevant papers to get
def get_field_for_all_relevant_authors(metadata,docs_to_get_field_for,k,field="paperAbstract"):
    author_dict = {}
    author_relevant_paper_count = k
    for id, authors in zip(metadata['cord_uid'], metadata['authors']):
        author_dict[id] = authors

    all_authors_relevant_papers = []
    for index,current_doc in docs_to_get_field_for.iterrows():
        current_doc_authors_string = metadata.loc[metadata["cord_uid"] == current_doc.docno].authors.item()
        current_doc_authors = current_doc_authors_string.split('; ')
        for author in current_doc_authors:
            tmp_author_name_array = author.split(", ")
            semantic_scholar_author_search_string = tmp_author_name_array[1] + " " + tmp_author_name_array[0]
            print("Searching current author: "+ author + "\nQuery: "+current_doc.query)
            current_author_relevant_papers = semantic_scholar_search_author_and_query_string([semantic_scholar_author_search_string],current_doc.query,author_relevant_paper_count)
            all_authors_relevant_papers = all_authors_relevant_papers + current_author_relevant_papers
            time.sleep(5) # Sleep to not hit API-Limit
    return list(map(lambda x:x[field]['text'], all_authors_relevant_papers))

# get query qid df from all preprocessed tokens of docs
# takes metadata, top_k_docs, top_k_docs_references_titles, all_authors_relevant_papers_abstracts
# returns a dataframe with qid and  query
def get_query_qid_df_from_all_preprocessed_tokens_of_docs(metadata,top_k_docs,top_k_docs_references_titles,all_authors_relevant_papers_abstracts,field="abstract"):
    tokens_top_k_docs_abstracts = []
    for index,doc in top_k_docs.iterrows():
        current_doc_abstract = metadata.loc[metadata["cord_uid"] == doc["docno"]][field].item()
        tokens_top_k_docs_abstracts.append(current_doc_abstract)
    print("Preparing abstracts done.")

    all_terms = []
    for abstract in tokens_top_k_docs_abstracts:
        all_terms = all_terms + preprocess(abstract)
    print("Tokens_top_k_docs_abstracts done.")
    for title in top_k_docs_references_titles:
        all_terms = all_terms + preprocess(title)
    print("Top_k_docs_references_titles done.")
    for abstract in all_authors_relevant_papers_abstracts:
        all_terms = all_terms + preprocess(abstract)
    print("All_authors_relevant_papers_abstracts done.")

    return pd.DataFrame([[i,str(x)] for i,x in enumerate(all_terms)], columns=["qid", "query"])

# tokenize sentences array
# takes a list of sentences
# returns a list of tokens
def tokenize_sentences_array(sentences_array):
    tokens_in_sentence = []
    for sentence in sentences_array:
        tokens_in_sentence.append(preprocess(sentence))
    return tokens_in_sentence

# get tokenized sentences for word2vec
# takes metadata, top_k_docs, top_k_docs_references_titles, all_authors_relevant_papers_abstracts
# returns a list of tokens
# Example: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'sentence']]
def get_tokenized_sentences_for_word2vec(metadata,top_k_docs,top_k_docs_references_titles,all_authors_relevant_papers_abstracts):
    print("preparing abstracts...")
    tokens_top_k_docs_abstracts = []
    for index,doc in top_k_docs.iterrows():
        current_doc_abstract = metadata.loc[metadata["cord_uid"] == doc["docno"]]["abstract"].item()
        tokens_top_k_docs_abstracts.append(current_doc_abstract)
    print("preparing abstracts done.")

    all_terms_per_sentences = []
    for abstract in tokens_top_k_docs_abstracts:
        tmp_sentences = caught_sent_tokenize(abstract)
        all_terms_per_sentences = all_terms_per_sentences + tokenize_sentences_array(tmp_sentences)
    print("all_terms_per_sentences done.")


    for title in top_k_docs_references_titles:
        tmp_sentences = caught_sent_tokenize(title)
        all_terms_per_sentences = all_terms_per_sentences + tokenize_sentences_array(tmp_sentences)
    print("top_k_docs_references_titles done.")


    for abstract in all_authors_relevant_papers_abstracts:
        tmp_sentences = caught_sent_tokenize(abstract)
        all_terms_per_sentences = all_terms_per_sentences + tokenize_sentences_array(tmp_sentences)
    print("all_authors_relevant_papers_abstracts done.")

    return all_terms_per_sentences


# retrain word2vec model
# takes model, tokenized_sentences, epochs=10
def retrain_word2vec_model(model,tokenized_sentences,epochs=10):
    # Die tokens aus 2.4 verwenden, um word2vec nach zu trainieren und zusätzliche termkanidaten erstellen
    model.build_vocab(tokenized_sentences, update=True)  # prepare the model vocabulary
    model.min_count = 1
    model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=epochs)  # train word vectors
    model.save(os.path.abspath(os.path.join(DATA_PATH, "word2vec_retrained.model")))
    return model



# This code is used to expand the search terms by finding the most similar words to the terms in the corpus
# and adding them to the list of search terms. The most similar words are found using the Word2Vec model
# and the word embedding vectors are used to find the similar words. The similar words are filtered based on
# a threshold value and the top MAX_EXPANSION_WORD_COUNT words are added to the search terms list.
def expand_search_terms(all_terms,abstracts_model,threshold=0.8,max_expansion_word_count=5):
    all_terms_expanded = []
    for term in all_terms:
        try:
            similar_words = abstracts_model.wv.most_similar(term)
            filtered_similar_words = [word for word, score in similar_words if score > threshold]
        except Exception as e:
            all_terms_expanded.append(term)
        
        all_terms_expanded.append(term)
        all_terms_expanded = all_terms_expanded + filtered_similar_words[:max_expansion_word_count]
    return pd.DataFrame([[i,str(x)] for i,x in enumerate(all_terms_expanded)], columns=["qid", "query"])

# return the query column of a dataframe as an array
def query_qid_df_to_array(query_qid_df):
    return query_qid_df["query"].values

# takes the original queries and expands them using the model_similiar_function
# returns a dataframe with the expanded queries
def expand_queries_with_model(queries_to_expand,model_similiar_function,threshold=0.8,max_expansion_word_count=5):
    expanded_title_queries = []
    i = 1
    for index,row in queries_to_expand.iterrows():
        row_query = row["query"]
        row_query_expanded = []
        preprocessed_token_array = preprocess_sentence(row_query,useStemmer=False)
        for token,original_token in preprocessed_token_array:
            try:
                similar_words = model_similiar_function(token)
            except Exception as e:
                print(e)
                similar_words = []
            filtered_similar_words = [word for word, score in similar_words if score > threshold]
            row_query_expanded.append(original_token)
            row_query_expanded = row_query_expanded + filtered_similar_words[:max_expansion_word_count]
        row_query_expanded = " ".join(row_query_expanded)
        expanded_title_queries.append([str(i),row_query_expanded,row_query])
        i = i + 1
    return pd.DataFrame(expanded_title_queries, columns=["qid", "query","query_1"])

# expands the original queries with a given array of tokens
# returns a dataframe with the expanded queries
def expand_queries_with_array_of_tokens(queries_to_expand,token_array):
    expanded_title_queries = []
    i = 1
    for index,row in queries_to_expand.iterrows():
        row_query = row["query"]
        row_query_expanded = row_query + " " + " ".join(token_array)
        expanded_title_queries.append([str(i),row_query_expanded,row_query])
        i = i + 1
    return pd.DataFrame(expanded_title_queries, columns=["qid", "query","query_1"])

# takes a list of tokens
# returns a dataframe with qid and query
def get_qid_query_df_from_list(list):
    return pd.DataFrame([[i+1,str(x)] for i,x in enumerate(list)], columns=["qid", "query"])

# Create new dataframe with qid as int64 column and query as string column
def get_qid_query_df_from_df(df):
    return pd.DataFrame([[int(i),str(x)] for i,x in df.values], columns=["qid", "query"])
