from gensim.models import Word2Vec
from tqdm import tqdm
from pyterrier.measures import *
from ir_utils import *
import requests
import os
import pandas as pd
import numpy as np
import nltk
import time
import string


def initialize():
    pt = get_pyterrier_instance()

    irds_trec_covid_dataset = pt.datasets.get_dataset('irds:cord19/trec-covid')
    irds_trec_covid_dataset_docs = pd.DataFrame(irds_trec_covid_dataset.get_corpus_iter())
    print('Got dataset and docs...')

    irds_trec_covid_dataset_metadata = pd.read_csv('~/.ir_datasets/cord19/2020-07-16/metadata.csv', low_memory=False)

    print('Creating index...')
    #irds_trec_covid_dataset_index = build_index('qetest', irds_trec_covid_dataset)
    irds_trec_covid_dataset_index = pt.IndexRef.of("F:\Bibliotheken\Desktop\Skripte\packgaabwir2022\indices\qetest1\data.properties")

    irds_trec_covid_dataset_topics_titles = irds_trec_covid_dataset.get_topics('title')
    irds_trec_covid_dataset_topics_qrels = irds_trec_covid_dataset.get_qrels()
    print('Got topics and qrels...')


    print('Starting  preprocessing...')
    irds_trec_covid_dataset_en_docs_preprocessed = preprocess_data(irds_trec_covid_dataset_docs)

    print('Loading models ...')
    initial_word2vec_model = get_initial_word2vec_model(irds_trec_covid_dataset_en_docs_preprocessed)
    #global_word2vec_model = get_global_word2vec_model()

    top_k_docs = get_top_k_docs_bm25(irds_trec_covid_dataset_index, irds_trec_covid_dataset_topics_titles, 1)
    top_k_docs_title_tokens_df = get_query_qid_df_tokens_from_docs_by_field(irds_trec_covid_dataset_docs,top_k_docs) # [qid, query]
    print("Got top",len(top_k_docs),"docs...")


    print('Getting refence titles from sm and pubmed...') 
    top_k_docs_references_titles = get_all_titles_from_references_of_docs_from_sm_and_pubmed(irds_trec_covid_dataset_metadata,irds_trec_covid_dataset_docs,top_k_docs)
    
    print('Getting relevant papers from authors...') 
    top_k_docs_all_authors_relevant_papers_abstracts = get_field_for_all_relevant_authors(irds_trec_covid_dataset_metadata,top_k_docs,2,field="paperAbstract")

    print('Tokens from top k docs...') 
    top_k_docs_tokens_df = get_query_qid_df_from_all_preprocessed_tokens_of_docs(irds_trec_covid_dataset_metadata,top_k_docs,top_k_docs_references_titles,top_k_docs_all_authors_relevant_papers_abstracts,field="abstract")

    print(top_k_docs_tokens_df)
    print('Ranking tokens...') 
    top_k_docs_top_k_tokens_ranked = get_top_k_ranked_tokens_from_dataframe_with_bm25_and_bo1(irds_trec_covid_dataset_index,top_k_docs_tokens_df,2)


    top_k_docs_all_tokenized_sentences = get_tokenized_sentences_for_word2vec(irds_trec_covid_dataset_metadata,top_k_docs,top_k_docs_references_titles,top_k_docs_all_authors_relevant_papers_abstracts)
    
    retrained_word2vec_model = retrain_word2vec_model(initial_word2vec_model,top_k_docs_all_tokenized_sentences)

    top_k_docs_tokens_array = query_qid_df_to_array(top_k_docs_tokens_df)
    print(len(top_k_docs_tokens_array))
    top_k_docs_relevant_tokens_expanded_df = expand_search_terms(top_k_docs_tokens_array,retrained_word2vec_model,threshold=0.8,max_expansion_word_count=5)

    top_k_docs_top_k_tokens_expanded_ranked = get_top_k_ranked_tokens_from_dataframe_with_bm25_and_bo1(irds_trec_covid_dataset_index,top_k_docs_tokens_df,2)
    print(top_k_docs_top_k_tokens_expanded_ranked)


initialize()