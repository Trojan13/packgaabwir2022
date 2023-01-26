import os
import pyterrier as pt
from pyterrier.measures import *

# Helper function to intialize multiple indices
# Prepares the index path and avoid errors with already existing indices
# Takes the index name and the index path as input
# Returns the index path
index_count = 0
def prepare_index_path(indexName, index_path):
    global index_count
    index_count = index_count + 1
    index_path = index_path + indexName + str(index_count)

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
    index_path = prepare_index_path(indexName)
    indexer = pt.IterDictIndexer(
        index_path, overwrite=True, blocks=True)
    indexer.setProperty(
        "stopwords.filename", os.path.abspath("en.txt"))
    index_created = indexer.index(dataset.get_corpus_iter(),
                                  fields=['title', 'doi', 'abstract'],
                                  meta=('docno',))
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
