from langdetect import detect
from unidecode import unidecode
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from krovetzstemmer import Stemmer
from semanticscholar import SemanticScholar
from gensim.models import Word2Vec
from itertools import islice, chain
from tqdm import tqdm
from pyterrier.measures import *
from pymed import PubMed
import requests
import os
import pyterrier as pt
import pandas as pd
import numpy as np
import nltk
import time
import string

# Global variables
# ----------------
# Path to the data folder

# Path to the data folder
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../indices"))
INDICES_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../indices"))

def initialize():
    pubmed = PubMed(tool="Trec-Covid-Local-QE-Tool", email="tim_colin.pack1@smail.th-koeln.de")

    krovetz_stemmer = Stemmer()
    regex_tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
    semantic_scholar = SemanticScholar()

    if not pt.started():
        pt.init()
    return pubmed, krovetz_stemmer, regex_tokenizer, semantic_scholar
