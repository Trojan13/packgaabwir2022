# Implementation and Evaluation of Local Word Embeddings for Query Expansion in PyTerrier

This repository contains the code for a research project that implements and evaluates local word embeddings based on co-authorship and citations for query expansion in PyTerrier on the TREC-Covid dataset.

## Approach
The approach used in this project is to generate local word embeddings for each document in the TREC-Covid dataset based on the co-authorship and citation information associated with the document. These embeddings are then used to expand the original query by adding the most similar terms from the document's embedding.

The main file of the project is main.ipynb where the complete pipeline is described and implemented, it includes the following steps:

1. Preprocessing of the TREC-Covid dataset, including cleaning and normalizing the text data and extracting the co-authorship and citation information.
2. Training the local and global word2vec model
3. Standard query to the index for first ranking
   1. Retrieval of Top-k documents
4. Getting terms from references
5. Getting terms from co-authors
6. Ranking all terms
7. Retrain the model
8. Select new terms and merge
9. Evaluate

# Usage
1. Clone this repository to your local machine.
2. Open den `main.ipynb` File
3. Run the 'Setup Code'-Part
4. Run the approach.ipynb file using Jupyter Notebook.
5. Follow the instructions in the notebook to reproduce the results.


# Files
```{r}
.
└── Filestructure/
    ├── main.ipynb - Main file
    ├── main.py - main file, but as a single, executable python file
    ├── ir_utils.py - Contains all the necessary functions for `main.ipynb` and `main.py`
    ├── images/
    │   ├── dates.png - Analysis of the cord/covid-19 publish dates
    │   ├── most_common.png - Analysis of the cord/covid-19 most common tokens
    │   └── tokensimilarity.png - Visualization of the trained word2vec model token similarities
    ├── old_files/
    │   ├── analysis.ipynb - Initial analysis of the cord/covid-19 dataset
    │   ├── approach.ipynb - Testing notebook, creating all functions from `ir_utils.py`
    │   └── questions.ipynb - Snippets we had questions about to send to our tutor
    └── models - Contains the trained word2vec models
   ```



# Results
See our paper. [LINK WILL FOLLOW]
