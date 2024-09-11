# HERA Job Description Classification using AI/ML Models

This project is an AI-powered system designed to classify job descriptions based on HERA (Higher Education Role Analysis) elements, focusing on the Single Response (Knowledge and Experience) and Linear Response (Communication) elements. The project utilizes Random Forest and BERT (Bidirectional Encoder Representations from Transformers) models to automate and enhance the efficiency of job evaluation tasks.
Project Structure
Main Files:

   **best_model_random_forest.pkl** — This file contains the pre-trained Random Forest model for the Single Response element (Knowledge and Experience) evaluation.
   
   **best_bert_model.pth** — This is the fine-tuned BERT model, designed for the Linear Response element (Communication) evaluation.

   **data.csv** — The CSV file containing the job descriptions and evaluation scores. This dataset was extracted and processed from PDF job descriptions provided by Leeds Trinity University.

   **data_new.csv** — An updated version of the job descriptions dataset, specifically processed for Linear Response evaluation.
   
   **tfidf_vectorizer.pkl** — The pre-trained TF-IDF Vectorizer used for feature extraction from job descriptions for the Random Forest model.
   
   **model.ipynb** — A Jupyter notebook containing the machine learning pipeline for the Single Response element (Random Forest).
   
   **model.ipynb** (within Linear Response folder) — Jupyter notebook containing the BERT-based model pipeline for Linear Response (Communication).

   **.gitattributes** and .DS_Store — Standard git and system files.

## Folder Structure:

    Linear Response/ — This folder contains files related to the BERT model implementation, specifically for evaluating communication-related job descriptions using HERA elements.
    
        best_bert_model.pth — The saved BERT model file.

        data_new.csv — The dataset containing job descriptions with a focus on Linear Response evaluation.

        model.ipynb — Jupyter notebook for Linear Response model pipeline.
        
        .DS_Store — System file.

        .gitattributes** — Git attributes file.

**How to Use**
### Prerequisites

   ### Python 3.x
   ### Required libraries:
        scikit-learn
        pandas
        numpy
        pytorch
        transformers
        nltk
        matplotlib

You can install all dependencies by running: 
`pip install -r requirements.txt`

### Running the Models

    ### Random Forest Model for Knowledge and Experience (Single Response):
    
        Open the model.ipynb notebook.
        
        Load the dataset data.csv.
        
        Load the TF-IDF Vectorizer (tfidf_vectorizer.pkl) and the Random Forest model (best_model_random_forest.pkl).
        
        Execute the cells to preprocess data, extract features using TF-IDF, and classify job descriptions.

    ### BERT Model for Communication (Linear Response):
    
        Navigate to the Linear Response/ folder.
        
        Open the model.ipynb file.
        
        Load the dataset data_new.csv.
        
        Load the BERT model (best_bert_model.pth).
        
        Execute the cells to preprocess the data and evaluate job descriptions using BERT embeddings.

### Evaluating Results

    The results for both models include accuracy, precision, recall, and F1-score for job classification based on HERA elements.
