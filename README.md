## Nutribot-Q&A

I am building NutriBot, a Retrieval-Augmented Generation (RAG)-based assistant that answers questions from large nutrition research documents (like the 1200-page HNutrients dataset). It uses document embeddings and a local LLM to provide accurate, context-rich responses.

Goal: Help health professionals, students, and researchers quickly extract insights from large nutrition documents instead of manually searching through them.

## Solution Summary
NutriBot-Q&A allows users to upload any nutrition-related PDF and ask natural language questions about its contents.
The system extracts text, embeds it into vector space using SentenceTransformers, retrieves the most relevant passages, and uses a local LLM (like LLama 3.2:1b) to generate context answers.
All computation runs locally — Ensuring Privacy, speed, cost.

## Tech Stack

Backend / Core Logic: Python, Jupyter Notebook
Libraries: PyMuPDF, spaCy, pandas, numpy, tqdm , fitz
Embeddings: SentenceTransformers (all-mpnet-base-v2)
LLM / AI Models: LLaMA 3.2:1b
Framework: Retrieval-Augmented Generation (RAG)
Environment: Anaconda + Jupyter Notebook
Version Control: Git + GitHub

## Project Structure
NutriBot-Q&A/
├── RAG_Q&A.ipynb           # Main Jupyter notebook (core code and analysis)
├── data/                   # HNutrition.pdf
├── embeddings/             # Saved embeddings (CSV)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── youtube_demo.mp4        # Demo video showing working model

## Setup Instructions (with Conda)
Follow these steps to run NutriBot-Q&A locally:

1. Clone the repository
git clone https://github.com/HarshatMandail/nutribot-ai-harshat-mandial-12201051.git
cd honeyenv

2. Create and activate environment
conda create -n honeyenv python=3.10 -y
conda activate honeyenv

3. Install dependencies
pip install -r requirements.txt

4. Launch Jupyter Notebook
jupyter notebook

## Demo Video
YouTube Link:
https://youtu.be/ffilH2AonvY

## Features
-Extracts and preprocesses large PDF documents.
-Embeds text into semantic vectors using all-mpnet-base-v2
-Retrieves relevant information for any query
-Generates context-rich answers using a local LLM
-Works offline ensuring privacy and security

## Technical Architecture
High-level flow:

User Query/quey_list
   ↓
PDF Text Extraction (PyMuPDF)
   ↓
Text Chunking & Embeddings (SentenceTransformers)
   ↓
Vector Retrieval (Cosine Similarity)
   ↓
LLM (LLama 3.2:1b) Generates Contextual Answer
   ↓
Response Displayed in Notebook 

## References
[PyMuPDF](https://github.com/pymupdf/PyMuPDF)
[SentenceTransformers](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)


## Acknowledgements
-AI Assistance:This project was developed by Harshat Mandial with partial assistance from Perplexity(For streamlit code) and ChatGPT  for code debugging, optimization, and documentation 
-Dataset Source:Thanks to the authors of the **HNutrients PDF** used for building and testing the RAG model.  
-Hackathon Organizers:Thanks to **CloudCosmos** for providing the platform and opportunity to present this AI-driven project. 
