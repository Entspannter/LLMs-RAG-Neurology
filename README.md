# Evaluating Base and Retrieval-Augmented Large Language Models With Document or Online-Supported Support for Evidence-Based Neurology

This is the GitHub repository of the manuscript titled "Evaluating Base and Retrieval-Augmented Large Language Models With Document or Online-Supported Support for Evidence-Based Neurology" currently under revision for the publication in npj Digital Medicine. 

## Credit
Part of this work builds on a recent paper from Ferber et al. with the retrieval-augmented generation being adapted from their paper (Ferber, Dyke & Wiest, Isabella & Wölflein, Georg & Ebert, Matthias & Beutel, Gernot & Eckardt, Jan-Niklas & Truhn, Daniel & Springfeld, Christoph & Jäger, Dirk & Kather, Jakob. (2024). GPT-4 for Information Retrieval and Comparison of Medical Oncology Guidelines. NEJM AI. 1. 10.1056/AIcs2300235.) and the respective codebase https://github.com/Dyke-F/RAG_Medical_Guidelines/.


## Setup Instructions

**Note:** Two separate virtual environments are used to handle differing package requirements:
- `venv` for RAG generation exactly as implemented in above-mentioned paper (`run_rag_script.ipynb`)
- `new_venv` for all other Jupyter notebooks.

**Python Installation**: Install Python from source. We tested the setup with both current 3.10 and 3.11 python versions.  

Set up the environments as follows:

### `venv` (RAG Generation)
1. Create and activate:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Use .\venv\Scripts\activate on Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Deactivate:
   ```bash
   deactivate
   ```

### `new_venv` (Other Notebooks and Analysis)
1. Create and activate:
   ```bash
   python -m venv new_venv
   source new_venv/bin/activate  # Use .\new_venv\Scripts\activate on Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements_new.txt
   ```
3. Deactivate:
   ```bash
   deactivate
   ```

## Usage
Activate the required environment before running:
- **RAG generation**: Activate `venv`
- **Other notebooks**: Activate `new_venv`
  
## Env variables
Make sure to set your own API keys in the .env in order for the notebooks to work. 

## Dataset
The dataset of neurological questions, answers, and ratings is available as Excel sheet in the [Results folder](Results/All_Answers_Full_DF_annotated_all.xlsx).

## Data Preparation
For the exact RAG setup see the original paper. 

## Jupyter Notebooks
- **[run_rag_script.ipynb](run_rag_script.ipynb)**: Used for running the RAG (Retrieval-Augmented Generation) script.
- **[query_datasets_without_rag.ipynb](query_datasets_without_rag.ipynb)**: Used for querying datasets without using RAG.
- **[combine_all_datasets.ipynb](combine_all_datasets.ipynb)**: Combines all datasets into a single DataFrame. This was then distributed to the raters. 
- **[visualisations.ipynb](visualisations.ipynb)**: Generates visualizations and statistical analyses for the data.


