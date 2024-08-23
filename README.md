# Gemma Model Demo

This project demonstrates the use of the Gemma-7b-it language model to answer questions based on the content of PDF documents. 
The application uses Streamlit for the interface, LangChain for document processing, and FAISS for efficient document retrieval.

## Prerequisites

Before you begin, ensure you have the following:

- **Python 3.8 or higher** installed on your machine.
- A valid **`GROQ_API_KEY`** and **`GOOGLE_API_KEY`**.
- PDF documents placed in a `data` folder at the root level of this project.

## Installation and Setup

Follow these steps to set up and run the project:

### 1. Clone the Repository

Open a terminal and clone the repository:

```bash
git clone <repository_url>
cd gemma
```
### 2. Create and activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install the Required Packages
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
### 5. Create a folder with pdf files named 'data'
### 6. Run the Application
```bash
streamlit run app.py
```

