# 🤖 AmbedkarGPT-Intern-Task

A simple command-line Q&A system built using the **LangChain** framework to demonstrate a Retrieval-Augmented Generation (RAG) pipeline. The system answers questions based **solely** on the text provided in `speech.txt`, an excerpt from Dr. B.R. Ambedkar's "Annihilation of Caste."

## 🚀 System Architecture

The pipeline follows these steps using completely open-source and local components:
1.  **Loading & Splitting:** `speech.txt` is loaded and split into chunks.
2.  **Embedding & Storage:** **HuggingFaceEmbeddings** (`all-MiniLM-L6-v2`) are used to create vector embeddings, which are stored locally in **ChromaDB**.
3.  **Retrieval:** The user's question is converted to an embedding, and the **ChromaDB** retrieves the most relevant text chunks.
4.  **Generation:** The retrieved chunks and the original question are passed to the **Mistral 7B** LLM running via **Ollama** to generate a final, grounded answer.

## 🛠️ Setup and Installation

### Prerequisites

You must have **Python 3.8+** installed.

### 1. Ollama Setup

The LLM component (**Mistral 7B**) requires **Ollama**.

1.  **Install Ollama:** Follow the installation instructions for your operating system from the official Ollama website.
    * *Quick Install (Linux/macOS):* `curl -fsSL https://ollama.ai/install.sh | sh`
2.  **Pull the Mistral Model:**
    ```bash
    ollama pull mistral
    ```
3.  **Crucially: Ensure Ollama is running** in the background before running the Python script.

### 2. Python Environment and Dependencies

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # .venv\Scripts\activate  # Windows
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Data File

Ensure the `speech.txt` file is present in the root directory with the following content:
```text
(Content of speech.txt as provided in the assignment)