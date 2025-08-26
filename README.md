# Medical RAG System

A Retrieval-Augmented Generation (RAG) system for clinical notes and diagnostic knowledge graphs, using Google Gemini, FAISS, and Sentence Transformers.

---

## Features

- Extracts and processes clinical notes and diagnostic knowledge graphs from RAR files.
- Converts nested JSON files into structured CSVs for easier data handling.
- Cleans and prepares text for embeddings using `SentenceTransformer`.
- Builds a FAISS index for fast document retrieval.
- Generates responses using Google Gemini API based on retrieved documents.
- Provides a Gradio interface for interactive testing and evaluation of queries.
- Saves embeddings, FAISS index, and document mappings for reuse.

---

## Installation

Install required packages:

```bash
pip install rarfile
apt-get install -y unrar

---

## Usage

1. **Extract RAR files**:

```python
extract_rar('/content/Finished.rar', '/content/extracted')
extract_rar('/content/diagnostic_kg.rar', '/content/extracted')
```

2. **Load and process data**:

```python
clinical_notes = load_finished_notes('/content/extracted/Finished')
knowledge_graphs = load_diagnostic_kg('/content/extracted/diagnostic_kg')
```

3. **Convert JSON data to CSV**:

```python
process_finished_data()
process_diagnostic_kg_data()
```

4. **Prepare embeddings and FAISS index**:

```python
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode([doc['content'] for doc in all_docs])
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype('float32'))
```

5. **Generate responses using Google Gemini**:

```python
os.environ["GOOGLE_API_KEY"] = "<YOUR_API_KEY>"
response = generate_response("What are the risk factors for Pulmonary Embolism?", retrieved_docs)
```

6. **Launch Gradio interface**:

```python
gr.Interface(
    fn=evaluate_rag_system_interface,
    inputs=gr.Textbox(...),
    outputs=gr.Markdown(...),
    title="🧠 RAG System Evaluator",
    description="Evaluate your medical RAG system."
).launch(share=True)
```

---

## File Structure

```
/content/extracted/
├── Finished/          # Clinical notes JSON files
├── diagnostic_kg/     # Diagnostic knowledge graph JSON files
├── finished_data.csv
├── diagnostic_kg_data.csv
├── medical_docs_faiss.index
├── all_docs.pkl
├── embedding_info.json
```

---

## Technologies Used

* **Python** for data processing and scripting
* **FAISS** for fast similarity search
* **Sentence Transformers** for embeddings
* **Google Gemini API** for language generation
* **Gradio** for interactive evaluation
* **RAR/JSON** extraction and processing

---



