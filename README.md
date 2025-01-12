research news tool.
## Purpose:

Extract content from news article URLs.
Enable users to query information and get answers with sources.
Workflow:

## Input: 
Accepts up to 3 URLs via the sidebar.

## Processing:
Loads article content using UnstructuredURLLoader.
Splits content into manageable chunks and generates embeddings using sentence-transformers/all-mpnet-base-v2.
Stores embeddings in a FAISS vector database.
Query Handling:
Retrieves the stored vector database.
Uses the ChatGroq LLM to answer user queries and display sources.

## Tech Stack:

Streamlit: For the web interface.
LangChain: For document loading, splitting, and QA processing.
FAISS: For efficient storage and retrieval of embeddings.
Hugging Face: For embedding generation.
ChatGroq: For generating query responses.
Output:

Provides answers and displays the sources of information.

## Use Case:
Suitable for researchers or users needing quick insights from multiple news articles.


