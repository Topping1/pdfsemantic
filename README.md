# PDF Semantic Viewer

## TL;DR
This simple app allows users to test Qwen3 embeddings in a PDF semantic search tool. It provides a graphical interface to load PDFs, perform semantic searches, and visualize the results.

## Screenshot
![image](https://github.com/user-attachments/assets/d6d3ce8a-78f7-44ca-b4e2-e464aadb9c8e)


## Detailed Description
The PDF Semantic Viewer is a PyQt5-based application that leverages Qwen3 embeddings to perform semantic searches within PDF documents. The application allows users to load PDF files, embed the text content using Qwen3 embeddings, and perform semantic searches to find relevant sections of the document. The search results are displayed in a list, and the relevant sections are highlighted in the PDF viewer.

## Features
- Load and view PDF documents.
- Perform semantic searches using Qwen3 embeddings.
- Highlight search results within the PDF.
- Navigate through the PDF using a graphical interface.
- Zoom in and out of the PDF.

## Requirements
- Python 3.x
- PyQt5
- numpy
- requests
- pymupdf (fitz)

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/Topping1/pdfsemantic.git
    cd pdfsemantic
    ```

2. Install the required packages:
    ```sh
    pip install PyQt5 numpy requests pymupdf
    ```

## Usage

### Step 1: Serve Qwen3 Embeddings with llama-server
To use Qwen3 embeddings, you need to serve them using `llama-server` from `llama.cpp`. Follow these steps to set up the server:

1. Download the Qwen3 Embedding model (this example uses the 4B model but any other Qwen3 Embedding model works) from [Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF).

2. Serve the model using `llama-server`. Make sure you have `llama.cpp` installed and the model file is accessible:
    ```sh
    ./llama-server -m /path/to/Qwen3-Embedding-4B-Q8_0.gguf --embedding --pooling last -ub 8192 --verbose-prompt

    ```
    This command starts the server with the specified model file, with the recommended settings by Qwen.

### Step 2: Run the PDF Semantic Viewer
1. Ensure the `llama-server` is running and serving the Qwen3 embeddings.

2. Run the PDF Semantic Viewer script:
    ```sh
    python pdf-semantic.py
    ```

### Step 3: Perform Semantic Searches
1. Open a PDF file using the "Open" button in the toolbar.
2. Enter a search query in the text box and press "Search" or hit Enter.
3. The search results will be displayed in the list on the right. Click on a result to highlight the relevant section in the PDF.

## Code Overview
The main components of the application are:

- **Embedding Helpers**: Functions to generate embeddings using the Qwen3 model.
- **Chunk Helpers**: Functions to split text into chunks for embedding.
- **Worker Thread**: A QThread subclass to handle the embedding process in the background.
- **GUI**: The main application window with a toolbar, PDF viewer, search box, and results list.

## Others
This app could potentially be used with other embeddings that can be served with llama-server, but some minor code modifications might be needed.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [Qwen3 Embedding Model](https://huggingface.co/collections/Qwen/qwen3-embedding-6841b2055b99c44d9a4c371f)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
- [pymupdf](https://pymupdf.readthedocs.io/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
