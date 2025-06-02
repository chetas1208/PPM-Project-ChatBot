# PPM Project Chatbot

## Overview

This repository contains a full-stack chatbot project for academic transfer query classification. It includes:

* **Backend**: Python-based FastAPI server utilizing Qdrant for vector storage and Hugging Face for zero-shot classification.
* **Frontend**: React.js application to interact with the backend.
* **Data**: A CSV file (`transfer_prompts.csv`) containing labeled prompts.

## Prerequisites

* **Python 3.9+**
* **Node.js 14+ and npm 6+**
* **Git**
* **Virtual environment (optional but recommended)**

## Repository Structure

```
├── backend/
│   ├── api.py
│   ├── classify.py
│   ├── client.py
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── requirements.txt
│   └── transfer_prompts.csv
├── frontend/
│   ├── package.json
│   ├── public/
│   └── src/
│       └── App.js
├── .gitignore
└── README.md
```

## Installation

### Backend Setup

1. Navigate to the backend folder:

   ```bash
   cd backend
   ```
2. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```
3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in `backend/` with the following variables:

   ```env
   HUGGINGFACE_API_TOKEN=<your_hf_token>
   QDRANT_URL=<your_qdrant_url>
   QDRANT_API_KEY=<your_qdrant_key>
   COLLECTION_NAME=intent_identification
   ```
5. Ensure `transfer_prompts.csv` is in the `backend/` folder.

### Frontend Setup

1. Navigate to the frontend folder:

   ```bash
   cd frontend
   ```
2. Install Node.js dependencies:

   ```bash
   npm install
   ```

## Running the Application

### Start Backend

1. From the `backend/` directory, run:

   ```bash
   uvicorn api:app --reload
   ```

   * This will upload the CSV data to Qdrant and start an interactive prompt loop.

### Start Frontend

1. Open a new terminal and navigate to `frontend/`:

   ```bash
   cd frontend
   ```
2. Run the development server:

   ```bash
   npm start
   ```
3. Open your browser to `http://localhost:3000` (or the port shown) to interact with the chatbot frontend.

## Project Workflow

1. **Backend**:

   * Loads environment variables and connects to Qdrant.
   * Reads `transfer_prompts.csv`, computes embeddings, and upserts to Qdrant.
   * Listens for user queries, retrieves similar prompts from Qdrant, or uses zero-shot classification if no match.
   * Prompts for user feedback and updates Qdrant accordingly.
   * Prints a classification report (precision, recall, F1, support) after each query.

2. **Frontend**:

   * Provides a single input box for users to enter queries.
   * Sends HTTP POST requests to the backend `/query` endpoint.
   * Displays predicted intent and source (database or model).

## Environment Variables Reference

* **HUGGINGFACE\_API\_TOKEN**: Token for Hugging Face model access.
* **QDRANT\_URL**: URL of the Qdrant vector database instance.
* **QDRANT\_API\_KEY**: API key for Qdrant authentication.
* **COLLECTION\_NAME**: Name of the Qdrant collection (`intent_identification`).

## License

This project is open source and available under the MIT License.
