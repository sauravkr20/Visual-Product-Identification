
# Visual-Product-Identification

A complete visual product identification system using CNN embeddings and FAISS for fast similarity search, with a React frontend for image-based product search.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Directory Structure](#directory-structure)  


---

## Project Overview

This repository contains a full-stack solution for visual product identification:

- **Backend:** FastAPI app that extracts image embeddings using a CNN model, indexes them with FAISS, and serves search and product metadata APIs.
- **Frontend:** React app that allows users to upload images, performs similarity search, and displays grouped search results with detailed product info.
- **Dataset utilities:** Scripts to filter and prepare shoe product images and metadata.

---

## Features

- Image embedding extraction with CNN  
- Fast similarity search using FAISS  
- Product metadata integration  
- React frontend with image preview, search, and modal detail view  
- Easy addition of new images and embeddings  
- CORS enabled for frontend-backend communication  
- Static file serving for product images  

---

## Directory Structure

```
.
├── 2_cnn_fais_soln
│ ├── app # FastAPI backend source code
│ │ ├── config.py # Configuration variables and paths
│ │ ├── init.py
│ │ ├── main.py # FastAPI app entrypoint
│ │ ├── model.py # CNN model and embedding extraction
│ │ ├── pycache
│ │ ├── requirements.txt # Backend dependencies
│ │ ├── search.py # FAISS index loading and search logic
│ │ └── startup.py # Index building and startup scripts
│ ├── README.md # Backend-specific docs (optional)
│ └── run_startup.py # Script to build index and prepare data
├── 3_frontend
│ └── image-search-frontend # React frontend source code
│ ├── node_modules
│ ├── package.json # Frontend dependencies and scripts
│ ├── package-lock.json
│ ├── public # Static public assets
│ ├── README.md # Frontend-specific docs (optional)
│ └── src # React components and styles
├── asn_env # Python virtual environment (gitignored recommended)
├── dataset_download
│   └── src # Dataset preparation scripts
│   ├── filter_images.py # Image filtering utilities
│   ├── listings_filter.py # Listing filtering utilities
│   └── shoe_products.json # Raw product metadata JSON
├── README.md # This file: project overview and setup
└── requirements.txt # Global Python dependencies (optional)
|__ data

```
//// the data folder is to  be downloaded from the link given ////

---

## Setup & Installation

### Prerequisites

- Python 3.10
- Node.js v20.12.0 and npm/yarn 10.5.0 
- Git  

### Backend Setup

1. Create and activate a Python virtual environment (recommended):
```
python3 -m venv asn_env
source asn_env/bin/activate # Linux/macOS

or
asn_env\Scripts\activate # Windows

```
2. Install backend dependencies:
```
pip install -r 2_cnn_fais_soln/app/requirements.txt
```
3. (Optional) Install global dependencies if `requirements.txt` exists at root:
```
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend folder:
```
cd 3_frontend/image-search-frontend
```

2. Install frontend dependencies:

```
npm install
```

---

## Running the Backend

Start the FastAPI server (default port 5000):


- The API will be accessible at `http://localhost:5000`.
- Static images served under `/images/`.

---

## Running the Frontend

Start the React development server:
```
npm start

```

## Usage

- Upload an image in the frontend to perform a similarity search.
- Results are grouped by product ID with colored borders.
- Click on any image to view detailed product metadata in a modal.
- Add new images via the backend `/add/` endpoint if needed.

---



## Adding New Images

Use the backend `/add/` API to add new images and update the FAISS index:

- Send a POST request with an image file and optional `image_path`.
- The backend saves the image, extracts embeddings, updates the index, and persists metadata.

---

