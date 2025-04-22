from app.startup import build_index_from_folder, build_cnn_faiss_index


if __name__ == "__main__":
    build_index_from_folder()
    
    build_cnn_faiss_index()
