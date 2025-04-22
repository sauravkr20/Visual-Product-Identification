from app.startup import build_cnn_faiss_index, build_products_col


if __name__ == "__main__":
    build_products_col()


    build_cnn_faiss_index()
