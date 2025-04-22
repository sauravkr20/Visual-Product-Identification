import numpy as np
import cv2

class CNNSIFTHybridSearch:
    def __init__(self, index, embedding_metadata, extract_cnn_func, kmeans_model, search_func):
        self.index = index
        self.embedding_metadata = embedding_metadata
        self.extract_cnn = extract_cnn_func
        self.kmeans_model = kmeans_model
        self.search = search_func

    def extract_sift(self, image):
        gray = np.array(image.convert("L"))
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            return np.zeros((1, 128), dtype=np.float32)
        return descriptors

    def aggregate_sift(self, descriptors, k=64):
        words = self.kmeans_model.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(k+1))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        return hist

    def extract_hybrid(self, image, k=64):
        cnn_feat = self.extract_cnn(image)
        sift_desc = self.extract_sift(image)
        sift_hist = self.aggregate_sift(sift_desc, k)
        return np.concatenate([cnn_feat, sift_hist])

    async def search_image(self, image, top_k=5):
        hybrid_feat = self.extract_hybrid(image)
        indices, scores = self.search(self.index, hybrid_feat, top_k)
        results = []
        for idx, score in zip(indices, scores):
            meta = self.embedding_metadata[idx]
            results.append({
                "image_id": meta["image_id"],
                "item_id": meta.get("item_id"),
                "image_path": meta["image_path"],
                "score": float(score),
            })
        return results
