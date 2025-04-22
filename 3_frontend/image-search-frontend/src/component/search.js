import React, { useState } from "react";
import axios from "axios";

function Search() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      setPreviewUrl(URL.createObjectURL(selectedFile));
    } else {
      setPreviewUrl(null);
    }
  };

  const handleSearch = async () => {
    if (!file) {
      alert("Please select an image to search");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("top_k", 5);

    setLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/search/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResults(response.data.results);
    } catch (error) {
      console.error("Search error:", error);
      alert("Failed to search. See console for details.");
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: "auto", padding: 20 }}>
      <h2>Image Search</h2>
      <input type="file" onChange={handleFileChange} accept="image/*" />
      {previewUrl && (
        <div style={{ margin: "20px 0" }}>
          <div>Preview:</div>
          <img
            src={previewUrl}
            alt="Selected preview"
            style={{ maxWidth: "100%", maxHeight: 300, border: "1px solid #ccc" }}
          />
        </div>
      )}
      <button onClick={handleSearch} disabled={loading} style={{ marginLeft: 10 }}>
        {loading ? "Searching..." : "Search"}
      </button>

      <div style={{ marginTop: 20 }}>
        {results.length > 0 && <h3>Results:</h3>}
        <ul style={{ listStyle: "none", padding: 0 }}>
          {results.map((item, idx) => (
            <li key={idx} style={{ marginBottom: 20 }}>
              <div><strong>Image ID:</strong> {item.image_id || "N/A"}</div>
              <div><strong>Item ID:</strong> {item.item_id || "N/A"}</div>
              <img
                src={`http://localhost:8000/images/${item.image_path}`}  // Adjust as needed
                alt={`Result ${idx + 1}`}
                style={{ maxWidth: "100%", maxHeight: 200, marginTop: 10 }}
              />
              <div><strong>Score:</strong> {item.score?.toFixed(4)}</div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default Search;
