import React, { useState } from "react";
import axios from "axios";
import {
  Box,
  Button,
  CircularProgress,
  Container,
  Typography,
} from "@mui/material";

import SearchResults from "./search_results";
import SearchMethodSelect from "./search_method_select";  
function Search() {
  const [file, setFile] = useState(null);
  const [method, setMethod] = useState("cnn_faiss");  
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
    formData.append("method", method);  
    formData.append("top_k", 5);

    setLoading(true);
    try {
      const response = await axios.post("http://localhost:5000/search/", formData, {
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
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Image Search
      </Typography>

      <SearchMethodSelect method={method} setMethod={setMethod} />

      <Box display="flex" alignItems="center" gap={2} mb={3}>
        <input
          type="file"
          onChange={handleFileChange}
          accept="image/*"
          style={{ flex: 1 }}
        />
        <Button
          variant="contained"
          onClick={handleSearch}
          disabled={loading}
          sx={{ minWidth: 120 }}
        >
          {loading ? <CircularProgress size={24} color="inherit" /> : "Search"}
        </Button>
      </Box>

      {previewUrl && (
        <Box mb={4}>
          <Typography variant="subtitle1">Preview:</Typography>
          <Box
            component="img"
            src={previewUrl}
            alt="Selected preview"
            sx={{ maxWidth: "100%", maxHeight: 300, borderRadius: 1, border: "1px solid #ccc" }}
          />
        </Box>
      )}

      <SearchResults results={results} method={method} />
    </Container>
  );
}

export default Search;
