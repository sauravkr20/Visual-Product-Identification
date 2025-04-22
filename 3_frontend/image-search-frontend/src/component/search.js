import React, { useState } from "react";
import axios from "axios";
import {
  Box,
  Button,
  CircularProgress,
  Container,
  Grid,
  Typography,
  Card,
  CardMedia,
  CardContent,
} from "@mui/material";

import SearchResults from "./search_results";

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

      {/* {results.length > 0 && (
        <>
          <Typography variant="h5" gutterBottom>
            Results:
          </Typography>
          <Grid container spacing={3}>
            {results.map((item, idx) => {
              const img = item.image_path; // This is the nested object
              return (
                <Grid item xs={12} sm={6} md={4} key={idx}>
                  <Card>
                    <CardMedia
                      component="img"
                      height="200"
                      image={`http://localhost:5000/images/${img.image_path}`} // construct URL from relative path
                      alt={`Result ${idx + 1}`}
                    />
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">
                        <strong>Image ID:</strong> {img.image_id || "N/A"}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        <strong>Item ID:</strong> {img.item_id || "N/A"}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        <strong>Score:</strong> {item.score?.toFixed(4)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </>
      )}
       */}

      <SearchResults results={results} />
    </Container>
  );
}

export default Search;
