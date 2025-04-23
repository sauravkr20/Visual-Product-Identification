import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  InputLabel,
  Grid,
} from "@mui/material";

const AddProduct = () => {
  const [itemId, setItemId] = useState("");
  const [productType, setProductType] = useState("");
  const [itemName, setItemName] = useState("");
  const [mainImage, setMainImage] = useState(null);
  const [mainImagePreview, setMainImagePreview] = useState(null);
  const [otherImages, setOtherImages] = useState([]);
  const [otherImagesPreview, setOtherImagesPreview] = useState([]);
  const [response, setResponse] = useState("");

  // Generate preview URL for main image
  useEffect(() => {
    if (!mainImage) {
      setMainImagePreview(null);
      return;
    }
    const objectUrl = URL.createObjectURL(mainImage);
    setMainImagePreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [mainImage]);

  // Generate preview URLs for other images
  useEffect(() => {
    if (otherImages.length === 0) {
      setOtherImagesPreview([]);
      return;
    }
    const objectUrls = otherImages.map((file) => URL.createObjectURL(file));
    setOtherImagesPreview(objectUrls);
    return () => {
      objectUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [otherImages]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!mainImage) {
      setResponse("Main image is required");
      return;
    }

    const formData = new FormData();
    formData.append("item_id", itemId);
    formData.append("product_type", productType);
    formData.append(
      "item_name",
      JSON.stringify([{ language_tag: "en", value: itemName }])
    );
    formData.append("main_image", mainImage, `${itemId}_main_${mainImage.name}`);
    otherImages.forEach((file, idx) =>
      formData.append("other_images", file, `${itemId}_other_${idx}_${file.name}`)
    );

    setResponse("Uploading...");

    try {
      const res = await fetch("http://localhost:5000/add_product", {
        method: "POST",
        body: formData, // Important: do NOT set Content-Type header manually
      });

      const data = await res.json();
      if (res.ok) {
        setResponse("Product added successfully!");
        // Reset form
        setItemId("");
        setProductType("");
        setItemName("");
        setMainImage(null);
        setOtherImages([]);
      } else {
        setResponse(`Error: ${data.detail || JSON.stringify(data)}`);
      }
    } catch (err) {
      setResponse("Network error: " + err.message);
    }
  };

  return (
    <Paper sx={{ p: 3, maxWidth: 600, mx: "auto" }}>
      <Typography variant="h5" gutterBottom>
        Add Product
      </Typography>
      <Box component="form" onSubmit={handleSubmit} noValidate>
        <TextField
          label="Item ID"
          value={itemId}
          onChange={(e) => setItemId(e.target.value)}
          fullWidth
          margin="normal"
          required
        />
        <TextField
          label="Product Type"
          value={productType}
          onChange={(e) => setProductType(e.target.value)}
          fullWidth
          margin="normal"
          required
        />
        <TextField
          label="Item Name"
          value={itemName}
          onChange={(e) => setItemName(e.target.value)}
          fullWidth
          margin="normal"
          required
          helperText="This will be wrapped as JSON for the backend"
        />

        <InputLabel sx={{ mt: 2 }}>Main Image</InputLabel>
        <input
          type="file"
          accept="image/*"
          required
          onChange={(e) => setMainImage(e.target.files[0])}
          style={{ marginBottom: 8 }}
        />
        {mainImagePreview && (
          <Box sx={{ mb: 2 }}>
            <img
              src={mainImagePreview}
              alt="Main Preview"
              style={{ maxWidth: "100%", maxHeight: 200, borderRadius: 4 }}
            />
          </Box>
        )}

        <InputLabel>Other Images</InputLabel>
        <input
          type="file"
          accept="image/*"
          multiple
          onChange={(e) => setOtherImages([...e.target.files])}
          style={{ marginBottom: 8 }}
        />
        {otherImagesPreview.length > 0 && (
          <Grid container spacing={2} sx={{ mb: 2 }}>
            {otherImagesPreview.map((src, idx) => (
              <Grid item key={idx} xs={4} sm={3}>
                <img
                  src={src}
                  alt={`Other Preview ${idx + 1}`}
                  style={{ width: "100%", height: "auto", borderRadius: 4 }}
                />
              </Grid>
            ))}
          </Grid>
        )}

        <Button variant="contained" color="primary" type="submit" sx={{ mt: 2 }}>
          Submit
        </Button>

        <Typography
          variant="body2"
          sx={{ mt: 2, color: response.startsWith("Error") ? "red" : "green" }}
        >
          {response}
        </Typography>
      </Box>
    </Paper>
  );
};

export default AddProduct;
