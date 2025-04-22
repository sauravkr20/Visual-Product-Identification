import React, { useState } from "react";
import {
  Grid,
  Card,
  CardMedia,
  Box,
  Typography,
} from "@mui/material";
import ProductModal from "./product_modal";

const IMAGE_BASE_URL = "http://localhost:5000/images/";

export default function SearchResults({ results }) {
  const [open, setOpen] = useState(false);
  const [selectedItemId, setSelectedItemId] = useState(null);

  const handleOpen = (itemId) => {
    setSelectedItemId(itemId);
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    setSelectedItemId(null);
  };

  return (
    <>
      {results.length>0 && <Typography variant="subtitle1">Responses:</Typography>}
      
      <Grid container spacing={3}>
        {results.map((item, idx) => {
          const img = item;
          return (
            <Grid item xs={12} sm={6} md={4} key={idx}>
              <Card
                sx={{ position: "relative", cursor: "pointer" }}
                onClick={() => handleOpen(img.item_id)}
              >
                <CardMedia
                  component="img"
                  height="200"
                  image={IMAGE_BASE_URL + img.image_path}
                  alt={`Result ${idx + 1}`}
                />
                <Box
                  sx={{
                    position: "absolute",
                    bottom: 0,
                    left: 0,
                    width: "100%",
                    bgcolor: "rgba(0,0,0,0.6)",
                    color: "#fff",
                    p: 1,
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  <Typography variant="body2">
                    <strong>Image ID:</strong> {img.image_id}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Item ID:</strong> {img.item_id}
                  </Typography>
                </Box>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {selectedItemId && (
        <ProductModal open={open} onClose={handleClose} itemId={selectedItemId} />
      )}
    </>
  );
}
