import React from "react";
import { Box, Typography } from "@mui/material";

export default function MainImage({ mainImage, baseUrl }) {
  if (!mainImage) return null;

  return (
    <Box mb={3} textAlign="center">
      <Typography variant="subtitle1" gutterBottom>
        Main Image
      </Typography>
      <Box
        component="img"
        src={baseUrl + mainImage.image_path}
        alt="Main Product"
        sx={{ maxWidth: "100%", maxHeight: 300, borderRadius: 1, border: "1px solid #ccc" }}
      />
    </Box>
  );
}
