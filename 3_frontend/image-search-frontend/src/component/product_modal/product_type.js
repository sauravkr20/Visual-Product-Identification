import React from "react";
import { Typography } from "@mui/material";

export default function ProductType({ productType }) {
  if (!productType || productType.length === 0) return null;

  return (
    <Typography variant="subtitle1" gutterBottom>
      <strong>Product Type:</strong> {productType.join(", ")}
    </Typography>
  );
}
