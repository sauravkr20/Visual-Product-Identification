import React from "react";
import { Typography } from "@mui/material";

export default function ProductName({ product }) {
  if (!product || !product.item_name) return null;

  const name = product.item_name[0]?.value ;

  return (
    <Typography variant="h6" gutterBottom>
      {name}
    </Typography>
  );
}
