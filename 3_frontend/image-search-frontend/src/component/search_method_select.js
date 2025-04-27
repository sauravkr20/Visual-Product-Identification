import React from "react";
import { FormControl, InputLabel, Select, MenuItem } from "@mui/material";

export default function SearchMethodSelect({ method, setMethod }) {
  return (
    <FormControl fullWidth margin="normal">
      <InputLabel id="search-method-label">Search Method</InputLabel>
      <Select
        labelId="search-method-label"
        value={method}
        label="Search Method"
        onChange={(e) => setMethod(e.target.value)}
      >
        <MenuItem value="cnn_faiss">CNN + FAISS</MenuItem>
        <MenuItem value="clip_faiss">CLIP + FAISS</MenuItem>
        {/* Add more */}
      </Select>
    </FormControl>
  );
}
