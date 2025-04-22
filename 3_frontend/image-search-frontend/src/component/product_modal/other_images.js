import React from "react";
import { Box, Typography, Grid } from "@mui/material";

export default function OtherImages({ otherImages, baseUrl }) {
  if (!otherImages || otherImages.length === 0) return null;

  return (
    <Box>
      <Typography variant="subtitle1" gutterBottom>
        Other Images
      </Typography>
      <Grid container spacing={2}>
        {otherImages.map((img) => (
          <Grid item xs={4} sm={3} md={2} key={img.image_id}>
            <Box
              component="img"
              src={baseUrl + img.image_path}
              alt={`Other ${img.image_id}`}
              sx={{
                width: "100%",
                height: 100,
                objectFit: "cover",
                borderRadius: 1,
                border: "1px solid #ccc",
              }}
            />
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
