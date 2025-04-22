import React, { useState } from "react";
import {
  Box,
  Card,
  CardMedia,
  Typography,
  Grid,
  Modal,
  Backdrop,
  Fade,
  CircularProgress,
} from "@mui/material";
import axios from "axios";

const IMAGE_BASE_URL = "http://localhost:5000/images/";

export default function SearchResults({ results }) {
  const [open, setOpen] = useState(false);
  const [productJson, setProductJson] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleOpen = async (itemId) => {
    setLoading(true);
    setOpen(true);
    try {
      const res = await axios.get(`http://localhost:5000/products/${itemId}`);
      setProductJson(res.data);
    } catch (e) {
      setProductJson({ error: "Product not found" });
    }
    setLoading(false);
  };

  const handleClose = () => {
    setOpen(false);
    setProductJson(null);
  };

  return (
    <>
      <Grid container spacing={3}>
        {results.map((item, idx) => {
          const img = item.image_path;
          return (
            <Grid item xs={12} sm={6} md={4} key={idx}>
              <Card
                sx={{
                  position: "relative",
                  cursor: "pointer",
                  "&:hover .overlay": { opacity: 1 },
                }}
                onClick={() => handleOpen(img.item_id)}
              >
                <CardMedia
                  component="img"
                  height="200"
                  image={IMAGE_BASE_URL + img.image_path}
                  alt={`Result ${idx + 1}`}
                />
                <Box
                  className="overlay"
                  sx={{
                    position: "absolute",
                    bottom: 0,
                    left: 0,
                    width: "100%",
                    bgcolor: "rgba(0,0,0,0.6)",
                    color: "#fff",
                    p: 1,
                    opacity: 1,
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

      <Modal
        open={open}
        onClose={handleClose}
        closeAfterTransition
        BackdropComponent={Backdrop}
        BackdropProps={{ timeout: 300 }}
      >
        <Fade in={open}>
          <Box
            sx={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              bgcolor: "background.paper",
              boxShadow: 24,
              p: 4,
              maxWidth: 600,
              width: "90vw",
              maxHeight: "80vh",
              overflow: "auto",
              borderRadius: 2,
            }}
          >
            <Typography variant="h6" gutterBottom>
              Product JSON
            </Typography>
            {loading ? (
              <Box display="flex" justifyContent="center" alignItems="center">
                <CircularProgress />
              </Box>
            ) : (
              <pre style={{ fontSize: 14, whiteSpace: "pre-wrap" }}>
                {JSON.stringify(productJson, null, 2)}
              </pre>
            )}
          </Box>
        </Fade>
      </Modal>
    </>
  );
}
