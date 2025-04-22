import React, { useState, useEffect } from "react";
import {
  Box,
  Typography,
  Modal,
  Backdrop,
  Fade,
  CircularProgress,
} from "@mui/material";
import axios from "axios";

// import LanguageSelector from "./product_modal/language_selector";
import MainImage from "./product_modal/main_image";
import ProductName from "./product_modal/product_name";
import ProductType from "./product_modal/product_type";
import OtherImages from "./product_modal/other_images";

const IMAGE_BASE_URL = "http://localhost:5000/images/";

export default function ProductModal({ open, onClose, itemId }) {
  const [product, setProduct] = useState(null);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    if (!itemId) return;

    setLoading(true);
    setProduct(null);

    axios
      .get(`http://localhost:5000/products/${itemId}`)
      .then((res) => setProduct(res.data))
      .catch(() => setProduct({ error: "Product not found" }))
      .finally(() => setLoading(false));
  }, [itemId]);

  return (
    <Modal
      open={open}
      onClose={onClose}
      closeAfterTransition
      BackdropComponent={Backdrop}
      BackdropProps={{ timeout: 300 }}
      aria-labelledby="product-modal-title"
      aria-describedby="product-modal-description"
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
            maxWidth: 800,
            width: "90vw",
            maxHeight: "80vh",
            overflowY: "auto",
            borderRadius: 2,
          }}
        >
          {loading ? (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
              <CircularProgress />
            </Box>
          ) : product?.error ? (
            <Typography color="error">{product.error}</Typography>
          ) : (
            <>
              <Typography id="product-modal-title" variant="h5" gutterBottom>
                Product Details
              </Typography>

              <ProductName product={product}  />

              <ProductType productType={product?.product_type} />

              <MainImage mainImage={product?.main_image} baseUrl={IMAGE_BASE_URL} />

              <OtherImages otherImages={product?.other_images} baseUrl={IMAGE_BASE_URL} />
            </>
          )}
        </Box>
      </Fade>
    </Modal>
  );
}
