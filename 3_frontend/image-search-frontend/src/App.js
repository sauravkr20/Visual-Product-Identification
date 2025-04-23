import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './component/navbar';
import Search from './component/search';  // your existing search component
import AddProduct from './component/add_product'; // the new add product component
import { Container } from '@mui/material';

function App() {
  return (
    <Router>
      <Navbar />
      <Container sx={{ mt: 4 }}>
        <Routes>
          <Route path="/" element={<Navigate to="/search" />} />
          <Route path="/search" element={<Search />} />
          <Route path="/add" element={<AddProduct />} />
        </Routes>
      </Container>
    </Router>
  );
}

export default App;
