import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { NavLink } from 'react-router-dom';

const Navbar = () => (
  <AppBar position="static">
    <Toolbar >
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Button color="inherit" component={NavLink} to="/search" sx={{ mx: 1 }}>
          Search Image
        </Button>
        <Button color="inherit" component={NavLink} to="/add" sx={{ mx: 1 }}>
          Add Product
        </Button>
      </Box>
    </Toolbar>
  </AppBar>
);

export default Navbar;
