

import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css'; // Global styling imports
import App from './App';
import './App.css';

// Get the root element from the HTML
const container = document.getElementById('root');
const root = createRoot(container);

// Render the application
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);