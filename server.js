const express = require('express');
const path = require('path');

const app = express();

// Middleware
app.use(express.json());

// Serve static files
const publicDir = path.join(__dirname, 'public');
app.use(express.static(publicDir));

// Basic API route
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', uptime: process.uptime() });
});

// Fallback to index.html for root
app.get('/', (req, res) => {
  res.sendFile(path.join(publicDir, 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

module.exports = app;

