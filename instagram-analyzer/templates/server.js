const express = require('express');
const fileUpload = require('express-fileupload');
const { v4: uuidv4 } = require('uuid');
const cors = require('cors'); // Add this

const app = express();

// Add middleware
app.use(cors()); // Enable CORS
app.use(fileUpload());
app.use('/reports', express.static('reports'));

// Enhanced upload endpoint
app.post('/api/upload-report', (req, res) => {
  try {
    if (!req.files?.pdf) {
      return res.status(400).json({ error: 'No PDF uploaded' });
    }

    const pdf = req.files.pdf;
    const reportId = uuidv4();
    const fileName = `${reportId}.pdf`;

    pdf.mv(`reports/${fileName}`, (err) => {
      if (err) {
        console.error('File save error:', err);
        return res.status(500).json({ error: 'File save failed' });
      }
      
      res.json({ 
        url: `http://${req.get('host')}/reports/${fileName}`
      });
    });

  } catch (error) {
    console.error('Server error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.listen(5000, () => {
  console.log('Server running on port 5000');
});