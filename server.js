// TODO convert to ES Modules.
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const { PythonShell } = require('python-shell');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 5000;

// Enable CORS
app.use(cors());

// Multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Handle file uploads
app.post('/upload', upload.single('file'), (req, res) => {
	if (!req.file) {
		return res.status(400).json({ error: 'No file uploaded.' });
	}
	let numMatches = 1;
	if (req.body.numMatches > 0) {
		numMatches = req.body.numMatches;
	}
	// Save the file to the 'public/uploads/' directory
	const imagePath = path.join(__dirname, 'public', 'uploads', 'pet.jpg');
	fs.writeFileSync(imagePath, req.file.buffer);

	// Run Python script for feature extraction using PyTorch
	const modelPath = path.join(__dirname, 'resnet18.pth');
	const options = {
		pythonPath: 'python',
		args: [imagePath, modelPath, numMatches],
	};


	PythonShell.run('feature_extraction.py', options)
		.then((result) => {
			try {
				const features = JSON.parse(result[0]);
				res.status(200).json({ result: features });
			} catch (error) {
				console.error("Error parsing JSON:", error);
				res.status(500).json({ error: 'Internal Server Error' });
			}
		})
		.catch((err) => {
			console.error(err);
			res.status(500).json({ error: 'Internal Server Error' });
		});

});

app.listen(PORT, () => {
	console.log(`Server is running on http://localhost:${PORT}`);
});
