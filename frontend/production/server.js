const express = require('express');
const path = require('path');
var request = require('request');

const app = express();

// Serve the static files from the React app
app.use(express.static(path.join(__dirname, 'build')));

app.post('/spectrogram', (req,res) => {
	console.log("Proxying /spectrogram to http://localhost:5000" + req.path)
	res.redirect(307, 'http://localhost:5000' + req.path);
});

// Handles any requests that don't match the ones above
app.get('*', (req,res) =>{
	console.log("Loading web app");
    res.sendFile(path.join(__dirname+'/build/index.html'));
});

// const port = process.env.PORT || 5000;
const port = 4000;

app.listen(port);

console.log('Listening on port ' + port);