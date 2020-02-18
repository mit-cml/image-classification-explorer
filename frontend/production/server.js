const express = require('express');
const path = require('path');
var request = require('request');
var http = require('http');
var https = require('https')
var fs = require('fs')
const app = express();

// Serve the static files from the React app
app.use(express.static(path.join(__dirname, 'build')));



app.post('/spectrogram', (req,res) => {
	var today = new Date();
	var time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
	console.log(time)
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

var privateKey  = fs.readFileSync('/etc/letsencrypt/live/c1.appinventor.mit.edu/privkey.pem', 'utf8');
var certificate = fs.readFileSync('/etc/letsencrypt/live/c1.appinventor.mit.edu/fullchain.pem', 'utf8');
var credentials = {key: privateKey, cert: certificate};

var httpServer = http.createServer(app);
var httpsServer = https.createServer(credentials, app);

httpsServer.listen(443)
//httpServer.listen(4000)

console.log('Listening on port ' + port);
