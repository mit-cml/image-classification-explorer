from flask import Flask, request, Response, json, send_file
import matplotlib.pyplot as plt
from scipy.io import wavfile
import requests
from flask_cors import CORS


application = Flask(__name__)
CORS(application)

@application.route("/")
def index():
    return "Index!"

@application.route("/hello")
def hello():
    return "Hello World!"

@application.route("/spectrogram", methods=['POST'])
def spectrogram():
	print(request.data)

	f = open('./temp/audio.wav', 'wb')
	f.write(request.data)
	f.close()

	rate, data = wavfile.read('./temp/audio.wav')
	data = data[:,0]
	fig,ax = plt.subplots(1)
	fig.subplots_adjust(left=-.031,right=1,bottom=0,top=1)
	ax.axis('off')
	pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=384, NFFT=512)
	ax.axis('off')
	fig.savefig('./temp/spectrogram.png', dpi=100)

	# return Response(json.dumps("Here's your spectrogram!"), status=200)
	return send_file('./temp/spectrogram.png', mimetype='image/png')

@application.route("/mobilenet", methods=['GET'])
def mobilenet():
	# return send_file('./models/mobilenet.json', mimetype='applicationlication/json')
	return Response(requests.get('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json').text, mimetype="applicationlication/json")

@application.route("/squeezenet", methods=['GET'])
def squeezenet():
	return send_file('./models/squeezenet.json', mimetype='applicationlication/json')

if __name__ == "__main__":
    application.run()
