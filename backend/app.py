from flask import Flask, request, Response, json, send_file
import matplotlib.pyplot as plt
from scipy.io import wavfile


app = Flask(__name__)

@app.route("/")
def index():
    return "Index!"

@app.route("/hello")
def hello():
    return "Hello World!"

@app.route("/spectrogram", methods=['POST'])
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


if __name__ == "__main__":
    app.run()
