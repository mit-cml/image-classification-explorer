import matplotlib
matplotlib.use('Agg')
import numpy as np

from flask import Flask, request, Response, json, send_file
import matplotlib.pyplot as plt
from scipy.io import wavfile
import requests
from flask_cors import CORS

from pydub import AudioSegment
from pydub.silence import split_on_silence

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


	f = open('./temp/audio.wav', 'wb')
	f.write(request.data)
	f.close()

	rate, data = wavfile.read('./temp/audio.wav')

	if len(data.shape) > 1:
		data = data[:,0]

	print("Rate: " + str(rate))
	
	silence_threshold = -50

	sound = AudioSegment.from_file("./temp/audio.wav", format="wav")
	sound = sound.set_channels(1)

	start_trim = detect_leading_silence(sound, silence_threshold)
	end_trim = detect_leading_silence(sound.reverse(), silence_threshold)
	duration = len(sound)    
	sound = sound[start_trim:duration-end_trim]

	chunks = split_on_silence(sound, min_silence_len=10, silence_thresh=silence_threshold, keep_silence=0)
	print("Num Chunks: " + str(len(chunks)))
	# for i,c in enumerate(chunks):
	# 	print("Chunk: " + i, flush=True)
	# 	print(c._data, flush=True)
	
	if len(chunks) == 1:
		sound = chunks[0]
	if len(chunks) > 1:
		sound = sum(chunks[1:], chunks[0])
	
	data = np.array(sound.get_array_of_samples())



	fig,ax = plt.subplots(1)
	fig.subplots_adjust(left=-0,right=1,bottom=0,top=1)


	fig.set_size_inches(10, 10)
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



def detect_leading_silence(sound, silence_threshold, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

# def remove_silence(sound, silence_threshold=-50.0, chunk_size=10):
# 	silence_start = 0
# 	while silence_start < len(sound) and sound[silence_start:silence_start+chunk_size].dBFS > silence_threshold:
# 		silence_start += chunk_size
# 	#now silence_start = ms where silence starts
# 	silence_end = silence_start + chunk_size
# 	while silence_end < len(sound) and sound[silence_end:silence_end+chunk_size].dBFS < silence_threshold:
# 		silence_end += chunk_size
	
# 	if silence_start > len(sound)
# 		return sound
# 	return remove_silence(sound[])
# 	#now silence_end = ms where silence ends
	


if __name__ == "__main__":
    application.run()
