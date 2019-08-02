# image-classification-explorer
Started with code from Webcam Pacman from TensorFlow.js examples.

# Instructions
Using yarn:
```sh
cd image-classification-explorer
yarn
```

When make changes, pull from this branch in the vm, then yarn build.

The http-server will automatically update after yarn build.


INSTRUCTIONS BELOW NOW COVERED BY JUST RUNNING ./run-me.sh (in ~)

If the http-server process dies, cd into image-classification-explorer and yarn build. 
cd ~/, you should see cert.pem and key.pem
cd into image-classification-explorer/dist
Start process 1 to serve index.html on the ssh port for the app on 443
http-server index.html -S -K '~/key.pem' -C '~/cert.pem' -p 443 (https://www.npmjs.com/package/http-server)

Start process 2 to serve squeezenet on port 8080 (https://classifier.appinventor.mit.edu:8080/model.json serves squeezenet. MobileNet fetched online.)
http-server squeezenet/model.json -S -K '~/key.pem' -C '~/cert.pem' -p 8080

I think need to add --cors 

Run with a & at the end so not blocked by process

Set flags for the https cert and key and ports. -S will enable https.
