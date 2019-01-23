/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
const JSZip = require('jszip');
const FileSaver = require('file-saver');

import {Dataset} from './dataset';
import {Results} from './results';
import * as ui from './ui';
import * as modal from './modal';
import * as saliency from './saliency';
import {Webcam} from './webcam';

// Later, maybe allow users to pick these values themselves?
const LEARNING_RATE = 0.0001;
const BATCH_SIZE_FRACTION = 0.4;
const EPOCHS = 20;
const DENSE_UNITS = 100;

const SALIENCY_NUM_SAMPLES = 15;
const SALIENCY_NOISE_STD = 0.1;
const SALIENCY_CLIP_PERCENT = 0.99;

const fetch = require('node-fetch');

// Variables for containing the model datasets, prediction results,
// the models themselves, and the webcam
const trainingDataset = new Dataset();
const testingDataset = new Dataset();

var trainingImgDict = {};
var testingImgDict = {}; 

let trainingResults;
let testingResults;

let transferModel;
let model;
let entireModel;

const webcam = new Webcam(document.getElementById('webcam'));

// model state stuff
// const modelNames = ["MobileNet", "SqueezeNet"];
const modelLastLayers = {"MobileNet" : "conv_pw_13_relu", "SqueezeNet" : "max_pooling2d_1"};
let currentModel = "MobileNet";

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadMobilenet() {
  const transferModel = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  
  const layer = transferModel.getLayer('conv_pw_13_relu');
  return tf.model({inputs: transferModel.inputs, outputs: layer.output});
}

// Loads squeezenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadSqueezenet() {
  const squeezenet = await tf.loadModel('http://127.0.0.1:8080/model.json'); // Go to squeezenet folder & run http-server . --cors -o
  return squeezenet;
}

// Methods for updating the dataset objects from the ui
ui.setAddExampleHandler((labelId, datasetName) => {
  tf.tidy(async () => {
    const img = webcam.capture();

    if (datasetName == "training") {
      if (labelId in trainingImgDict) {
        trainingImgDict[labelId].push(tf.keep(img)); 
      } else {
        trainingImgDict[labelId] = [tf.keep(img)]; 
      }
    } else {
      if (labelId in testingImgDict) {
        testingImgDict[labelId].push(tf.keep(img));
      } else {
        testingImgDict[labelId] = [tf.keep(img)];
      }
    }

    ui.drawThumb(img, datasetName, labelId);
  });
});

ui.setAddLabelHandler(labelName => {
  testingDataset.addLabel(labelName);
  return trainingDataset.addLabel(labelName);
});
ui.setRemoveLabelHandler(labelId => {
  delete trainingImgDict[labelId]; 
  delete testingImgDict[labelId]; 
  testingDataset.removeLabel(labelId);
  trainingDataset.removeLabel(labelId);
});

// Methods to supply data to the results modal
modal.setGetResultsHandler(() => {
  if (ui.getCurrentTab() == "training") {
    return trainingResults;
  } else {
    return testingResults;
  }
});

modal.setGetSaliencyHandler(async function(img) {
  return await saliency.smoothGrad(img, SALIENCY_NUM_SAMPLES, SALIENCY_NOISE_STD, entireModel, SALIENCY_CLIP_PERCENT);
});

// Sets up and trains the classifier
async function train() {
  if (Object.values(trainingImgDict) == []) {
    throw new Error('Add some examples before training!');
  }

  // TODO: check model input, load appropriate model, and create trainingData and testingData
  // look at input from dropdown menu 
  var currentModel = document.getElementById("choose-model-dropdown").value;
  console.log(currentModel);

  if (currentModel == "MobileNet") {
    transferModel = await loadMobilenet();
    console.log("Loaded MobileNet!");
  } else if (currentModel == "SqueezeNet") {
    transferModel = await loadSqueezenet(); 
    console.log("Loaded SqueezeNet!");
  } else {
    console.log("Model loading failed!");
  }

  // TODO: create clear fxn that also gets rid of old tensors 
  trainingDataset.removeExamples();

  // loop over trainingImgDict and testingImgDict and process 
  for (let label in trainingImgDict) {
    for (let img in trainingImgDict[label]) {
      const img_copy = tf.clone(trainingImgDict[label][img]); 
      trainingDataset.addExample(trainingImgDict[label][img], transferModel.predict(trainingImgDict[label][img]), label); 
      trainingImgDict[label][img] = tf.keep(img_copy); 
    }
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  // look at input from dropdown menu 
  var currentModel2 = document.getElementById("choose-model-dropdown2").value;
  console.log(currentModel2);

  if (currentModel2 == "Model 1") {
    model = tf.sequential({
      layers: [
        // Flattens the input to a vector so we can use it in a dense layer. While
        // technically a layer, this only performs a reshape (and has no training
        // parameters).
        tf.layers.flatten({inputShape: [7, 7, 256]}),
        // Layer 1
        tf.layers.dense({
          units: DENSE_UNITS,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
          useBias: true
        }),
        // Layer 2. The number of units of the last layer should correspond
        // to the number of classes we want to predict.
        tf.layers.dense({
          units: trainingDataset.numLabels,
          kernelInitializer: 'varianceScaling',
          useBias: false,
          activation: 'softmax'
        })
      ]
    });
  } else {
    model = tf.sequential({
      layers: [
        tf.layers.conv2d({
          inputShape: [7, 7, 256],
          kernelSize: 5,
          filters: 32, 
          strides: 1, 
          activation: 'relu',
          kernelInitializer: 'varianceScaling'
        }),
        tf.layers.flatten(),
        tf.layers.dense({
          units: trainingDataset.numLabels, 
          kernelInitializer: 'varianceScaling', 
          useBias: false, 
          activation: 'softmax'
        })
      ]
    }); 

    
  }

  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  const optimizer = tf.train.adam(LEARNING_RATE);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // Get data from the training dataset
  const trainingData = await tf.tidy(() => {
    return trainingDataset.getData();
  });

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(trainingData.xs.shape[0] * BATCH_SIZE_FRACTION);
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  await model.fit(trainingData.xs, trainingData.ys, {
    batchSize,
    epochs: EPOCHS,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
      },

      onTrainEnd: () => {
        // Piece together the entire model

        // TODO: add logic to determine what model we're using
        // For mobilenet
        let output = transferModel.getLayer(modelLastLayers[currentModel]).output; 

        for (let i = 0; i < model.layers.length; i++) {
          const currentLayer = model.getLayer("filler", i);
          output = currentLayer.apply(output);
        }

        entireModel = tf.model({inputs: transferModel.inputs, outputs: output});
      }
    }
  });
}

// Uses the classifier to classify examples
async function predict(dataset, modelLabelsJson) {

  // Gets the data from the dataset and predicts on it
  const datasetData = await tf.tidy(() => {
    return dataset.getData();
  });

  const predictedClass = tf.tidy(() => {
    return model.predict(datasetData.xs);
  });

  // Calculates the top k predictions for each image
  const labelNamesMap = JSON.parse(modelLabelsJson);

  const numPredictions = Math.min(3, Object.keys(labelNamesMap).length);
  const topPredictions = await predictedClass.topk(numPredictions);

  const predictedIndices = await topPredictions.indices.data();
  const predictedValues = await topPredictions.values.data();

  const actualIndices = await datasetData.ys.argMax(1).data();

  predictedClass.dispose();

  // Creates a results object to store all of the results
  return new Results(datasetData.imgs, actualIndices, predictedIndices, predictedValues, labelNamesMap);
}

// Train and predict button functionality. Also updates the results' prev/next buttons.
let resultsPrevButtonFunctionTraining = null;
let resultsNextButtonFunctionTraining = null;

document.getElementById('train').addEventListener('click', async () => {
  // First, we train the model on the training dataset
  ui.trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();

  /**
   * Updates HTML timer 
   */
  function clockRunning(){
    var currentTime = new Date()
        , timeElapsed = new Date(currentTime - timeBegan - stoppedDuration)
        , hour = timeElapsed.getUTCHours()
        , min = timeElapsed.getUTCMinutes()
        , sec = timeElapsed.getUTCSeconds()
        , ms = timeElapsed.getUTCMilliseconds();

    document.getElementById("display-area").innerHTML = 
        (hour > 9 ? hour : "0" + hour) + ":" + 
        (min > 9 ? min : "0" + min) + ":" + 
        (sec > 9 ? sec : "0" + sec) + "." + 
        (ms > 99 ? ms : ms > 9 ? "0" + ms : "00" + ms);
  };

  // measuring training time as sanity check.. 
  let startTime = new Date().getTime();

  // reset & start 
  let stoppedDuration = 0
  document.getElementById("display-area").innerHTML = "00:00:00.000";
  let timeBegan = new Date();
  let started = setInterval(clockRunning, 10); 

  await train();

  // stop 
  clearInterval(started);

  let endTime = new Date().getTime();
  console.log("The training took: " + (endTime - startTime) + "ms.");
  console.log("The training took: " + (endTime - startTime)/1000 + "s.");

  // Then, we use the model we trained to make predictions on the training dataset
  trainingResults = await predict(trainingDataset, trainingDataset.getCurrentLabelNamesJson());

  // Then, we update the results column of the interface with the results
  ui.updateResult(trainingResults.getNextResult(), "training");

  const resultsPrevButton = document.getElementById("results-image-button-prev-training");
  const resultsNextButton = document.getElementById("results-image-button-next-training");

  if (resultsPrevButtonFunctionTraining != null) {
    resultsPrevButton.removeEventListener('click', resultsPrevButtonFunctionTraining);
    resultsNextButton.removeEventListener('click', resultsNextButtonFunctionTraining);
  }

  // We store the methods to step through results so we can remove them from the buttons if
  // we get new results
  resultsPrevButtonFunctionTraining = () => {
    ui.updateResult(trainingResults.getPreviousResult(), "training");
  }

  resultsNextButtonFunctionTraining = () => {
    ui.updateResult(trainingResults.getNextResult(), "training");
  }

  resultsPrevButton.addEventListener('click', resultsPrevButtonFunctionTraining);
  resultsNextButton.addEventListener('click', resultsNextButtonFunctionTraining);
});

let resultsPrevButtonFunctionTesting = null;
let resultsNextButtonFunctionTesting = null;

document.getElementById('predict').addEventListener('click', async () => {

  // TODO: create clear fxn that also gets rid of old tensors 
  testingDataset.removeExamples();

  console.log("raw testing  images");
  console.log(testingImgDict); 

  // loop over testingImgDict and testingImgDict and process 
  for (let label in testingImgDict) {
    for (let img in testingImgDict[label]) {
      const img_copy = tf.clone(testingImgDict[label][img]); 
      testingDataset.addExample(testingImgDict[label][img], transferModel.predict(testingImgDict[label][img]), label); 
      testingImgDict[label][img] = tf.keep(img_copy); 
    }
  }

  testingResults = await predict(testingDataset, trainingDataset.getCurrentLabelNamesJson());

  // Then, we update the results column of the interface with the results
  ui.updateResult(testingResults.getNextResult(), "testing");

  const resultsPrevButton = document.getElementById("results-image-button-prev-testing");
  const resultsNextButton = document.getElementById("results-image-button-next-testing");

  if (resultsPrevButtonFunctionTesting != null) {
    resultsPrevButton.removeEventListener('click', resultsPrevButtonFunctionTesting);
    resultsNextButton.removeEventListener('click', resultsNextButtonFunctionTesting);
  }

  // We store the methods to step through results so we can remove them from the buttons if
  // we get new results
  resultsPrevButtonFunctionTesting = () => {
    ui.updateResult(testingResults.getPreviousResult(), "testing");
  }

  resultsNextButtonFunctionTesting = () => {
    ui.updateResult(testingResults.getNextResult(), "testing");
  }

  resultsPrevButton.addEventListener('click', resultsPrevButtonFunctionTesting);
  resultsNextButton.addEventListener('click', resultsNextButtonFunctionTesting);
});

// Download button functionality
document.getElementById('download-button').addEventListener('click', async () => {
  // The TensorFlow.js save method doesn't work properly in Firefox, so we write
  // our own. This methods zips up the model's topology file, weights files, and
  // a json of the mapping of model predictions to label names. The resulting file
  // is given the .mdl extension to prevent tampering with.
  const zipSaver = {save: function(modelSpecs) {
    const modelTopologyFileName = "model.json";
    const weightDataFileName = "model.weights.bin";
    const modelLabelsName = "model_labels.json";
    const modelZipName = "model.mdl";

    const weightsBlob = new Blob(
      [modelSpecs.weightData], {type: 'application/octet-stream'});

    const weightsManifest = [{
      paths: ['./' + weightDataFileName],
      weights: modelSpecs.weightSpecs
    }];
    const modelTopologyAndWeightManifest = {
      modelTopology: modelSpecs.modelTopology,
      weightsManifest
    };
    const modelTopologyAndWeightManifestBlob = new Blob(
      [JSON.stringify(modelTopologyAndWeightManifest)],
      {type: 'application/json'});

    const zip = new JSZip();
    zip.file(modelTopologyFileName, modelTopologyAndWeightManifestBlob);
    zip.file(weightDataFileName, weightsBlob);
    zip.file(modelLabelsName, trainingDataset.getCurrentLabelNamesJson());

    zip.generateAsync({type:"blob"})
      .then(function (blob) {
          FileSaver.saveAs(blob, modelZipName);
      });
  }};

  const savedModel = await model.save(zipSaver);
});

// Helper method to convert a blob to an actual file, which TensorFlow.js requires
// in order to load in the model
function blobToFile(blob, fileName) {
  // A Blob() is almost a File() - it's just missing the two properties below which we will add
  blob.lastModifiedDate = new Date();
  blob.name = fileName;
  return blob;
}

// Upload button functionality
const modelUpload = document.getElementById('model-upload');

document.getElementById('upload-button').addEventListener('click', async () => {
  modelUpload.click();
});

modelUpload.addEventListener('change', async () => {
  const modelZipFile = modelUpload.files[0];

  const modelJsonName = "model.json";
  const modelWeightsName = "model.weights.bin";
  const modelLabelsName = "model_labels.json";

  const modelFiles = await JSZip.loadAsync(modelZipFile);
  const modelJsonBlob = await modelFiles.file(modelJsonName).async("blob");
  const modelWeightsBlob = await modelFiles.file(modelWeightsName).async("blob");
  const modelLabelsText = await modelFiles.file(modelLabelsName).async("text");

  const modelJsonFile = blobToFile(modelJsonBlob, modelJsonName);
  const modelWeightsFile = blobToFile(modelWeightsBlob, modelWeightsName);

  model = await tf.loadModel(
    tf.io.browserFiles([modelJsonFile, modelWeightsFile]));
  trainingDataset.setCurrentLabelNames(modelLabelsText);

  const modelLabelsJson = JSON.parse(modelLabelsText);

  // After uploading the model, we update the ui to reflect the labels in the model
  ui.removeLabels();
  for (let labelNumber in modelLabelsJson) {
    if (modelLabelsJson.hasOwnProperty(labelNumber)) {
        ui.addLabel(modelLabelsJson[labelNumber]);
    }
  }

  modelUpload.value = "";
});

// Initialize the application

async function init() {
  try {
    await webcam.setup();
  } catch (e) {
    document.getElementById('no-webcam').style.display = 'block';
  }

  console.log(await tf.io.listModels()); 

  ui.init();
  modal.init();
}

init();