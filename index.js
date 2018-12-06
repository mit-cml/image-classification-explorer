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
import {Webcam} from './webcam';

// We pick these values for the users of our interface.
const LEARNING_RATE = 0.0001;
const BATCH_SIZE_FRACTION = 0.4;
const EPOCHS = 20;
const DENSE_UNITS = 100;

// A webcam class that generates Tensors from the images from the webcam.
const webcam = new Webcam(document.getElementById('webcam'));

// Dataset objects for the training and test sets
const trainingDataset = new Dataset();
const testingDataset = new Dataset();

let trainingResults;
let testingResults;

let mobilenet;
let model;

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadMobilenet() {
  const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// When a label's add example button is clicked, read a frame from the 
// webcam and associate it with the corresponding label
ui.setAddExampleHandler((labelId, datasetName) => {
  tf.tidy(async () => {
    const img = webcam.capture();

    if (datasetName === "training") {
      trainingDataset.addExample(img, mobilenet.predict(img), labelId);
    } else {
      testingDataset.addExample(img, mobilenet.predict(img), labelId);
    }

    // Draw the preview thumbnail.
    ui.drawThumb(img, datasetName, labelId);
  });
});

// Functions to update the state of the datasets
ui.setAddLabelHandler(labelName => {
  testingDataset.addLabel(labelName);
  return trainingDataset.addLabel(labelName);
});
ui.setRemoveLabelHandler(labelId => {
  testingDataset.removeLabel(labelId);
  trainingDataset.removeLabel(labelId);
});

// Functions to supply results to the results modal
modal.setGetResultsHandler(() => {
  if (ui.getCurrentTab() == "training") {
    return trainingResults;
  } else {
    return testingResults;
  }
});

/**
 * Sets up and trains the classifier.
 */
async function train() {
  if (trainingDataset.labelXs == {}) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
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

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(LEARNING_RATE);
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // Get xs and ys
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
      }
    }
  });
}

async function predict(dataset, modelLabelsJson) {
  const datasetData = await tf.tidy(() => {
    return dataset.getData();
  });

  const predictedClass = tf.tidy(() => {
    // Make predictions from the mobilenet activations of the dataset
    return model.predict(datasetData.xs);
  });

  const labelNamesMap = JSON.parse(modelLabelsJson);

  const numPredictions = Math.min(3, Object.keys(labelNamesMap).length);
  const topPredictions = await predictedClass.topk(numPredictions);

  const predictedIndices = await topPredictions.indices.data();
  const predictedValues = await topPredictions.values.data();

  const actualIndices = await datasetData.ys.argMax(1).data();

  predictedClass.dispose();

  return new Results(datasetData.imgs, actualIndices, predictedIndices, predictedValues, labelNamesMap);
}

let resultsPrevButtonFunctionTraining = null;
let resultsNextButtonFunctionTraining = null;

document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  await train();

  trainingResults = await predict(trainingDataset, trainingDataset.getCurrentLabelNamesJson());
  ui.updateResult(trainingResults.getNextResult(), "training");

  const resultsPrevButton = document.getElementById("results-image-button-prev-training");
  const resultsNextButton = document.getElementById("results-image-button-next-training");

  if (resultsPrevButtonFunctionTraining != null) {
    resultsPrevButton.removeEventListener('click', resultsPrevButtonFunctionTraining);
    resultsNextButton.removeEventListener('click', resultsNextButtonFunctionTraining);
  }

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
  testingResults = await predict(testingDataset, trainingDataset.getCurrentLabelNamesJson());
  ui.updateResult(testingResults.getNextResult(), "testing");

  const resultsPrevButton = document.getElementById("results-image-button-prev-testing");
  const resultsNextButton = document.getElementById("results-image-button-next-testing");

  if (resultsPrevButtonFunctionTesting != null) {
    resultsPrevButton.removeEventListener('click', resultsPrevButtonFunctionTesting);
    resultsNextButton.removeEventListener('click', resultsNextButtonFunctionTesting);
  }

  resultsPrevButtonFunctionTesting = () => {
    ui.updateResult(testingResults.getPreviousResult(), "testing");
  }

  resultsNextButtonFunctionTesting = () => {
    ui.updateResult(testingResults.getNextResult(), "testing");
  }

  resultsPrevButton.addEventListener('click', resultsPrevButtonFunctionTesting);
  resultsNextButton.addEventListener('click', resultsNextButtonFunctionTesting);
});

document.getElementById('download-button').addEventListener('click', async () => {
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

// From https://stackoverflow.com/questions/27159179/how-to-convert-blob-to-file-in-javascript
function blobToFile(blob, fileName) {
  // A Blob() is almost a File() - it's just missing the two properties below which we will add
  blob.lastModifiedDate = new Date();
  blob.name = fileName;
  return blob;
}

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

  ui.removeLabels();
  for (let labelNumber in modelLabelsJson) {
    if (modelLabelsJson.hasOwnProperty(labelNumber)) {
        ui.addLabel(modelLabelsJson[labelNumber]);
    }
  }

  modelUpload.value = "";
});

async function init() {
  try {
    await webcam.setup();
  } catch (e) {
    document.getElementById('no-webcam').style.display = 'block';
  }
  mobilenet = await loadMobilenet();

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // quick.
  tf.tidy(() => mobilenet.predict(webcam.capture()));

  ui.init();
  modal.init();
}

// Initialize the application.
init();
