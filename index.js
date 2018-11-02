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

import {ControllerDataset} from './controller_dataset';
import * as ui from './ui';
import {Webcam} from './webcam';

// We pick these values for the users of our interface.
const LEARNING_RATE = 0.0001;
const BATCH_SIZE_FRACTION = 0.4;
const EPOCHS = 20;
const DENSE_UNITS = 100;

// A webcam class that generates Tensors from the images from the webcam.
const webcam = new Webcam(document.getElementById('webcam'));

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset();

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
ui.setAddExampleHandler(labelId => {
  tf.tidy(() => {
    const img = webcam.capture();
    controllerDataset.addExample(img, mobilenet.predict(img), labelId);

    // Draw the preview thumbnail.
    ui.drawThumb(img, labelId);
  });
});

// Functions to update the state of the controller dataset
ui.setAddLabelHandler(labelName => {
  return controllerDataset.addLabel(labelName);
});
ui.setRemoveLabelHandler(labelId => {
  controllerDataset.removeLabel(labelId);
});

/**
 * Sets up and trains the classifier.
 */
async function train() {
  if (controllerDataset.labelXs == {}) {
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
        units: controllerDataset.numLabels,
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
  const xsAndYs = await tf.tidy(() => {
    return controllerDataset.getXsAndYs();
  });

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(xsAndYs.xs.shape[0] * BATCH_SIZE_FRACTION);
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(xsAndYs.xs, xsAndYs.ys, {
    batchSize,
    epochs: EPOCHS,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
      }
    }
  });
}

async function predict() {
  const predictedClass = tf.tidy(() => {
    // Capture the frame from the webcam.
    const img = webcam.capture();

    // Make a prediction through mobilenet, getting the internal activation of
    // the mobilenet model.
    const activation = mobilenet.predict(img);

    // Make a prediction through our newly-trained model using the activation
    // from mobilenet as input.
    const predictions = model.predict(activation);

    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    return predictions.as1D();
  });

  const numPredictions = Math.min(3, controllerDataset.numLabels);
  const topPredictions = await predictedClass.topk(numPredictions);

  const predictionIndices = await topPredictions.indices.data();
  const predictionValues = await topPredictions.values.data();

  let predictionText = "Result:";

  for (let i = 0; i < numPredictions; i++) {
    const currentIndex = predictionIndices[i];
    const currentValue = predictionValues[i];

    const labelName = controllerDataset.getLabelNameFromModelPrediction(currentIndex);

    predictionText += "\n" + labelName + ": " + currentValue.toFixed(5);
  }

  ui.predictResult(predictionText);
  predictedClass.dispose();
  await tf.nextFrame();
}

document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  train();
});
document.getElementById('predict').addEventListener('click', () => {
  predict();
});

document.getElementById('download-button').addEventListener('click', async () => {
  // console.log("downloading");
  const savedModel = await model.save('downloads://my-model');
  // console.log(savedModel);
});

document.getElementById('upload-button').addEventListener('click', async () => {
  const jsonUpload = document.getElementById('json-upload');
  const weightsUpload = document.getElementById('weights-upload');

  model = await tf.loadModel(
    tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));

  // console.log(model);
});

async function init() {
  try {
    await webcam.setup();
  } catch (e) {
    document.getElementById('no-webcam').style.display = 'block';
  }
  mobilenet = await loadMobilenet();

  // model = tf.sequential(
  //    {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
  // const saveResults = await model.save('downloads://my-model-1');

  // console.log(saveResults);

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // quick.
  tf.tidy(() => mobilenet.predict(webcam.capture()));

  ui.init();
}

// Initialize the application.
init();
