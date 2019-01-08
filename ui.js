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

export function init() {
  document.getElementById('explorer').style.display = '';
  document.getElementById('status').style.display = 'none';

  trainingTabButton.click();
}

// Trackers for the current state of the explorer (training or testing)
const tabNames = ["training", "testing"];
let currentTab = tabNames[0];

// Elements dealing with adding labels to the model
const addLabelsInput = document.getElementById('label-name');
const addLabelsButton = document.getElementById('add-labels-button');

// Element for displaying training loss
const trainStatusElement = document.getElementById('train-status');

export function trainStatus(status) {
  trainStatusElement.innerText = status;
}

// These are set in index.js
export let addExampleHandler;
export function setAddExampleHandler(handler) {
  addExampleHandler = handler;
}

export let addLabelHandler;
export function setAddLabelHandler(handler) {
  addLabelHandler = handler;
}

export let removeLabelHandler;
export function setRemoveLabelHandler(handler) {
  removeLabelHandler = handler;
}

// Handlers for adding/removing labels
export function addLabel(newLabelName) {
  const newLabelId = addLabelHandler(newLabelName);

  const removeLabelButtonText = "Remove this label";
  const addLabelExampleButtonText = "Add example";

  // We use this array so that we can remove labelBoxes from
  // both the training and testing tabs when clicking the
  // remove buttons in the training tab
  const labelBoxTraining = document.createElement("div");
  const labelBoxTesting = document.createElement("div");
  const labelBoxes = [labelBoxTraining, labelBoxTesting];

  for (let i = 0; i < tabNames.length; i++) {
    const datasetName = tabNames[i];
    const labelsOuter = document.getElementById("labels-container-" + datasetName);

    // Overarching div for a label
    const labelBox = labelBoxes[i];
    labelBox.setAttribute("class", "label-container");
    labelBox.setAttribute("id", datasetName + "-" + newLabelId);

    // Div for counting number of examples for this label
    const labelCount = document.createElement("div");
    labelCount.setAttribute("class", "label-count-outer");
    const labelCountText = document.createElement("span");
    labelCountText.textContent = "Num examples: ";
    const labelCountSpan = document.createElement("span");
    labelCountSpan.textContent = 0;
    labelCount.appendChild(labelCountText);
    labelCount.appendChild(labelCountSpan);

    // Div for placing the label name
    const labelName = document.createElement("div");
    labelName.setAttribute("class", "label-name-outer");
    const labelNameSpan = document.createElement("span");
    labelNameSpan.textContent = "Label: " + newLabelName;
    labelName.appendChild(labelNameSpan);

    // Div where we will put a thumbnail/previews of images users have added
    const labelImagesOuter = document.createElement("div");
    labelImagesOuter.setAttribute("class", "label-images-outer");
    const labelImagesInner = document.createElement("div");
    labelImagesInner.setAttribute("class", "label-images-inner");
    const labelImagesCanvas = document.createElement("canvas");
    labelImagesCanvas.setAttribute("class", "label-images-canvas");
    labelImagesCanvas.setAttribute("width", 224);
    labelImagesCanvas.setAttribute("height", 224);
    labelImagesCanvas.setAttribute("id", "label-images-canvas-" + datasetName + "-" + newLabelId);
    labelImagesInner.appendChild(labelImagesCanvas);
    labelImagesOuter.appendChild(labelImagesInner);

    // Button for adding an example to this label
    const labelAddExample = document.createElement("button");
    labelAddExample.setAttribute("class", "label-add-example");
    const labelAddExampleSpan = document.createElement("span");
    labelAddExampleSpan.textContent = addLabelExampleButtonText;
    labelAddExample.appendChild(labelAddExampleSpan);

    labelAddExample.addEventListener("click", function() {
      labelCountSpan.textContent = parseInt(labelCountSpan.textContent) + 1;
      addExampleHandler(newLabelId, datasetName);
    });

    // Add all elements we just created to the label box and append to
    // the container on the page. Also creates a remove label button if
    // this is for the training tab
    labelBox.appendChild(labelName);
    labelBox.appendChild(labelCount);
    labelBox.appendChild(labelImagesOuter);

    if (datasetName === "training") {
      const labelRemove = document.createElement("button");
      labelRemove.setAttribute("class", "label-remove");
      const labelRemoveSpan = document.createElement("span");
      labelRemoveSpan.textContent = removeLabelButtonText;
      labelRemove.appendChild(labelRemoveSpan);

      labelRemove.addEventListener("click", () => {
        removeLabelHandler(newLabelId);
        labelBox.parentNode.removeChild(labelBox);
        labelBoxes[i + 1].parentNode.removeChild(labelBoxes[i + 1]);
      });

      labelBox.appendChild(labelRemove);
    }

    labelBox.appendChild(labelAddExample);

    labelsOuter.appendChild(labelBox);
  }
}

addLabelsButton.addEventListener('click', () => {
  const newLabelName = addLabelsInput.value;
  addLabel(newLabelName);
  addLabelsInput.value = "";
});

export function removeLabels() {
  var removeButtons = document.getElementsByClassName('label-remove');
  while (removeButtons.length) {
    removeButtons[0].click();
  }
}

// Handler for updating the results column
export function updateResult(result, datasetName) {
  document.getElementById('results-container-' + datasetName).style.display = "";

  const resultCanvas = document.getElementById("results-image-canvas-" + datasetName);
  draw(result.img, resultCanvas);

  const resultPredictionsDiv = document.getElementById("results-image-predictions-inner-" + datasetName);
  while (resultPredictionsDiv.firstChild) {
    resultPredictionsDiv.removeChild(resultPredictionsDiv.firstChild);
  }

  for (let i = 0; i < result.predictedLabels.length; i++) {
    const currentLabel = result.predictedLabels[i];
    const currentValue = result.predictedValues[i];

    const resultPredictionSpan = document.createElement("span");
    resultPredictionSpan.innerText = currentLabel + ": " + currentValue.toFixed(5);

    if (currentLabel === result.actualLabel) {
      resultPredictionSpan.setAttribute("class", "result-image-prediction correct");
    } else {
      resultPredictionSpan.setAttribute("class", "result-image-prediction incorrect");
    }

    resultPredictionsDiv.appendChild(resultPredictionSpan);
  }
}

// Handlers for switching tabs between training and testing
const trainingTabButton = document.getElementById("training-tab");
const testingTabButton = document.getElementById("testing-tab");

const trainingContainer = document.getElementById("training");
const testingContainer = document.getElementById("testing");

const webcam = document.getElementById("webcam");
const webcamBoxTraining = document.getElementById("webcam-box-inner-training");
const webcamBoxTesting = document.getElementById("webcam-box-inner-testing")

trainingTabButton.addEventListener('click', () => {
  trainingContainer.style.display = "";
  testingContainer.style.display = "none";

  trainingTabButton.className += " active";
  testingTabButton.className = testingTabButton.className.replace(" active", "");

  webcam.parentNode.removeChild(webcam);
  webcamBoxTraining.appendChild(webcam);

  currentTab = tabNames[0];
});

testingTabButton.addEventListener('click', () => {
  trainingContainer.style.display = "none";
  testingContainer.style.display = "";

  trainingTabButton.className = trainingTabButton.className.replace(" active", "");
  testingTabButton.className += " active";

  webcam.parentNode.removeChild(webcam);
  webcamBoxTesting.appendChild(webcam);

  currentTab = tabNames[1];
});

export function getCurrentTab() {
  return currentTab;
}

// Handlers for drawing images on canvases
export function drawThumb(img, datasetName, labelId) {
  const thumbCanvas = document.getElementById("label-images-canvas-" + datasetName + "-" + labelId);
  draw(img, thumbCanvas);
}

export function draw(image, canvas) {
  const [width, height] = [224, 224];
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
    imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
    imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
