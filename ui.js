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

  document.getElementById('label-add-example-training').addEventListener('click', function() {
    addExample(trainingLabelCountSpan, trainingLabelId, "training");
  });

  document.getElementById('label-add-example-testing').addEventListener('click', function() {
    addExample(testingLabelCountSpan, testingLabelId, "testing");
  });

  initProgressBar();
}

// Trackers for the current state of the explorer (training or testing)

const datasetNames = ["training", "testing"];

// Elements dealing with adding labels to the model
const addLabelsInput = document.getElementById('label-name');
const addLabelsButton = document.getElementById('add-labels-button');

// Element for displaying training loss
const trainStatusElement = document.getElementById('train-status');

export function trainStatus(status) {
  trainStatusElement.innerText = status;
}

// Methods to set in index.js that will allow it to pass data to the ui.
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
let trainingLabelCountSpan;
let trainingLabelId;
let trainingLabelBox = null;

let testingLabelCountSpan;
let testingLabelId;
let testingLabelBox = null;

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

  for (let i = 0; i < datasetNames.length; i++) {
    const datasetName = datasetNames[i];
    const labelsOuter = document.getElementById("labels-container-" + datasetName);

    // Overarching div for a label
    const labelBox = labelBoxes[i];
    labelBox.setAttribute("class", "label-container");
    labelBox.setAttribute("id", datasetName + "-" + newLabelId);

    // Div for placing the label name and number of examples
    const labelDescriptionOuter = document.createElement("div");
    labelDescriptionOuter.setAttribute("class", "label-description-outer");

    const labelNameSpan = document.createElement("span");
    labelNameSpan.textContent = newLabelName + " ";

    const labelCountTextPrefix = document.createElement("span");
    labelCountTextPrefix.textContent = " (";
    labelCountTextPrefix.setAttribute("class", "label-count-text");

    const labelCountSpan = document.createElement("span");
    labelCountSpan.textContent = 0;
    labelCountSpan.setAttribute("class", "label-count-text");

    const labelCountTextSuffix = document.createElement("span");
    labelCountTextSuffix.textContent = " examples)";
    labelCountTextSuffix.setAttribute("class", "label-count-text");

    labelDescriptionOuter.appendChild(labelNameSpan);
    labelDescriptionOuter.appendChild(labelCountTextPrefix);
    labelDescriptionOuter.appendChild(labelCountSpan);
    labelDescriptionOuter.appendChild(labelCountTextSuffix);

    // Div where we will put a preview thumbnail of images users have added
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

    // Setting up state changes for clicking on labels to add images for
    if (datasetName === "training") {
      labelBox.addEventListener("click", function() {
        trainingLabelCountSpan = labelCountSpan;
        trainingLabelId = newLabelId;

        if (trainingLabelBox !== null) {
          trainingLabelBox.setAttribute("class", "label-container");
        }

        trainingLabelBox = labelBox;
        trainingLabelBox.setAttribute("class", "label-container active");
      });
    } else if (datasetName === "testing") {
      labelBox.addEventListener("click", function() {
        testingLabelCountSpan = labelCountSpan;
        testingLabelId = newLabelId;

        if (testingLabelBox !== null) {
          testingLabelBox.setAttribute("class", "label-container");
        }

        testingLabelBox = labelBox;
        testingLabelBox.setAttribute("class", "label-container active");
      });
    }

    labelBox.click();

    // Add all elements we just created to the label box and append to
    // the container on the page. Also creates a remove label button if
    // this is for the training tab
    if (datasetName === "training") {
      const labelRemoveOuter = document.createElement("div");
      labelRemoveOuter.setAttribute("class", "label-remove-outer");

      const labelRemove = document.createElement("button");
      labelRemove.setAttribute("class", "label-remove");
      const labelRemoveSpan = document.createElement("span");
      labelRemoveSpan.innerHTML = "&times;";
      labelRemove.appendChild(labelRemoveSpan);

      labelRemove.addEventListener("click", () => {
        removeLabelHandler(newLabelId);
        labelBox.parentNode.removeChild(labelBox);
        labelBoxes[i + 1].parentNode.removeChild(labelBoxes[i + 1]);
      });

      labelRemoveOuter.appendChild(labelRemove);
      labelBox.appendChild(labelRemoveOuter);
    }

    labelBox.appendChild(labelImagesOuter)
    labelBox.appendChild(labelDescriptionOuter);

    labelsOuter.appendChild(labelBox);
  }
}

function addExample(labelCountSpan, labelId, datasetName) {
  labelCountSpan.textContent = parseInt(labelCountSpan.textContent) + 1;
  addExampleHandler(labelId, datasetName);
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
export function updateResult(result) {
  document.getElementById('results-container').style.display = "";

  // First, draw the image to the results canvas
  const resultCanvas = document.getElementById("results-image-canvas");
  draw(result.img, resultCanvas);

  // Then, remove the predictions from the previous result being displayed
  const resultPredictionsDiv = document.getElementById("results-image-predictions-inner");
  while (resultPredictionsDiv.firstChild) {
    resultPredictionsDiv.removeChild(resultPredictionsDiv.firstChild);
  }

  // Then, add the predictions for the new result
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

// Handlers for switching between steps in the progress bar.
const progressBarIdPrefix = "progress-bar-";
const explorerStepIdPrefix = "explorer-step-";
const nextStepPrefix = "next-step-";
const backStepPrefix = "back-step-";
const numSteps = 4;
let currentStep = 0;

const webcam = document.getElementById("webcam");
const webcamBoxTraining = document.getElementById("webcam-box-inner-training");
const webcamBoxTesting = document.getElementById("webcam-box-inner-testing");
const trainingStep = 0;
const testingStep = 2;
const resultsStep = 3;

export function switchSteps(i) {
  for (let j = 0; j < numSteps; j++) {
    const progressBarElementToToggle = document.getElementById(progressBarIdPrefix + j);

    if (j <= i) {
      progressBarElementToToggle.setAttribute("class", "progress-bar-active");
    } else {
      progressBarElementToToggle.setAttribute("class", "");
    }
  }

  if (i === trainingStep) {
    webcam.parentNode.removeChild(webcam);
    webcamBoxTraining.appendChild(webcam);
  } else if (i === testingStep) {
    webcam.parentNode.removeChild(webcam);
    webcamBoxTesting.appendChild(webcam);
  } else if (i === resultsStep) {
    document.getElementsByClassName("analysis-tools-button")[0].click();
  }

  document.getElementById(explorerStepIdPrefix + currentStep).style.display = "none";
  document.getElementById(explorerStepIdPrefix + i).style.display = "";

  currentStep = i;
}

function initProgressBar() {
  for (let i = 0; i < numSteps; i++) {
    const progressBarElementToInit = document.getElementById(progressBarIdPrefix + i);

    progressBarElementToInit.addEventListener("click", function() {
      switchSteps(i);
    });
  }

  for (let i = 0; i < numSteps - 1; i++) {
    const nextButtonToInit = document.getElementById(nextStepPrefix + i);

    nextButtonToInit.addEventListener("click", function() {
      switchSteps(i + 1);
    });
  }

  for (let i = 1; i < numSteps; i++) {
    const backButtonToInit = document.getElementById(backStepPrefix + i);

    backButtonToInit.addEventListener("click", function() {
      switchSteps(i - 1);
    });
  }
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
