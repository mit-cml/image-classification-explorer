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
import {Webcam} from './webcam';

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

document.addEventListener('dragover', e => {
  e.preventDefault();
})

// Trackers for the current state of the explorer (training or testing)
const datasetNames = ["training", "testing"];

// Elements dealing with adding labels to the model
const addLabelsInput = document.getElementById('label-name');
const addLabelsButton = document.getElementById('add-labels-button');

addLabelsInput.addEventListener('keydown', e => {
  if (e.keyCode === 13) {
    addLabelsButton.click();
    e.preventDefault();
  }
});

// Element for displaying training loss
const trainStatusElement = document.getElementById('train-status');

export function trainStatus(status) {
  trainStatusElement.innerText = status;
}

// Methods to set in index.js that will allow it to pass data to the ui.
export let addExampleHandler;

/**
 * Set the handler function when an image is added.
 *
 * @param {function(string, string, ?HTMLImageElement=)} handler the handler to
 * be called when an image is added.
 */
export function setAddExampleHandler(handler) {
  addExampleHandler = handler;
}
//copy of setAddExampleHandler but for uploaded files NATALIE
export let addExampleHandlerUpload;
export function setAddExampleHandlerUpload(handler) {
  addExampleHandlerUpload = handler;
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

    labelBox.addEventListener('dragover', e => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
      labelBox.classList.add('droppable');
    });
    labelBox.addEventListener('dragexit', e => {
      labelBox.classList.remove('droppable');
    });
    labelBox.addEventListener('dragleave', e => {
      labelBox.classList.remove('droppable');
    });
    labelBox.addEventListener('drop', e => {
      for (let i = 0; i < e.dataTransfer.files.length; i++) {
        let file = e.dataTransfer.files[i];
        const name = file.name;
        let reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
          let image = new Image(224, 224);
          image.onload = () => {
            addExample(labelCountSpan, newLabelId, datasetName, image);
          };
          image.onerror = () => {
            console.log('Unable to load ' + name);
          };
          image.src = /** @type string */ reader.result;
        };
      }
      labelBox.classList.remove('droppable');
      e.preventDefault();
    });

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

    //NATALIE START ADD UPLOAD BUTTON
    const addLocalImg = document.createElement("div");
    addLocalImg.setAttribute("class", "add-local-image");
    
    const addImgButton = document.createElement("input");
    addImgButton.setAttribute("type", "file");

    addImgButton.addEventListener('change', function(){
        var reader  = new FileReader();
        var imgFile = this.files[0];

        /*reader.onloadend = function () {
          console.log("IN READER ONLOADEND")
          uploadImg.src = reader.result;
          console.log(uploadImg.src)
          //TODO: tf.tidy tempImg and store into the right dataset
        }*/

        //CHECK IF THE LOCAL FILE IS AN IMAGE

        if (imgFile.type.split('/')[0]=='image'){
          //uploadImg.src = URL.createObjectURL(imgFile); no width or height
          //reader.readAsDataURL(imgFile);

          var _URL = window.URL || window.webkitURL;
          var uploadImg = new Image();

          var imgToTensor4D = null;
          
          uploadImg.onload = function () {
            //console.log("INSIDE ONLOAD")
            //shrink smallest part to 224
            console.log("initial width and height: "+uploadImg.width+" "+uploadImg.height)
            if (uploadImg.width > uploadImg.height){
              var ratio = 224/uploadImg.height
              uploadImg.width = uploadImg.width*ratio
              uploadImg.height = 224
            } else{
              var ratio = 224/uploadImg.width
              uploadImg.width = 224
              uploadImg.height = uploadImg.height*ratio
            }
            console.log("ratio: "+ratio)
            console.log("new width and height: "+uploadImg.width+" "+uploadImg.height)

            imgToTensor4D = tf.tidy(() => {
              //copied from webcam.capture
              var webcamImage2 = tf.fromPixels(uploadImg)
              //console.log("After fromPixels: "+webcamImage2.shape)
              var croppedImage2 = webcamPortedCropImage(webcamImage2);
              //console.log("After cropping: "+croppedImage2.shape)
              var batchedImage2 = croppedImage2.expandDims(0);
              //console.log("After expand dims: "+batchedImage2.shape)

              //console.log("Final output: "+batchedImage2.toFloat().div(tf.scalar(127)).sub(tf.scalar(1)).shape)
              return batchedImage2.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));              
            });

            //THIS IS STUFF FROM function addExample(labelCountSpan, labelId, datasetName) replace with newLabelId
            labelCountSpan.textContent = parseInt(labelCountSpan.textContent) + 1;
          
            addExampleHandlerUpload(newLabelId, datasetName, imgToTensor4D);
          }

          uploadImg.src = _URL.createObjectURL(imgFile);
        }
        else{
          alert("The file was not a valid image. Please import JPEG, PNG, or GIF file types only.");
          //CLEARS THE INPUT FILE
          addImgButton.value = null
        }
      }
    );

    addLocalImg.appendChild(addImgButton);
    labelBox.appendChild(addLocalImg);

    //NATALIE END

    labelsOuter.appendChild(labelBox);
  }
}

//NATALIE START HELPER METHOD
//HACKY ported from webcam.js, can figure out how to call it directly later
//for some reason I can't call it from the imported {Webcam} class... Prolly cuz i dont know javascript lol
function webcamPortedCropImage(img) {
  const size = Math.min(img.shape[0], img.shape[1]);
  const centerHeight = img.shape[0] / 2;
  const beginHeight = centerHeight - (size / 2);
  const centerWidth = img.shape[1] / 2;
  const beginWidth = centerWidth - (size / 2);
  return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}
//NATALIE END

function addExample(labelCountSpan, labelId, datasetName, image) {
  labelCountSpan.textContent = parseInt(labelCountSpan.textContent) + 1;
  addExampleHandler(labelId, datasetName, image);
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
    let firstItem = document.getElementById('training-0');
    if (firstItem) {
      firstItem.click();
    }
  } else if (i === testingStep) {
    webcam.parentNode.removeChild(webcam);
    webcamBoxTesting.appendChild(webcam);
    let firstItem = document.getElementById('testing-0');
    if (firstItem) {
      firstItem.click();
    }
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
