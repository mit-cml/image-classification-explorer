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
  // document.getElementById('controller').style.display = '';
  statusElement.style.display = 'none';
}

const trainStatusElement = document.getElementById('train-status');

const statusElement = document.getElementById('status');

// Elements dealing with label display
const labelsOuter = document.getElementsByClassName("labels-outer")[0];

const addLabelsInput = document.getElementById('label-name');
const addLabelsButton = document.getElementById('add-labels-button');

export function isPredicting() {
  statusElement.style.visibility = 'visible';
}
export function donePredicting() {
  statusElement.style.visibility = 'hidden';
}
export function trainStatus(status) {
  trainStatusElement.innerText = status;
}

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

/*
<div class="label-box" id="0">
  <div class="label-name">
    <span>Cat</span>
  </div>
  <div
  <div class="label-images">

  </div>
  <button class="label-remove"><span>Remove this label</span></button>
  <button class="label-add-example"><span>Add example</span></button>
</div>
*/

// Handlers for adding a new label
function addLabel() {
  const newLabelName = addLabelsInput.value;
  const newLabelId = addLabelHandler(newLabelName);
  addLabelsInput.value = "";

  const removeLabelButtonText = "Remove this label";
  const addLabelExampleButtonText = "Add example";

  // Overarching div for a label
  const labelBox = document.createElement("div");
  labelBox.setAttribute("class", "label-box");
  labelBox.setAttribute("id", newLabelId.toString());

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
  labelImagesCanvas.setAttribute("id", "label-images-canvas-" + newLabelId);
  labelImagesInner.appendChild(labelImagesCanvas);
  labelImagesOuter.appendChild(labelImagesInner);

  // Button for removing this label
  const labelRemove = document.createElement("button");
  labelRemove.setAttribute("class", "label-remove");
  const labelRemoveSpan = document.createElement("span");
  labelRemoveSpan.textContent = removeLabelButtonText;
  labelRemove.appendChild(labelRemoveSpan);

  labelRemove.addEventListener("click", () => {
    removeLabelHandler(newLabelId);
    labelBox.parentNode.removeChild(labelBox);
  });

  // Button for adding an example to this label
  const labelAddExample = document.createElement("button");
  labelAddExample.setAttribute("class", "label-add-example");
  const labelAddExampleSpan = document.createElement("span");
  labelAddExampleSpan.textContent = addLabelExampleButtonText;
  labelAddExample.appendChild(labelAddExampleSpan);

  labelAddExample.addEventListener("click", function() {
    labelCountSpan.textContent = parseInt(labelCountSpan.textContent) + 1;
    addExampleHandler(newLabelId);
  });

  // Add all elements we just created to the label box and append to
  // the container on the page
  labelBox.appendChild(labelName);
  labelBox.appendChild(labelCount);
  labelBox.appendChild(labelImagesOuter);
  labelBox.appendChild(labelRemove);
  labelBox.appendChild(labelAddExample);

  labelsOuter.appendChild(labelBox);
}

addLabelsButton.addEventListener('click', () => addLabel());

// let mouseDown = false;

// async function handler(label) {
//   mouseDown = true;
//   const className = CONTROLS[label];
//   const button = document.getElementById(className);
//   const total = document.getElementById(className + '-total');
//   while (mouseDown) {
//     addExampleHandler(label);
//     document.body.setAttribute('data-active', CONTROLS[label]);
//     total.innerText = totals[label]++;
//     await tf.nextFrame();
//   }
//   document.body.removeAttribute('data-active');
// }

export function drawThumb(img, labelId) {
  const thumbCanvas = document.getElementById("label-images-canvas-" + labelId);
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
