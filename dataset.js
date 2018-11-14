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

/**
 * A class representing a dataset which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */
export class Dataset {
  constructor() {
    // Current number of active labels
    this.numLabels = 0;

    // Keeps track of total number of unique labels added
    this.totalNumLabelsAdded = 0;

    // Keeps track of all active labels and maps their ids to their names
    this.activeLabels = {};

    // Maps model label prediction numbers to label names
    this.currentLabelNamesMap = {};

    // A mapping from class labels to lists of images users uploaded for that class (imgs as in those returned by webcam.capture())
    this.labelImgs = {};

    // A mapping from class labels to lists of activation tensors obtained from corresponding images in this.classImgs
    this.labelXs = {};
  }

  /**
   * Adds an example to the controller dataset.
   * @param {Tensor} image A tensor of the example image representing the example. It can be an image,
   *     an activation, or any other type of Tensor.
   * @param {Tensor} activation A tensor of an activation of image
   * @param {number} label The label of the example. Should be a number.
   */
  addExample(image, activation, labelId) {
    if (!(labelId in this.labelImgs)) {
      // For the first example that gets added for each label, keep the image 
      // and its activation so that the Dataset owns the memory of 
      // the inputs. This makes sure that if addExample() is called in a tf.tidy(),
      // these Tensors will not get disposed.
      this.labelImgs[labelId] = tf.keep(image);
      this.labelXs[labelId] = tf.keep(activation);
    } else {
      const oldImgs = this.labelImgs[labelId];
      this.labelImgs[labelId] = tf.keep(oldImgs.concat(image, 0));

      const oldXs = this.labelXs[labelId];
      this.labelXs[labelId] = tf.keep(oldXs.concat(activation, 0));

      oldImgs.dispose();
      oldXs.dispose();
    }
  }

  getLabelNameFromModelPrediction(prediction) {
    return this.currentLabelNamesMap[prediction];
  }

  getCurrentLabelNamesJson() {
    return JSON.stringify(this.currentLabelNamesMap);
  }

  setCurrentLabelNames(labelNamesJson) {
    this.currentLabelNamesMap = JSON.parse(labelNamesJson);
  }

  addLabel(labelName) {
    this.activeLabels[this.totalNumLabelsAdded] = labelName;
    this.numLabels += 1;
    this.totalNumLabelsAdded += 1;

    return this.totalNumLabelsAdded - 1;
  }

  removeLabel(labelId) {
    delete this.activeLabels[labelId];
    this.numLabels -= 1;
  }

  getImages(labelId) {
    return this.labelImgs[labelId];
  }

  getData() {
    const activeLabelIds = Object.keys(this.activeLabels);

    this.currentLabelNamesMap = {};

    // Initialize the xs and ys
    let labelXs = tf.tensor([]);
    let labelYs = tf.tensor([]);

    // Initialize the label images
    let labelImgs = tf.tensor([]);

    // Adding the labels' xs and ys and images
    for (let i = 0; i < activeLabelIds.length; i++) {
      this.currentLabelNamesMap[i] = this.activeLabels[activeLabelIds[i]];

      const currentLabelXs = this.labelXs[activeLabelIds[i]];
      labelXs = labelXs.concat(currentLabelXs, 0);

      const currentLabelImgs = this.labelImgs[activeLabelIds[i]];
      labelImgs = labelImgs.concat(currentLabelImgs, 0);

      const currentY = tf.oneHot(tf.tensor1d([i]).toInt(), this.numLabels);
      const currentNumXs = currentLabelXs.shape[0];
      labelYs = labelYs.concat(currentY.tile([currentNumXs, 1]), 0);
    }

    return {'xs': tf.keep(labelXs), 'ys': tf.keep(labelYs), 'imgs': tf.keep(labelImgs)};
  }
}
