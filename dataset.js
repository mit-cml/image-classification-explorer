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
 * A class representing a dataset featuring methods for adding labels and
 * examples, as well as for formatting them for training or predicting.
 */
export class Dataset {

  /**
   * Creates an empty dataset.
   */
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
   * Adds an example to the dataset.
   *
   * @param {Tensor} image - A tensor of the example image.
   * @param {Tensor} activation - A tensor of an activation of the example image that will be used for training or predicting.
   * @param {number} labelId - The example's label's id.
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

  /**
   * Gets the name of a label corresponding to a given model prediction.
   *
   * @param {number} prediction - The model's prediction.
   *
   * @returns {string} The corresponding label name.
   */
  getLabelNameFromModelPrediction(prediction) {
    return this.currentLabelNamesMap[prediction];
  }

  /**
   * Gets the mapping of model predictions to label names.
   *
   * @returns {string} A JSON string of the object containing the mapping.
   */
  getCurrentLabelNamesJson() {
    return JSON.stringify(this.currentLabelNamesMap);
  }

  /**
   * Sets the mapping of model predictions to label names.
   *
   * @param {string} labelNamesJsonString - A JSON string of the object containing the mapping.
   */
  setCurrentLabelNames(labelNamesJsonString) {
    this.currentLabelNamesMap = JSON.parse(labelNamesJsonString);
  }

  /**
   * Adds a label to the dataset.
   *
   * @param {labelName} labelName - The name of the label to add.
   *
   * @returns {number} The id of the added label.
   */
  addLabel(labelName) {
    this.activeLabels[this.totalNumLabelsAdded] = labelName;
    this.numLabels += 1;
    this.totalNumLabelsAdded += 1;

    return this.totalNumLabelsAdded - 1;
  }

  /**
   * Removes a label from the dataset
   *
   * @param {number} labelId - The id of the label to remove.
   */
  removeLabel(labelId) {
    delete this.activeLabels[labelId];
    this.numLabels -= 1;
  }

  /**
   * Gets the images corresponding to a given label id.
   *
   * @param {number} labelId - The id of the label to get images for.
   *
   * @returns {Tensor} A Tensor containing the images.
   */
  getImages(labelId) {
    return this.labelImgs[labelId];
  }

  /**
   * Gets the data from the dataset for training or predicting.
   *
   * @returns {object} An object containing the dataset's data. Contains
   *    fields 'xs', 'ys', and 'imgs', corresponding to the training data,
   *    training labels, and their images respectively.
   */
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
      const currentActiveLabelId = activeLabelIds[i];

      this.currentLabelNamesMap[i] = this.activeLabels[currentActiveLabelId];

      if (currentActiveLabelId in this.labelImgs) {
        const currentLabelXs = this.labelXs[currentActiveLabelId];
        labelXs = labelXs.concat(currentLabelXs, 0);

        const currentLabelImgs = this.labelImgs[currentActiveLabelId];
        labelImgs = labelImgs.concat(currentLabelImgs, 0);

        const currentY = tf.oneHot(tf.tensor1d([i]).toInt(), this.numLabels);
        const currentNumXs = currentLabelXs.shape[0];
        labelYs = labelYs.concat(currentY.tile([currentNumXs, 1]), 0);
      }
    }

    return {'xs': tf.keep(labelXs), 'ys': tf.keep(labelYs), 'imgs': tf.keep(labelImgs)};
  }
}
