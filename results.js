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
 * A class representing results from a classifier
 */
export class Results {

  /**
   * Creates a results object with the provided data.
   *
   * @param {Tensor} imgs - A tensor of the images that the predictions correspond to.
   * @param {number[]} actualIndices - An array of the actual indices of the images.
   * @param {number[]} predictedIndices - An array of the predicted indices of the images.
   * @param {number[]} predictedValues - An array of the confidence values of the predictions.
   * @param {object} labelNamesMap - A mapping of indices to label names.
   */
  constructor(imgs, actualIndices, predictedIndices, predictedValues, labelNamesMap) {
    this.imgs = tf.keep(imgs);
    this.actualIndices = actualIndices;
    this.predictedIndices = predictedIndices;
    this.predictedValues = predictedValues;
    this.labelNamesMap = labelNamesMap;

    this.absoluteIndex = -1;
    this.numResults = actualIndices.length;
    this.topK = predictedIndices.length / this.numResults;
  }

  /**
   * Gets the next result.
   *
   * @returns {object} An object containing data for the next result. Contains fields
   *    'img', 'actualLabel', 'predictedLabels' and 'predictedValues' corresponding to
   *    the result's corresponding image, its actual label, its top predicted labels, and
   *    its confidence values for its top predictions, respectively.
   */
  getNextResult() {
    this.absoluteIndex += 1;
    return this.getResult();
  }

  /**
   * Gets the previous result.
   *
   * @returns {object} An object containing data for the previous result. Contains fields
   *    'img', 'actualLabel', 'predictedLabels' and 'predictedValues' corresponding to
   *    the result's corresponding image, its actual label, its top predicted labels, and
   *    its confidence values for its top predictions, respectively.
   */
  getPreviousResult() {
    this.absoluteIndex -= 1;
    return this.getResult();
  }

  /**
   * Creates and returns the current result object.
   *
   * @returns {object} An object containing data for the current result. Contains fields
   *    'img', 'actualLabel', 'predictedLabels' and 'predictedValues' corresponding to
   *    the result's corresponding image, its actual label, its top predicted labels, and
   *    its confidence values for its top predictions, respectively.
   */
  getResult() {
    const result = {};

    // Calculate what image we are currently on
    const index = ((this.absoluteIndex % this.numResults) + this.numResults) % this.numResults;

    // Get the corresponding image and activation
    const tensorIndex = tf.tensor1d([index], 'int32');
    result.img = this.imgs.gather(tensorIndex);

    // Get the actual label of the image
    result.actualLabel = this.labelNamesMap[this.actualIndices[index]];

    // Get the topk predicted labels of the image
    const topKIndex = index * this.topK;
    const predictedIndices = this.predictedIndices.slice(topKIndex, topKIndex + this.topK);

    result.predictedLabels = [];
    for (let i = 0; i < predictedIndices.length; i++) {
      result.predictedLabels.push(this.labelNamesMap[predictedIndices[i]]);
    }

    result.predictedValues = this.predictedValues.slice(topKIndex, topKIndex + this.topK);

    return result;
  }

  /**
   * Gets all of the results in this results object.
   *
   * @returns {object[]} An array of objects for each result. Each object contains fields
   *    'img', 'actualLabel', 'predictedLabels' and 'predictedValues' corresponding to
   *    the result's corresponding image, its actual label, its top predicted labels, and
   *    its confidence values for its top predictions, respectively.
   */
  getAllResults() {
    const results = [];
    const oldAbsoluteIndex = this.absoluteIndex;

    this.absoluteIndex = -1;

    for (let i = 0; i < this.numResults; i++) {
      results.push(this.getNextResult());
    }

    this.absoluteIndex = oldAbsoluteIndex;
    return results;
  }

  /**
   * Gets the mapping of model predictions to label names.
   *
   * @returns {string} A JSON string of the object containing the mapping.
   */
  getLabelNamesMap() {
    return JSON.stringify(this.labelNamesMap);
  }

}
