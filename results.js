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

  getNextResult() {
    this.absoluteIndex += 1;
    return this.getResult();
  }

  getPreviousResult() {
    this.absoluteIndex -= 1;
    return this.getResult();
  }

  getResult() {
    const result = {};
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

  getLabelNamesMap() {
    return JSON.stringify(this.labelNamesMap);
  }

}
