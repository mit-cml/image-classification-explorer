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
 * A dataset for webcam controls which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */
export class ControllerDataset {
  constructor() {
    this.numLabels = 0;

    // A mapping from class labels to lists of images users uploaded for that class (imgs as in those returned by webcam.capture())
    this.labelImgs = {};

    // A mapping from class labels to lists of activation tensors obtained from corresponding images in this.classImgs
    this.labelXs = {};

    // Checks whether or not we have calculated xs and ys to return already. If so, we cannot add new labels.
    this.gotXsAndYs = false;
  }

  /**
   * Adds an example to the controller dataset.
   * @param {Tensor} image A tensor of the example image representing the example. It can be an image,
   *     an activation, or any other type of Tensor.
   * @param {Tensor} activation A tensor of an activation of image
   * @param {number} label The label of the example. Should be a number.
   */
  addExample(image, activation, label) {
    if (!(label in this.labelImgs)) {
      // For the first example that gets added for each label, keep the image 
      // and its activation so that the ControllerDataset owns the memory of 
      // the inputs. This makes sure that if addExample() is called in a tf.tidy(),
      // these Tensors will not get disposed.
      this.labelImgs[label] = tf.keep(image);
      this.labelXs[label] = tf.keep(activation);
    } else {
      const oldImgs = this.labelImgs[label];
      this.labelImgs[label] = this.keep(oldImgs.concat(image, 0));

      const oldXs = this.labelImgs[label];
      this.labelXs[label] = tf.keep(oldXs.concat(activation, 0));

      oldImgs.dispose();
      oldXs.dispose();
    }
  }

  addLabel() {
    this.numLabels += 1;
  }

  removeLabel(label) {
    if (label in this.labelImgs) {
      this.labelImgs[label].dispose();
      this.labelXs[label].dispose();

      delete this.labelImgs[label];
      delete this.labelXs[label];
    }
    
    this.numLabels -= 1;
  }

  getImages(label) {
    return this.labelImgs[label];
  }

  getXsAndYs() {
    const allLabels = this.labelXs.keys();

    // Initialize the xs and ys
    let labelXs = tf.tensor([]);
    let labelYs = tf.tensor([]);

    // Adding the labels' xs and ys
    for (let i = 0; i < allLabels.length; i++) {
      const currentLabelXs = this.labelXs[allLabels[i]];
      labelXs = labelXs.concat(currentLabelXs, 0);

      const currentY = tf.oneHot(tf.tensor1d([i]).toInt(), this.numLabels);
      const currentNumXs = currentLabelXs.shape[0];
      labelYs = labelYs.concat(currentY.tile([currentNumXs, 1]), 0);
    }

    this.gotXsAndYs = true;
    return {'xs': tf.keep(labelXs), 'ys': tf.keep(labelYs)};
  }
}
