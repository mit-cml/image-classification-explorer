import { None } from "vega";

// import * as tf from '@tensorflow/tfjs';

/**
 * A class that represents a layer in our editable model.
 */
export class Layer {
  /**
   * @param {String} i The layer id suffix (i.e. "0", "1", "2", ..., "final")
   */
  constructor(i) {
    this.id = i;
    this.inputWrapperId = `inputWrapper-${i}`;
    this.layerDimsDisplayId = `dimensions-${i}`;
    this.inputDims = None;
    this.outputDims = None;
  }

  updateDims(inputDims){
      
  }

  /**
   * Adjusts the video size so we can make a centered square crop without
   * including whitespace.
   * @param {number} width The real width of the video element.
   * @param {number} height The real height of the video element.
   */
  adjustVideoSize(width, height) {
    const aspectRatio = width / height;
    if (width >= height) {
      this.webcamElement.width = aspectRatio * this.webcamElement.height;
    } else if (width < height) {
      this.webcamElement.height = this.webcamElement.width / aspectRatio;
    }
  }
}
