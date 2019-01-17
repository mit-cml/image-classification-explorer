import * as tf from '@tensorflow/tfjs';

// Given an image Tensor, returns an array of duplicates of the image
function duplicateImg(img, numSamples) {
  // Using an array with a for loop instead of tile for now
  const imgArr = [];

  for (let i = 0; i < numSamples; i++) {
    imgArr.push(tf.clone(img));
  }

  return imgArr;
}

// Given an array of images, adds gaussian noise to each image.
function addGaussianNoise(imgArr, noiseStd) {
  return imgArr.map(img => img.add(tf.randomNormal(img.shape, 0.0, noiseStd)))
}

// Given an array of images, calculates raw saliencies for each image for a model.
function calculateSaliencies(imgArr, mdl) {
  const imgSaliencies = [];

  for (let i = 0; i < imgArr.length; i++) {
    const img = imgArr[i];

    const saliency = tf.tidy(() => {
      let activation = x => mdl.predict(x);
      let gradient = tf.grad(activation);
      return gradient(img).square().sum(3);
    });

    imgSaliencies.push(tf.clone(saliency));
    saliency.dispose();
  }

  return imgSaliencies;
}

// Given an array of images' saliencies, calculates its avg norm saliency
async function calculateAvgNormSaliency(imgArr, clipToPercentile) {
  const total = tf.tidy(() => imgArr.reduce((x, v) => tf.add(x, v)));
  const asTypedArray = (await total.data()).slice();
  asTypedArray.sort();

  let percentileIdx = Math.floor(asTypedArray.length * clipToPercentile);
  const percentileVal = tf.scalar(asTypedArray[percentileIdx]);

  const minVal = total.min();

  let avgNorm = tf.tidy(() => {
    let avg = total.sub(minVal).div(percentileVal.sub(minVal));
    return avg.clipByValue(0.0, 1.0);
  });

  return tf.squeeze(avgNorm);
}

/**
 * Calculates a saliency map for an image using the smoothGrad method.
 *
 * @param {Tensor} img - A Tensor representing the image.
 * @param {number} numSamples - How many samples of the image to take.
 * @param {number} noiseStd - The standard deviation of the gaussian noise to add
 * @param {Tensorflow.Model} mdl - The model to calculate the saliency with
 * @param {number} clipToPercentile - The percentile to clip values to in the saliency map
 */
export async function smoothGrad(img, numSamples, noiseStd, mdl, clipToPercentile) {
  const duplicateImgs = duplicateImg(img, numSamples);
  const noisyImgs = addGaussianNoise(duplicateImgs, noiseStd);
  const imgSaliencies = calculateSaliencies(noisyImgs, mdl);
  return await calculateAvgNormSaliency(imgSaliencies, clipToPercentile);
}
