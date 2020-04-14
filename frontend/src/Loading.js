// @ts-ignore
import React from 'react';
import ProgressBar from 'react-bootstrap/ProgressBar';
import Button from 'react-bootstrap/Button';
import * as tf from '@tensorflow/tfjs';

import './App.css';

class Loading extends React.Component {
    constructor(props) {
        super(props)

        this.state = { 
            imageMap: this.props.location.state.imageMap,
            transferModel: undefined,
            customModel: undefined,
            trainActivations: undefined,

            progress: 0,
            message: "Loading...",
            loss: 1,
        }
    }

    async componentDidMount() {
        await this.train()
    }

    async train() {
        const transferModel = await this.getTransferModel();
        const customModel = this.getCustomModel()
        customModel.summary()

        const {images, labels} = await this.generateInputTensors();
        const activations = this.generateActivations(images, transferModel);

        this.setState({
            trainActivations: activations,
            customModel: customModel,
            transferModel: transferModel,
        })

        await this.fitCustomModel(activations, labels, customModel)
       
        console.log("done")

        const saveResultsCustom = await customModel.save('localstorage://customModel');
        const saveResultsTransfer = await transferModel.save('localstorage://transferModel');

        console.log(saveResultsTransfer)
        console.log(saveResultsCustom)

        this.props.history.push({
            pathname: '/test',
            state: {
                customMode: this.state.customModel
            }
        })
    }

    async getTransferModel() {
        console.log('Loading mobilenet..');
        this.setState({message: "Loading mobilenet...", progress: 10})

        const model = await tf.loadLayersModel(
            "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
        );
        console.log('Successfully loaded model');

        const layer = model.getLayer("conv_pw_13_relu")
        let modelModified = tf.model({inputs: model.inputs, outputs: layer.output, name: 'modelModified' });
        console.log('Mobilenet model is modified')
        setTimeout(() => this.setState({message: "Truncating mobilenet...", progress: 30}), 0)

        modelModified.summary()
        return modelModified
    }

    getCustomModel() {
        console.log('Building the model...');
        setTimeout(() => this.setState({message: "Building custom model...", progress: 40}), 0)

        var model = tf.sequential();
        model.add(tf.layers.conv2d({ 
            inputShape: [7, 7, 256],
            kernelSize: 5,
            filters: 5, 
            strides: 1, 
            activation: 'relu',
            kernelInitializer: 'varianceScaling'}))
        
        model.add(tf.layers.flatten())

        model.add(tf.layers.dense({
            units: 100, 
            kernelInitializer: 'varianceScaling', 
            useBias: true,
            activation: 'relu'}))

        model.add(tf.layers.dense({
            kernelInitializer: 'varianceScaling',
            units: Object.keys(this.state.imageMap).length,
            useBias: false,
            activation: 'softmax'}))

        model.compile({optimizer: tf.train.adam(.0001), loss: 'categoricalCrossentropy'});
        return model
    }

    async fitCustomModel(activations, labels, customModel) {
        let batchSize = Math.floor(activations.shape[0] * .4)
        try {
            await customModel.fit(activations, labels, {
                batchSize,
                epochs: 20,
                callbacks: {
                  onBatchEnd: async (batch, logs) => {
                    this.setState({message: "Training model... loss: " + logs.loss.toFixed(5) + "", progress: this.state.progress + 1})
                    console.log('Loss: ' + logs.loss.toFixed(5));
                  },
                }
              }); 
        } catch(e) {
            console.log(e)
        }
    }

    async generateInputTensors() {
        setTimeout(() => this.setState({message: "Generating input tensors...", progress: 30}), 300)

        const imageTensors = [];
        const labelTensors = [];

        var imageMapConverted = {}
        for(let label in this.state.imageMap) {
            imageMapConverted[label] = []
        }

        for(let label in this.state.imageMap) {
            for(let imgIndex in this.state.imageMap[label]) {
                var convertedImg = await this.convertImg(this.state.imageMap[label][imgIndex])
                imageMapConverted[label].push(convertedImg)
            }
        }

        var allLabels = Object.keys(imageMapConverted).sort()
        allLabels.forEach(label => {
            console.log(label)
            imageMapConverted[label].forEach(tensor => {
                imageTensors.push(tensor)
                var labelTensor = []
                allLabels.forEach(l => {
                    labelTensor.push(l === label ? 1 : 0)
                })
                console.log(labelTensor)
                labelTensors.push(labelTensor)
            })
        })
        var images = tf.stack(imageTensors)
        var labels = tf.stack(labelTensors)
        return {images, labels}
    }

    async convertImg(imgUrl) {
        const load = () => new Promise((resolve, reject) => {
            var img = new Image()
            img.onload = () => {
              resolve({img})
            }
            img.src = imgUrl
            img.width = 200;
            img.height = 200;
          });
      
        const {img} = await load()
        const trainImage = tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]);
        const trainImageNormalized =  trainImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
        return trainImageNormalized;
    }

    generateActivations(images, transferModel) {
        let activations = []
        tf.unstack(images).forEach(image => {
            let activation = transferModel.predict(
                tf.stack(
                    [image]
            ))
            activations.push(tf.unstack(activation)[0])
        })
        return tf.stack(activations)
    }

    render() {
        console.log(this.state)
        return (
            <div className="loading">
                <p className="message">{this.state.message}</p>
                <ProgressBar animated variant="warning" style={{width: "500px"}} now={this.state.progress} />              
            </div>
        )
    }
}

export default Loading