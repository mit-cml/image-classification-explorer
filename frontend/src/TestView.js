// @ts-ignore
import React from 'react';
import { ReactMic } from '@cleandersonlobo/react-mic';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Popover from 'react-bootstrap/Popover'
import Alert from 'react-bootstrap/Alert';
import ProgressBar from 'react-bootstrap/ProgressBar';
import './App.css';
import * as tf from '@tensorflow/tfjs';

// import * as mobilenet from './models/mobilenet.json';
import Audio from './Audio.js';
// import * as mobilenet from '@tensorflow-models/mobilenet';


class TestView extends React.Component {
    constructor(props) {
        super(props)
        this.canvasRef = React.createRef();

        this.handleTestImage = this.handleTestImage.bind(this)

        this.state = { 
            imageMap: this.props.location.state.imageMap,
            transferModel: undefined,
            customModel: undefined,
            trainActivations: undefined,

            testImage: undefined,
            testRanks: undefined,
            testConfidences: undefined,

            loading: true,
            progress: 0,
            message: "Loading..."
        }
    }

    async componentDidMount() {
        // if(Math.random() < .5) {
        //     this.setState({testLabel: Object.keys(this.state.imageMap).sort()[0]})
        // } else {
        //     this.setState({testLabel: Object.keys(this.state.imageMap).sort()[1]})
        // }

        await this.train()
    }

    async getTransferModel() {
        // const trainableLayers = [
        //     'denseModified',
        //     'conv_pw_13_bn',
        //     'conv_pw_13',
        //     'conv_dw_13_bn',
        //     'conv_dw_13'
        // ];
        console.log('Loading mobilenet..');
        this.setState({message: "Loading mobilenet...", progress: 30})

        // const model = await tf.loadLayersModel("http://localhost:3000/mobilenet");
        const model = await tf.loadLayersModel(
            "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
        );

        console.log('Successfully loaded model');
        
        // const x=model.getLayer('global_average_pooling2d_1');
        // const predictions= tf.layers.dense({
        //     units: Object.keys(this.state.imageMap).length,  
        //     activation: 'softmax', 
        //     name: 'denseModified'
        // }).apply(x.output); 
        const layer = model.getLayer("conv_pw_13_relu")
        let modelModified = tf.model({inputs: model.inputs, outputs: layer.output, name: 'modelModified' });
        console.log('Mobilenet model is modified')
        setTimeout(() => this.setState({message: "Truncating mobilenet...", progress: 50}), 0)

        // modelModified = this.freezeModelLayers(trainableLayers,modelModified)
        // console.log('ModifiedMobilenet model layers are frozen')

        // modelModified.compile({
        //     loss: "categoricalCrossentropy",  
        //     optimizer: tf.train.adam(1e-3), 
        //     metrics:   ['accuracy','crossentropy']
        // });

        modelModified.summary()
        // model.dispose()
        return modelModified
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

        // var canvas = document.createElement('canvas');
        // var context = canvas.getContext('2d');
        // context.drawImage(img, 0, 0);
        // var imageData = context.getImageData(0, 0, img.width, img.height);

        // console.log(imageData)
        // console.log(this.canvasRef)

        // this.canvasRef.current.getContext("2d").putImageData(imageData, 0, 0)
        // this.imageRef.current.src = imgUrl;
        
        // document.body.appendChild(img)

        const trainImage = tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]);
        // this.canvasRef.current.getContext("2d").clearRect(0, 0, this.canvasRef.current.width, this.canvasRef.current.height)
        // tf.browser.toPixels(trainImage, this.canvasRef.current)
        const trainImageNormalized =  trainImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
        return trainImageNormalized;
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

        this.setState({loading: false})

        // const testImgUrl = this.state.imageMap["Hello"][0]
        // const testImgUrl2 = this.state.imageMap["Hello"][1]

        // console.log(this.convertImg(testImgUrl).shape)
        // console.log(tf.stack([this.convertImg(testImgUrl)]).shape)

        // const input = tf.stack([this.convertImg(testImgUrl), this.convertImg(testImgUrl2)])
        // const layers = modelModified.layers
        // for (var i = 1; i < 2; i ++) {
        //     var layer = layers[i]
        //     var output = layer.apply(input)
        //     input = output
        //     output.print()
        // }

        // const modelTest = tf.model({
        //     inputs: modelModified.layers[0].input,
        //     outputs: modelModified.layers[5].output
        // })
        // const result = await modelTest.predict(
        //     tf.stack(
        //         [this.convertImg(testImgUrl)]
        //     )
        // )
        // console.log(result)

        // const input = tf.input({shape: [5]});
        // const denseLayer1 = tf.layers.dense({units: 10, activation: 'relu'});
        // const denseLayer2 = tf.layers.dense({units: 4, activation: 'softmax'});
        // const output = denseLayer2.apply(denseLayer1.apply(input));
        // const modelBasic = tf.model({inputs: input, outputs: output});
        // modelBasic.summary()
        // modelBasic.predict(tf.ones([2, 5])).print();
    }

    async fitCustomModel(activations, labels, customModel) {
        let batchSize = Math.floor(activations.shape[0] * .4)
        try {
            await customModel.fit(activations, labels, {
                batchSize,
                epochs: 20,
                callbacks: {
                  onBatchEnd: async (batch, logs) => {
                    console.log('Loss: ' + logs.loss.toFixed(5));
                    this.setState({message: "Training model... loss: " + logs.loss.toFixed(5) + "", progress: this.state.progress + 3})
                  },
                }
              }); 
        } catch(e) {
            console.log(e)
        }
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

    getCustomModel() {
        // function onBatchEnd(batch, logs) {
        //     console.log('Accuracy', logs.acc);
        //     console.log('CrossEntropy', logs.ce);
        //     console.log('All', logs);
        // }
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

        // await modelModified.fit(images, labels, 
        //     {
        //       epochs: 5,
        //       batchSize: 24,
        //       validationSplit: 0.2,
        //       callbacks: {onBatchEnd}
           
        //     }).then(info => {
        //         console.log("")
        //         console.log('Final accuracy', info.history.acc);
        //         console.log('Cross entropy', info.ce);
        //         console.log('All', info);
        //         console.log('All', info.history['acc'][0]);
                
        //         // for (let k = 0; k < 5; k++) {
        //         //     this.traningMetrics.push({acc: 0, ce: 0 , loss: 0});
            
        //         //     this.traningMetrics[k].acc=info.history['acc'][k];
        //         //     this.traningMetrics[k].ce=info.history['ce'][k];
        //         //     this.traningMetrics[k].loss=info.history['loss'][k]; 
        //         // }
        //         // images.dispose();
        //         // labels.dispose();
        //         // modelModified.dispose();
        // });;
    }

    async generateInputTensors() {
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
                // console.log("Image and Label Tensors:")
                // console.log(imageTensors)
                // console.log(labelTensors)
            })
        })
        var images = tf.stack(imageTensors)
        var labels = tf.stack(labelTensors)
        return {images, labels}
    }


    handleTestImage(image) {
        console.log("image changed")
        this.setState({
            testImage: image,
        }, async () => await this.test())
    }

    async test() {
        let transferModel = this.state.transferModel
        let customModel = this.state.customModel
        var convertedImg = await this.convertImg(this.state.testImage)
        
        var activation = transferModel.predict(
            tf.stack([convertedImg])
        )

        var prediction = customModel.predict(activation)
        var top = await prediction.topk(Math.min(3, Object.keys(this.state.imageMap).length))
        let confidences = await top.values.data()
        let ranks = await top.indices.data()
        console.log(confidences)
        console.log(ranks)

        var testRanks = {}
        var testConfidences = {}
        Object.keys(this.state.imageMap).sort().forEach((label, index)=> {
            let rank =  ranks.indexOf(index)
            testRanks[label] = rank
            testConfidences[label] = confidences[rank]
        })
        this.setState({
            testRanks: testRanks,
            testConfidences: testConfidences,
        })

    }

    render () {
        console.log(this.state)
        if(this.state.loading) {
            return (
                <div className="loading">
                    <p className="message">{this.state.message}</p>
                    <ProgressBar animated variant="warning" style={{width: "500px"}} now={this.state.progress} />              
                </div>
            )
        } else {
            return (
                <div className="view-all">
                    <Audio 
                        handleNewImage={this.handleTestImage}
                        allLabels={Object.keys(this.state.imageMap)}/>
                    <div className="test-pic-background">
                        <p className="test-pic-p">SPECTROGRAM:</p>
                        <img src={this.state.testImage} className="test-pic hover"></img>
                    </div>
                    <div className="results-background">
                        <p className="test-pic-p">CLASSIFICATION:</p>
                        {Object.keys(this.state.imageMap).sort(
                            // (x,y) => {
                            //     if(this.state.testRanks[x] === 0) { return -1 }
                            //     if(this.state.testRanks[y] !== 0) { return  1 }
                            //     else { return 0 }
                            // }
                        ).map((k, i, arr) => {
                        return (
                            <div className="results-bubble" key={k}>
                                {/* <OverlayTrigger
                                    trigger="hover"
                                    placement="right"
                                    overlay={
                                        this.state.testRanks !== undefined &&
                                        <Popover>
                                            <Popover.Title as="h3">{"Details"}</Popover.Title>
                                            <Popover.Content>
                                                <strong>Confidence: {Math.round(this.state.testConfidences[k]*10000)/100}%    </strong>
                                                <div style={{marginTop:"5px"}}></div>
                                                <ProgressBar now={Math.round(this.state.testConfidences[k]*10000)/100} variant={this.state.testRanks[k] === 0 ? "success" : "danger"} />
                                            </Popover.Content>
                                        </Popover>
                                    }> */}
                                    <Alert variant={this.state.testRanks === undefined ? "info" : this.state.testRanks[k] === 0 ? "success" : "danger"}>
                                        <p align="center" className="results-bubble-title">{k}</p>
                                    </Alert>
                                {/* </OverlayTrigger> */}
                            </div>
                        )
                    })}
                    </div>
                </div>
            )
        }

        
    }
}

export default TestView;




    // freezeModelLayers(trainableLayers, modelModified) {
    //     // console.log(modelModified.layers)
    //     for (const layer of modelModified.layers){
    //         layer.trainable = false;
    //         for (const tobeTrained of trainableLayers) {
    //             if (layer.name.indexOf(tobeTrained) === 0) {
    //                 layer.trainable = true;
    //                 break;
    //             }
    //         }
    //     }
    //     // console.log(modelModified.summary())
    //     return modelModified;
    // }