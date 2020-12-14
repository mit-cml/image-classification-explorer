// @ts-ignore
import React from 'react';
import * as JSZip from 'jszip';
import * as FileSaver from 'file-saver';
import Button from 'react-bootstrap/Button';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Popover from 'react-bootstrap/Popover'
import Jumbotron from 'react-bootstrap/Jumbotron'
import Container from 'react-bootstrap/Container'
import Alert from 'react-bootstrap/Alert';
import ProgressBar from 'react-bootstrap/ProgressBar';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import Cam from './Cam.js';

import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';

import { Link } from 'react-router-dom';

class TestView extends React.Component {
    constructor(props) {
        super(props)
        this.canvasRef = React.createRef();
        this.handleTestImage = this.handleTestImage.bind(this)
        this.exportModel = this.exportModel.bind(this)
        this.exportData = this.exportData.bind(this)
        this.handleDragOver = this.handleDragOver.bind(this);
        this.handleDragExit = this.handleDragExit.bind(this);
        this.handleDragLeave = this.handleDragLeave.bind(this);
        this.handleDrop = this.handleDrop.bind(this);
        this.state = { 
            imageMap: this.props.location.state.imageMap,
            loadedModel: this.props.location.state.loadedModel,
            transferModel: undefined,
            customModel: undefined,
            trainActivations: undefined,
            testImage: undefined,
            testRanks: undefined,
            testConfidences: undefined,
            testResults: undefined,
            loading: true,
            progress: 0,
            message: "Loading...",
            names: "test-pic-background",
        }
    }

    async componentDidMount() {
        await this.train()
    }

    async getTransferModel() {
        console.log(tf.version.tfjs)
        console.log('Loading mobilenet..');
        this.setState({message: "Loading mobilenet...", progress: 20})
        const model = await tf.loadLayersModel(
            "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
        );
        const layer = model.getLayer("conv_pw_13_relu")
        let modelModified = tf.model({inputs: model.inputs, outputs: layer.output, name: 'modelModified' });
        console.log('Mobilenet model is modified')
        this.setState({message: "Truncating mobilenet...", progress: 50})
        modelModified.summary()
        return modelModified
    }

    async getImg(imgUrl) {
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
        return img
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

    async train() {
        const transferModel = await this.getTransferModel();
        const customModel = this.state.loadedModel? this.state.loadedModel : this.getCustomModel();
        customModel.summary()
        const {images, labels} = await this.generateInputTensors();
        const activations = this.generateActivations(images, transferModel);
        const testResults = {}
        Object.keys(this.state.imageMap).sort().forEach((k) => {
            testResults[k] = []
        })
        this.setState({
            trainActivations: activations,
            customModel: customModel,
            transferModel: transferModel,
            testResults: testResults,
        });
        await this.fitCustomModel(activations, labels, customModel)
        this.setState({loading: false})
    }

    async fitCustomModel(activations, labels, customModel) {
        let batchSize = Math.floor(activations.shape[0] * .4)
        try {
            await customModel.fit(activations, labels, {
                batchSize,
                epochs: 20,
                callbacks: {
                  onBatchEnd: async (batch, logs) => {
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
        this.setState({message: "Building custom model...", progress: 70})
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
            imageMapConverted[label].forEach(tensor => {
                imageTensors.push(tensor)
                var labelTensor = []
                allLabels.forEach(l => {
                    labelTensor.push(l === label ? 1 : 0)
                })
                labelTensors.push(labelTensor)
            })
        })
        var images = tf.stack(imageTensors)
        var labels = tf.stack(labelTensors)
        return {images, labels}
    }

    async exportModel() {
        // The TensorFlow.js save method doesn't work properly in Firefox, so we write
        // our own. This methods zips up the model's topology file, weights files, and
        // a json of the mapping of model predictions to label names. The resulting file
        // is given the .mdl extension to prevent tampering with.
        const zipSaver = {save: (modelSpecs) => {
            const modelTopologyFileName = "model.json";
            const weightDataFileName = "model.weights.bin";
            const modelLabelsName = "model_labels.json";
            const transferModelInfoName = "transfer_model.json";
            const modelZipName = "model.mdl";
            const weightsBlob = new Blob(
            [modelSpecs.weightData], {type: 'application/octet-stream'});
            const weightsManifest = [{
            paths: ['./' + weightDataFileName],
            weights: modelSpecs.weightSpecs
            }];
            const modelTopologyAndWeightManifest = {
            modelTopology: modelSpecs.modelTopology,
            weightsManifest
            };
            const modelTopologyAndWeightManifestBlob = new Blob(
            [JSON.stringify(modelTopologyAndWeightManifest)],
            {type: 'application/json'});
            const zip = new JSZip();
            zip.file(modelTopologyFileName, modelTopologyAndWeightManifestBlob);
            zip.file(weightDataFileName, weightsBlob);
            const labels = {}
            Object.keys(this.state.imageMap).sort().forEach((label, i) => {
                labels[i] = label
            })
            console.log(JSON.stringify(labels))
            zip.file(modelLabelsName, JSON.stringify(labels));
            zip.file(transferModelInfoName, "{\"name\":\"mobilenet\",\"lastLayer\":\"conv_pw_13_relu\",\"url\":\"https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json\"}");
            zip.generateAsync({type:"blob"})
            .then(function (blob) {
                FileSaver.saveAs(blob, modelZipName);
            });
        }};
        await this.state.customModel.save(zipSaver);
    }

    async exportData() {
        const imagesFileName = "images.json";
        const dataZipName = "data.zip";
        const zip = new JSZip();
        zip.file(imagesFileName, JSON.stringify(this.props.location.state.imageMap));
        zip.generateAsync({type:"blob"})
        .then(function (blob) {
            FileSaver.saveAs(blob, dataZipName);
        });
    }

    handleTestImage(image) {
        let cropCanvas = document.createElement('canvas');
        cropCanvas.width = 224;
        cropCanvas.height = 224;
        let cropCtx = cropCanvas.getContext('2d');
        var img = new Image();
        img.src = image;
        img.onload = () => {
            cropCtx.drawImage(img, 0, 0, 224, 224);
            cropCtx.save();
            let croppedImage = cropCanvas.toDataURL();
            this.setState({
                testImage: croppedImage,
            }, async () => await this.test())
        };
    }

    handleDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
        this.setState({names: "test-pic-background droppable"});
    }
    handleDragExit(e) {
        this.setState({names: "test-pic-background"});
    }
    handleDragLeave(e) {
        this.setState({names: "test-pic-background"});
    }
    handleDrop(e) {
        for (let i = 0; i < e.dataTransfer.files.length; i++) {
            let file = e.dataTransfer.files[i];
            const name = file.name;
            let reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                this.handleTestImage(reader.result, this.props.name);
            };
            reader.onerror = () => {
                console.log("Failed to load" + name);
            } 
          }
          this.setState({names: "test-pic-background"});
          e.preventDefault();
    }

    async test() {
        let transferModel = this.state.transferModel
        let customModel = this.state.customModel
        var convertedImg = await this.convertImg(this.state.testImage)
        var activation = transferModel.predict(
            tf.stack([convertedImg])
        )
        var prediction = customModel.predict(activation)
        var top = await prediction.topk(Object.keys(this.state.imageMap).length)
        let confidences = await top.values.data()
        let ranks = await top.indices.data()
        var testRanks = {}
        var testConfidences = {}
        Object.keys(this.state.imageMap).sort().forEach((label, index)=> {
            let rank =  ranks.indexOf(index)
            if(rank === 0) {
                this.setState({testResults: {
                    ...this.state.testResults,
                    [label]: [...this.state.testResults[label], [this.state.testImage, confidences[rank]]]
                }})
            }
            testRanks[label] = rank
            testConfidences[label] = confidences[rank]
        })
        this.setState({
            testRanks: testRanks,
            testConfidences: testConfidences,
        })
    }

    render () {
        if(this.state.loading) {
            return (
                <header className="App-header">
                    <Navbar bg="dark" variant="dark">
                        <Navbar.Brand href="/">Personal Image Classifier</Navbar.Brand>
                        <Nav className="mr-auto">
                            <Link to={{ pathname: "/", state: {imageMap: this.state.imageMap}}}>
                                Train
                            </Link>
                            <Link to={{ pathname: "/test", state: {imageMap: this.state.imageMap}}}>
                                Test
                            </Link>
                        </Nav>
                    </Navbar>
                    <div className="loading">
                        <p className="message">{this.state.message}</p>
                        <ProgressBar animated variant="warning" style={{width: "500px"}} now={this.state.progress} />              
                    </div>
                    <div></div>
                </header>
            )
        } else {
            return (
                <div>
                    <header className="App-header">
                        <Navbar bg="dark" variant="dark">
                            <Navbar.Brand href="/">Personal Image Classifier</Navbar.Brand>
                            <Nav className="mr-auto">
                                <Link to={{ pathname: "/", state: {imageMap: this.state.imageMap}}}>
                                    Train
                                </Link>
                                <Link to={{ pathname: "/test", state: {imageMap: this.state.imageMap}}}>
                                    Test
                                </Link>
                            </Nav>
                        </Navbar>
                        <div className="page-title">Testing Page</div>
                        <div>

                            <div className="view-all">
                                <div className="test-pic-background">
                                    <p className="page-info">With a model now generated, you can simply add images
                                        as you did in the Training portion to classify them. You can then scroll down
                                        to see an overview of the results. When done, you can export the model and data
                                        for later use.
                                    </p>
                                </div>
                            <Cam 
                                handleNewImage={this.handleTestImage}
                                allLabels={Object.keys(this.state.imageMap)}
                                testing={true}
                            />
                            <div className={this.state.names} onDragOver={this.handleDragOver} onDragExit={this.handleDragExit} onDragLeave={this.handleDragLeave} onDrop={this.handleDrop}>
                                <p className="test-pic-p">CAPTURED PIC:</p>
                                {this.state.testImage? <img src={this.state.testImage} alt="test" className="test-pic hover"></img> : <></>}
                            </div>
                            <div className="results-background">
                                <p className="classification-p">CLASSIFICATION:</p>
                                {Object.keys(this.state.imageMap).sort().map((k, i, arr) => {
                                    if(this.state.testRanks !== undefined) {
                                        return (
                                            <div className="results-bubble" key={k}>
                                                <OverlayTrigger
                                                trigger={['hover', 'focus']}
                                                placement="right"
                                                overlay={
                                                    <Popover>
                                                        <Popover.Title as="h3">{"Details"}</Popover.Title>
                                                        <Popover.Content>
                                                            <strong>Confidence: {Math.round(this.state.testConfidences[k]*10000)/100}%    </strong>
                                                            <div style={{marginTop:"5px"}}></div>
                                                            <ProgressBar now={Math.round(this.state.testConfidences[k]*10000)/100} variant={this.state.testRanks[k] === 0 ? "success" : "danger"} />
                                                        </Popover.Content>
                                                    </Popover>
                                                }>
                                                    <Alert variant={this.state.testRanks === undefined ? "info" : this.state.testRanks[k] === 0 ? "success" : "danger"}>
                                                        <p align="center" className="results-bubble-title">{k}</p>
                                                    </Alert>
                                                </OverlayTrigger>
                                            </div>
                                        )
                                    } else {
                                        return (
                                            <div className="results-bubble" key={k}>
                                                <Alert variant={this.state.testRanks === undefined ? "info" : this.state.testRanks[k] === 0 ? "success" : "danger"}>
                                                    <p align="center" className="results-bubble-title">{k}</p>
                                                </Alert>
                                            </div>
                                        )
                                    }
                                })}
                            </div>
                        </div>
                            <Button variant={"dark"} className="train-button" onClick={this.exportModel}>Export Model</Button>
                            <Button variant={"dark"} className="train-button" onClick={this.exportData}>Export Training Data</Button>
                        </div>
                        <div></div>
                    </header>
                    <Jumbotron fluid>
                        <Container>
                            <h1>Test Results</h1>
                            <p></p>
                            {this.state.testResults !== undefined && Object.keys(this.state.testResults).sort().map(label => {
                                const results = this.state.testResults[label]
                                return (
                                    <div className="results-row" key={label}>
                                        <div className="test-label-bubble" key={label}>
                                            <Alert variant="info">
                                                <p align="center" className="results-bubble-title">{label}</p>
                                            </Alert>
                                        </div>
                                        <div className="results-cells-wrapper">
                                            {results.map(r => {
                                                const imgUrl = r[0]
                                                const confidence = r[1]
                                                return (
                                                    <div className="results-cell" key={r}>
                                                        <img src={imgUrl} alt="result" width={100} className="results-cell-pic"></img>
                                                        <ProgressBar now={Math.round(confidence*100)} label={Math.round(confidence*100) + "%"}/>
                                                    </div>
                                                )
                                            })}
                                        </div>
                                    </div>
                                )
                            })}
                        </Container>
                    </Jumbotron>
                </div>
            )
        }
    }
}

export default TestView;