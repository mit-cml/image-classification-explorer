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
            labelMap: {},
            testImage: undefined,
            testLabel: undefined,
            modelModified: undefined,
        }
    }

    componentDidMount() {
        this.train()
        if(Math.random() < .5) {
            this.setState({testLabel: Object.keys(this.state.imageMap).sort()[0]})
        } else {
            this.setState({testLabel: Object.keys(this.state.imageMap).sort()[1]})
        }
    }

    async modifyModel() {
        const trainableLayers = [
            'denseModified',
            // 'conv_pw_13_bn',
            // 'conv_pw_13',
            // 'conv_dw_13_bn',
            // 'conv_dw_13'
        ];
        console.log('Loading mobilenet..');
        // const model = await tf.loadLayersModel("http://localhost:3000/mobilenet");
        const model = await tf.loadLayersModel(
            "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
        );

        console.log('Successfully loaded model');
        
        const x=model.getLayer('global_average_pooling2d_1');
        const predictions= tf.layers.dense({
            units: Object.keys(this.state.imageMap).length,  
            activation: 'softmax', 
            name: 'denseModified'
        }).apply(x.output); 
        let modelModified = tf.model({inputs: model.input, outputs: predictions, name: 'modelModified' });
        console.log('Mobilenet model is modified')

        modelModified = this.freezeModelLayers(trainableLayers,modelModified)
        console.log('ModifiedMobilenet model layers are frozen')

        modelModified.compile({
            loss: "categoricalCrossentropy",  
            optimizer: tf.train.adam(1e-3), 
            metrics:   ['accuracy','crossentropy']
        });

        modelModified.summary()
        // model.dispose()
        return modelModified
    }

    convertImg(imgUrl) {
        var img = new Image();
        img.src = imgUrl;
        img.width = 200;
        img.height = 200;

        // var canvas = document.createElement('canvas');
        // var context = canvas.getContext('2d');
        // context.drawImage(img, 0, 0);
        // var imageData = context.getImageData(0, 0, img.width, img.height);

        // console.log(imageData)
        // console.log(this.canvasRef)

        // this.canvasRef.current.getContext("2d").putImageData(imageData, 0, 0)
        // this.imageRef.current.src = imgUrl;

        const trainImage = tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]);
        this.canvasRef.current.getContext("2d").clearRect(0, 0, this.canvasRef.current.width, this.canvasRef.current.height)
        tf.browser.toPixels(trainImage, this.canvasRef.current)
        const trainImageNormalized =  trainImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
        return trainImageNormalized;
    }

    async train() {
        const {images, labels} = this.generateInputTensors();
        const modelModified = await this.modifyModel();
        await this.tuneModel(modelModified, images, labels)
        this.setState({modelModified: modelModified})

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

    async tuneModel(modelModified, images, labels) {
        function onBatchEnd(batch, logs) {
            console.log('Accuracy', logs.acc);
            console.log('CrossEntropy', logs.ce);
            console.log('All', logs);
        }
        console.log('Finetuning the model...');

        await modelModified.fit(images, labels, 
            {
              epochs: 5,
              batchSize: 24,
              validationSplit: 0.2,
              callbacks: {onBatchEnd}
           
            }).then(info => {
                console.log("")
                console.log('Final accuracy', info.history.acc);
                console.log('Cross entropy', info.ce);
                console.log('All', info);
                console.log('All', info.history['acc'][0]);
                
                for (let k = 0; k < 5; k++) {
                    this.traningMetrics.push({acc: 0, ce: 0 , loss: 0});
            
                    this.traningMetrics[k].acc=info.history['acc'][k];
                    this.traningMetrics[k].ce=info.history['ce'][k];
                    this.traningMetrics[k].loss=info.history['loss'][k]; 
                }
                // images.dispose();
                // labels.dispose();
                // modelModified.dispose();
        });;
    }

    generateInputTensors() {
        const imageTensors = [];
        const labelTensors = [];

        var allLabels = Object.keys(this.state.imageMap).sort()
        allLabels.forEach(label => {
            this.state.imageMap[label].forEach(imgUrl => {
                imageTensors.push(this.convertImg(imgUrl))
                var labelTensor = []
                allLabels.forEach(l => {
                    labelTensor.push(l === label ? 1 : 0)
                })
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

    freezeModelLayers(trainableLayers, modelModified) {
        // console.log(modelModified.layers)
        for (const layer of modelModified.layers){
            layer.trainable = false;
            for (const tobeTrained of trainableLayers) {
                if (layer.name.indexOf(tobeTrained) === 0) {
                    layer.trainable = true;
                    break;
                }
            }
        }
        // console.log(modelModified.summary())
        return modelModified;
    }

    handleTestImage(image) {
        this.setState({
            testImage: image,
        })
        this.test()
    }

    test() {
        let modelModified = this.state.modelModified
        console.log("==============================")
        console.log("Model results:")
        modelModified.predict(this.convertImg(this.state.testImage)).print()
    }

    render () {
        console.log(this.state)
        return (
            <div className="view-all">
                <Audio 
                    handleNewImage={this.handleTestImage}
                    allLabels={Object.keys(this.state.imageMap)}/>
                <div className="test-pic-background">
                    <p className="test-pic-p">TEST PICTURE:</p>
                    <img src={this.state.testImage} className="test-pic hover"></img>
                </div>
                <div className="results-background">
                    <p className="test-pic-p">CLASSIFICATION:</p>
                    {Object.keys(this.state.imageMap).sort((x,y) => {
                        if  (x === this.state.testLabel) { return -1 }
                        if(y === this.state.testLabel) { return  1 }
                        else { return 0 }
                    }).map((k, i, arr) => {
                    return (
                        <div className="results-bubble" key={k}>
                            <OverlayTrigger
                                trigger="hover"
                                placement="right"
                                overlay={
                                    <Popover>
                                        <Popover.Title as="h3">{"Details"}</Popover.Title>
                                        <Popover.Content>
                                            <strong>Confidence: 45.0554%    </strong>
                                            <div style={{marginTop:"5px"}}></div>
                                            <ProgressBar now={45} variant={k === this.state.testLabel ? "success" : "danger"} />
                                        </Popover.Content>
                                    </Popover>
                                }>
                                <Alert variant={k === this.state.testLabel ? "success" : "danger"}>
                                    <p align="center" className="results-bubble-title">{k}</p>
                                </Alert>
                            </OverlayTrigger>
                        </div>
                    )
                })}
                </div>
                <div>
                    <canvas ref={this.canvasRef} width={200} height={200} />
                </div>
            </div>
        )
        
    }
}

export default TestView;
