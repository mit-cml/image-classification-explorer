// @ts-ignore
import React from 'react';
import chroma from 'chroma-js';
// import spectrogram from 'spectrogram';
// import spectrogram from 'spectrogram-fork';
import spectrogram from './spectrogram/spectrogram';
import { ReactMic } from '@cleandersonlobo/react-mic';
import Button from 'react-bootstrap/Button';
import DropdownButton from 'react-bootstrap/DropdownButton';
import Dropdown from 'react-bootstrap/Dropdown';
import ProgressBar from 'react-bootstrap/ProgressBar';

import './App.css';
import redCircle from'./images/red_circle.png';

class Audio extends React.Component {
    constructor(props) {
        super(props)

        this.recordInterval = undefined

        this.maxAudioTime = 1
        this.startRecording = this.startRecording.bind(this)
        this.stopRecording = this.stopRecording.bind(this)
        this.onStop = this.onStop.bind(this)
        this.decrement = this.decrement.bind(this)
        this.timer = this.timer.bind(this)
        this.getSpectrogram = this.getSpectrogram.bind(this)
        this.formatRecordButtonText = this.formatRecordButtonText.bind(this)

        this.state = { 
            record: false,
            recordProgress: 0,
            countdown: this.maxAudioTime,   
            allLabels: this.props.allLabels,
            currentLabel: this.props.allLabels.length === 0 ? undefined : this.props.allLabels[0],        
        }

    }

    componentWillReceiveProps(nextProps) {
        this.setState({ allLabels: nextProps.allLabels });
        if(this.state.currentLabel === undefined) {
            this.setState({currentLabel: nextProps.allLabels[0]})
        } 
        
    }
    
    decrement = () => {
        this.setState({countdown: this.state.countdown-1}, () => {
            if(this.state.countdown === 0) {
                this.stopRecording()
                this.setState({countdown: this.maxAudioTime})
            }
            else {
                this.timer()
            }
        })
        
    }

    timer = () => {
        var t = setTimeout(this.decrement, 1000);

    }

    startRecording = () => {
        this.setState({
          record: true,
        });
        this.recordInterval = setInterval(() => {
            this.setState({recordProgress: this.state.recordProgress + (20/this.maxAudioTime)})
            // console.log(this.state.recordProgress)
          },50)
        this.timer()
    }
    
    stopRecording = () => {
        this.setState({
            record: false,
            recordProgress: 0
        });
        clearInterval(this.recordInterval)
    }

    onStop = (recordedBlob) => {
        console.log('recordedBlob is: ', recordedBlob);
        console.log('recordedBlob.blob is: ', recordedBlob.blob);

        this.getSpectrogram(recordedBlob).then(reader => {
            reader.onload = () => { 
                let result = reader.result;
                // console.log("IMAGE STATE UPDATED")
                // this.setState({image: result}); 
                this.props.handleNewImage(result, this.state.currentLabel)
            }
        })

        // this.getSpectrogram(recordedBlob)

        // var blob = new Blob([recordedBlob], {type: "audio/wav"})
        // var url = window.URL.createObjectURL(blob);
    
        // this.href = url;
        // this.target = '_blank';
        
        // // target filename
        // this.download = 'test.wav';

        // var a = document.createElement("a");
        // document.body.appendChild(a);
        // a.style = "display: none";
        // var blob = new Blob([recordedBlob.blob], {type: "audio/wav"}),
        //     url = window.URL.createObjectURL(blob);
        // a.href = url;
        // a.download = "bbb.wav";
        // a.click();
        // window.URL.revokeObjectURL(url);
        // document.body.removeChild(a);
    }

    formatRecordButtonText() {
        if(this.state.record) {
            return <img src={redCircle} width={10} height={10}/>
        }
    }

    getSpectrogram = async (recordedBlob) => {
        console.log("AUDIO BLOB:")
        console.log(recordedBlob)

        const response = await fetch('https://c1.appinventor.mit.edu/spectrogram', {
        // const response = await fetch('/spectrogram', {
                method: 'POST',
            body: recordedBlob.blob
        });
        const blob = await response.blob()
        // let matrixBlob = new Blob([res.data], {type:"image/png"}); 
        let reader = new FileReader();
        reader.readAsDataURL(blob); 
        return reader;

        // var canvas = document.createElement('canvas')
        // canvas.width = 200;
        // canvas.height = 200;
        // // document.body.appendChild(canvas)
 
        // var spectro = spectrogram(canvas, {
        //     audio: {
        //         enable: false
        //     },
        //     colors: (steps) => {
        //         var baseColors = [[65,65,90,1], [0,255,255,1], [0,255,0,1], [255,255,0,1], [ 255,0,0,1]];
        //         var positions = [0, 0.15, 0.30, 0.50, 0.75];
             
        //         var scale = new chroma.scale(baseColors, positions)
        //         .domain([0, steps]);
             
        //         var colors = [];
             
        //         for (var i = 0; i < steps; ++i) {
        //           var color = scale(i);
        //           colors.push(color.hex());
        //         }
             
        //         return colors;
        //       }
        // });
        
        // var audioContext = new AudioContext({sampleRate: 384000});
        // var request = new XMLHttpRequest();
        // request.open('GET', URL.createObjectURL(recordedBlob.blob), true);
        // request.responseType = 'arraybuffer';

        // request.onload = () => {
        //     audioContext.decodeAudioData(request.response, (buffer) => {
        //         spectro.connectSource(buffer, audioContext);
        //         var canvasPromise = new Promise(function(resolve, reject) {
        //             spectro.start(0, resolve);
        //         })
        //         canvasPromise.then(() => {
        //             console.log("Promise Resolved")
        //             var dataURL = canvas.toDataURL();
        //             this.props.handleNewImage(dataURL, this.state.currentLabel)
        //         })
        //     });
        // };
        // request.send();
        




    }

    handleDropdownSelect(selectedLabel) {
        this.setState({currentLabel: selectedLabel})
    }

    render () {
        console.log(this.state.recordProgress)
        return (
            <div className="record-box">
                <div className="recording-for-dropdown-wrapper">
                    <a className="record-p">RECORDING FOR: &nbsp;</a>
                    <DropdownButton 
                        disabled={this.state.allLabels.length === 0}
                        title={this.state.allLabels.length === 0 ? "No Labels" : this.state.currentLabel} 
                        size="sm" 
                        variant="outline-light"
                    >
                        {this.state.allLabels.map(l => {
                            return (
                                <Dropdown.Item key={l} onClick={() => this.handleDropdownSelect(l)}>{l}</Dropdown.Item>
                            )
                        })}
                    </DropdownButton>
                </div>
                <ReactMic
                    record={this.state.record}
                    className="sound-wave"
                    onStop={this.onStop}
                    strokeColor="#adc2ff"
                    backgroundColor="#1a1a1d" 
                    width={200}
                    height={140}
                    />
                <div className="record-and-countdown">
                    <Button onClick={this.startRecording} disabled={this.state.allLabels.length === 0} variant={this.state.record ? "outline-danger" : "outline-light"} className="record-button">
                        {/* {this.state.record ? [
                            <img key={0} src={redCircle} width={10} height={10}/>,
                            <span key={1}>&nbsp;&nbsp;</span>,
                            "Recording",
                        ] : "Record"} */}
                        {"Record"}
                    </Button>
                    {/* <p className="countdown">{"0:0" + (this.maxAudioTime - this.state.countdown)}</p> */}

                </div>
                <ProgressBar animated variant="danger" style={{width: 100, borderRadius: 3, backgroundColor: "#e9ecef", visibility: this.state.recordProgress == 0 ? "hidden" : "visible"}} now={this.state.recordProgress} />              
                
                {/* <img src={this.state.image} height="100" alt="Image preview..."></img> */}
            </div>
        )
        
    }
}

export default Audio;
