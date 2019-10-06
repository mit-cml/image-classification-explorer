// @ts-ignore
import React from 'react';
import { ReactMic } from '@cleandersonlobo/react-mic';
import Button from 'react-bootstrap/Button';
import DropdownButton from 'react-bootstrap/DropdownButton';
import Dropdown from 'react-bootstrap/Dropdown';

import './App.css';
import redCircle from'./images/red_circle.png';

class Audio extends React.Component {
    constructor(props) {
        super(props)

        this.maxAudioTime = 2
        this.startRecording = this.startRecording.bind(this)
        this.stopRecording = this.stopRecording.bind(this)
        this.onStop = this.onStop.bind(this)
        this.decrement = this.decrement.bind(this)
        this.timer = this.timer.bind(this)
        this.getSpectrogram = this.getSpectrogram.bind(this)
        this.formatRecordButtonText = this.formatRecordButtonText.bind(this)

        this.state = { 
            record: false,
            countdown: this.maxAudioTime,   
            allLabels: this.props.allLabels,
            currentLabel: this.props.allLabels[0],        
        }

    }

    componentWillReceiveProps(nextProps) {
        this.setState({ allLabels: nextProps.allLabels });  
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
          record: true
        });
        this.timer()
    }
    
    stopRecording = () => {
        this.setState({
            record: false
        });
    }

    onStop = (recordedBlob) => {
        console.log('recordedBlob is: ', recordedBlob);

        this.getSpectrogram(recordedBlob).then(reader => {
            reader.onload = () => { 
                let result = reader.result;
                // console.log("IMAGE STATE UPDATED")
                // this.setState({image: result}); 
                this.props.handleNewImage(result, this.state.currentLabel)
            }
        })

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
        const response = await fetch('/spectrogram', {
            method: 'POST',
            body: recordedBlob.blob
        });
        const blob = await response.blob()
        // let matrixBlob = new Blob([res.data], {type:"image/png"}); 
        let reader = new FileReader();
        reader.readAsDataURL(blob); 
        
        return reader;
    }

    handleDropdownSelect(selectedLabel) {
        this.setState({currentLabel: selectedLabel})
    }
    render () {
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
                    <Button onClick={this.startRecording} disabled={this.state.allLabels.length === 0} variant="outline-light" className="record-button">
                        {this.state.record ? [
                            <img key={0} src={redCircle} width={10} height={10}/>,
                            <span key={1}>&nbsp;&nbsp;</span>,
                            "Recording",
                        ] : "Record"}
                        {/* {<img src={redCircle} width={10} height={10}/>} &#8239; {this.state.record ? "Recording 0:0" + this.state.countdown : "Record"} */}
                    </Button>
                    <p className="countdown">{"0:0" + (this.maxAudioTime - this.state.countdown)}</p>
                </div>
                
                {/* <img src={this.state.image} height="100" alt="Image preview..."></img> */}
            </div>
        )
        
    }
}

export default Audio;
