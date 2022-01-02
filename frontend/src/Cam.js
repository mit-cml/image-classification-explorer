// @ts-ignore
import React from 'react';
import Button from 'react-bootstrap/Button';
import DropdownButton from 'react-bootstrap/DropdownButton';
import Dropdown from 'react-bootstrap/Dropdown';
import './App.css';
import * as tf from '@tensorflow/tfjs';

class Cam extends React.Component {
  constructor(props) {
    super(props)
    this.loadedData = this.loadedData.bind(this);
    this.capture = this.capture.bind(this);
    this.webcam = React.createRef()
    this.state = {
      currentLabel: this.props.currentLabel,
      homeLabel: this.props.currentLabel,
    }
  }

  componentDidMount() {
    new Promise((resolve, reject) => {
      const navigatorAny = navigator;
      navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
      if (navigator.getUserMedia) {
        navigator.mediaDevices.getUserMedia(
          { video: true }).then(stream => {
            this.webcam.current.srcObject = stream;
            this.setState({
              working: true, videoWidth: stream.getVideoTracks()[0].getSettings()['width'],
              videoHeight: stream.getVideoTracks()[0].getSettings()['height']
            });
          }).catch(error => { this.setState({ working: false }); });
      } else {
        reject(this.setState({ working: false }));
      }
    });
  }

  static getDerivedStateFromProps(props, state) {
    if (props.currentLabel !== state.currentLabel && props.currentLabel !== state.homeLabel) {
      return { currentLabel: props.currentLabel, homeLabel: props.currentLabel };
    }
    return null;
  }

  handleDropdownSelect(selectedLabel) {
    this.setState({ currentLabel: selectedLabel })
  }

  loadedData() {
    this.adjustVideoSize(this.state.videoWidth, this.state.videoHeight);
  }

  /**
   * Captures a frame from the webcam and normalizes it between -1 and 1.
   * Returns a batched image (1-element batch) of shape [1, w, h, c].
   */
  async capture() {
    const result = tf.tidy((video = this.webcam) => {
      //console.log("Inside Webcam")
      // Reads the image as a Tensor from the webcam <video> element.      
      const webcamImage = tf.browser.fromPixels(video.current);
      //console.log("After fromPixels: "+webcamImage.shape)

      // Crop the image so we're using the center square of the rectangular
      // webcam.
      const croppedImage = tf.keep(this.cropImage(webcamImage));
      return croppedImage;
      //console.log("After croppedImage: "+croppedImage.shape)
    });
    // Do this weird thing where I put the image in a canvas and then
    // pull it from that canvas as a DataURL
    let newImage = await tf.browser.toPixels(result);
    let imageData = new ImageData(newImage, result.shape[0]);
    let canvas = document.createElement('canvas');
    let ctx = canvas.getContext('2d');
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    ctx.putImageData(imageData, 0, 0);
    let imageString = canvas.toDataURL();
    this.props.handleNewImage(imageString, this.state.currentLabel);
  }

  /**
   * Crops an image tensor so we get a square image with no white space.
   * @param {Tensor4D} img An input image Tensor to crop.
   */
  cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
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
      this.webcam.current.width = aspectRatio * this.webcam.current.height;
    } else if (width < height) {
      this.webcam.current.height = this.webcam.current.width / aspectRatio;
    }
  }

  render() {
    return (
      <div className="record-box">
        {!this.props.testing ? <div className="recording-for-dropdown-wrapper">
          <p className="record-p">CAPTURING FOR: &nbsp;</p>
          <DropdownButton
            disabled={this.props.allLabels.length === 0}
            title={this.props.allLabels.length === 0 ? "No Labels" : this.state.currentLabel}
            size="sm"
            variant="outline-light">
            {this.props.allLabels.map(l => {
              return (
                <Dropdown.Item key={l} onClick={() => this.handleDropdownSelect(l)}>{l}</Dropdown.Item>
              )
            })}
          </DropdownButton>
        </div> : <div />}
        <video className='videoflip' hidden={this.state && this.state.working ? '' : 'hidden'} autoPlay playsInline muted id="webcam" width="100" height="100" onLoadedData={this.loadedData} ref={this.webcam}></video>
        <div id="no-webcam" hidden={this.state && this.state.working ? 'hidden' : ''}>
          No webcam found. <br />
          To use this interface, use a device with a webcam.
        </div>
        <div className="record-and-countdown">
          <Button onClick={this.capture} disabled={this.props.allLabels.length === 0} variant="outline-light" className="record-button">
            {"Capture"}
          </Button>
        </div>
      </div>
    )
  }
}

export default Cam;