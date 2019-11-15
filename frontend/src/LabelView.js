// @ts-ignore
import React from 'react';
import { useState } from 'react';
import InputGroup from 'react-bootstrap/InputGroup';
import FormControl from 'react-bootstrap/FormControl'
import Form from 'react-bootstrap/Form'
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Modal from 'react-bootstrap/Modal';

import Popover from 'react-bootstrap/Popover'
import './App.css';
import Label from './Label.js';
import Audio from './Audio.js';
import Button from 'react-bootstrap/Button';

import plus from'./images/plus.png';
import { Link } from 'react-router-dom';

import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';


class LabelView extends React.Component {
    constructor(props) {
        super(props)
        this.handleNewImage = this.handleNewImage.bind(this)
        this.handleRemoveImage = this.handleRemoveImage.bind(this)
        this.handleLabelKeyDown = this.handleLabelKeyDown.bind(this)
        this.createNewLabel = this.createNewLabel.bind(this)
        this.tuneModal = this.tuneModal.bind(this)

        this.state = { 
            imageMap: this.props.location.state === undefined ? {} : this.props.location.state.imageMap,
        }
    }


    downloadURI(uri, name) {
        var link = document.createElement("a");
        link.download = name;
        link.href = uri;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    handleNewImage(image, currentLabel) {
        this.setState({
            imageMap: {
                ...this.state.imageMap,
                [currentLabel]: [...this.state.imageMap[currentLabel], image]
            }
        })

        // this.downloadURI(image, currentLabel)
    }

    handleRemoveImage(i, name) {
        var images = [...(this.state.imageMap[name])]
        var index = images.indexOf(i)
        if (index !== -1) {
            images.splice(index, 1);
            this.setState({imageMap: {
                ...this.state.imageMap,
                [name]: images
            }});
        }
    }

    handleLabelKeyDown(e) {
        if(e.keyCode === 13) {
            this.createNewLabel(e.target.value)
            document.body.click()
        }
    }

    createNewLabel(labelName) {
        this.setState({imageMap: {
            ...this.state.imageMap,
            [labelName]: []
        }})
    }

    tuneModal() {
        const [show, setShow] = useState(false);
      
        const handleClose = () => setShow(false);
        const handleShow = () => setShow(true);
      
        return (
          <>
            <Button 
                variant="dark" 
                size="sm" 
                className="train-button" 
                onClick={handleShow} 
                disabled={Object.keys(this.state.imageMap).length > 0 && Math.min(...Object.keys(this.state.imageMap).map(k => this.state.imageMap[k].length)) > 1 ? false : true}
            >
                Train
            </Button>
            <Modal show={show} onHide={handleClose} centered>
                <Modal.Header closeButton>
                    <Modal.Title>Customize Hyperparameters</Modal.Title>
                </Modal.Header>
                <Modal.Body></Modal.Body>
                <Form>
                    <Form.Group controlId="exampleForm.ControlInput1">
                        <Form.Label>Learning Rate</Form.Label>
                        <Form.Control type="number" defaultValue="0.0001" step=".0001" />
                    </Form.Group>
                    <Form.Group controlId="exampleForm.ControlSelect1">
                        <Form.Label>Optimizer</Form.Label>
                        <Form.Control as="select">
                        <option>Adam</option>
                        <option>SGD</option>
                        <option>Adagrad</option>
                        <option>Adadelta</option>
                        </Form.Control>
                    </Form.Group>
                    <Form.Group controlId="exampleForm.ControlInput2">
                        <Form.Label>Epochs</Form.Label>
                        <Form.Control type="number" defaultValue="20" step="1" />
                    </Form.Group>
                    <Form.Group controlId="exampleForm.ControlInput2">
                        <Form.Label>Training Data Fraction</Form.Label>
                        <Form.Control type="number" defaultValue=".4" step=".1" />
                    </Form.Group>
                </Form>
                <Modal.Footer>
                <Button variant="danger" onClick={handleClose}>
                    Close
                </Button>
                <Link to={{ pathname: "/test", state: {imageMap: this.state.imageMap}}}>
                    <Button variant="warning">  Train Model </Button>
                </Link>
                </Modal.Footer>
            </Modal>
          </>
        );
      }

    render () {
        console.log(this.state.imageMap)
        return (
            <header className="App-header">
                <Navbar bg="dark" variant="dark">
                    <Navbar.Brand href="/">Spectrogram Audio Classifier</Navbar.Brand>
                    <Nav className="mr-auto">
                        <Link to={{ pathname: "/", state: {imageMap: this.state.imageMap}}}>
                            Train
                        </Link>
                        <Link to={{ pathname: "/test", state: {imageMap: this.state.imageMap}}} className={Object.keys(this.state.imageMap).length > 0 && Math.min(...Object.keys(this.state.imageMap).map(k => this.state.imageMap[k].length)) > 1 ? "": "disable-link"}>
                            Test
                        </Link>
                        <Link to={{ pathname: "/", state: {imageMap: this.state.imageMap}}}>
                            Export
                        </Link>
                        {/* <Nav.Link>Train</Nav.Link>
                        <Nav.Link>Test</Nav.Link>
                        <Nav.Link>Export</Nav.Link> */}
                    </Nav>
                </Navbar>
                <div className="view-all">
                <Audio 
                    handleNewImage={this.handleNewImage}
                    allLabels={Object.keys(this.state.imageMap)}/>
                <div className="all-labels">
                    <div className="plus-wrapper">
                        <OverlayTrigger
                            trigger="click"
                            placement={"right"}
                            rootClose={true}
                            overlay={
                                <Popover>
                                    <Popover.Title 
                                        as="h3">{"Create New Label"}</Popover.Title>
                                    <Popover.Content>
                                    <input 
                                        type="text" 
                                        name="name" 
                                        autoFocus={true} 
                                        onKeyDown={this.handleLabelKeyDown}
                                    />
                                    </Popover.Content>
                                </Popover>
                            }
                            >
                            <img src={plus} className="plus"></img>
                        </OverlayTrigger>
                    
                        {/* {Object.keys(this.state.imageMap).length > 0 && Math.min(...Object.keys(this.state.imageMap).map(k => this.state.imageMap[k].length)) > 1 && 
                            <this.tuneModal/>
                        } */}
                        {Object.keys(this.state.imageMap).length > 0  && 
                            <this.tuneModal/>
                        }
                    </div>
                    {Object.keys(this.state.imageMap).map(k => {
                        return (
                            <Label 
                                name={k}
                                images={this.state.imageMap[k]}
                                handleRemoveImage={this.handleRemoveImage}
                                key={k}/>
                        )
                    })}
                </div>

            </div>
                <div></div>
            </header>
        )
    }
}

export default LabelView;
