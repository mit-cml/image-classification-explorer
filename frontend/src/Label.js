// @ts-ignore
import React from 'react';
import Button from 'react-bootstrap/Button';
import './App.css';
import Audio from './Audio.js';
import arrow from'./images/arrow.png';

class Label extends React.Component {
    constructor(props) {
        super(props)
        this.handleNewImage = this.handleNewImage.bind(this)
        this.handleNameChange = this.handleNameChange.bind(this)
        this.handleNameKeyPress = this.handleNameKeyPress.bind(this)

        this.state = { 
            images: [],
            name: "Label"
        }

    }

    handleNewImage(image) {
        this.setState({images: [...this.state.images, image]});
    }

    handleRemoveImage(i) {
        var images = [...this.state.images]
        var index = images.indexOf(i)
        if (index !== -1) {
            images.splice(index, 1);
            this.setState({images: images});
        }
    }

    handleNameChange(e) {
        this.setState({name: e.target.value})
    }

    handleNameKeyPress(e) {
        console.log("hello")
        if(e.keyCode === 13){
            e.preventDefault();
            e.target.blur(); 
        }
    }

    render () {
        console.log(this.state.images)
        return (
            <div className="label-border">
                <div className="label-box">
                    <div className="title-box">
                        <form>
                            <input type="text" autoFocus={true} value={this.state.name} onChange={this.handleNameChange} onKeyDown={this.handleNameKeyPress}/>
                        </form>
                    </div>
                    <div className="vertical-line">
                        <img src={arrow} width={10} style={{margin: "auto"}}></img>
                    </div>
                    <Audio 
                        handleNewImage={this.handleNewImage}/>
                    <div className="vertical-line">
                        <img src={arrow} width={10} style={{margin: "auto"}}></img>
                    </div>
                        {/* <p className="image-text">0 EXAMPLES</p> */}
                    <div className="image-box">
                        {this.state.images.map(i => {
                            return (
                                <div id="image-wrapper" key={i}>
                                    <img src={i} className="slide-in-fwd-center image hover"></img>
                                    <p className="text" onClick={() => this.handleRemoveImage(i)}>x</p>
                                </div>
                            )
                        })}
                    </div>
                </div>
            </div>
        )
    }
}

export default Label;
