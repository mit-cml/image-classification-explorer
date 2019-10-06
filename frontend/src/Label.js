// @ts-ignore
import React from 'react';
import Button from 'react-bootstrap/Button';
import './App.css';
import Audio from './Audio.js';
import arrow from'./images/arrow.png';

class Label extends React.Component {
    constructor(props) {
        super(props)

        this.state = { 
            images: this.props.images,
            name: this.props.name,
        }

    }

    componentWillReceiveProps(nextProps) {
        this.setState({ images: nextProps.images });  
    }

    render () {
        return (
            <div className="label-border">
                <div className="label-box">
                    <div className="title-box">
                        <p className="name-p">{this.state.name}</p>
                        <p className="count-p">{this.state.images.length + (this.state.images.length === 1 ? " example" : " examples")}</p>
                    </div>
                    <div className="vertical-line">
                        <img src={arrow} width={10} style={{margin: "auto"}}></img>
                    </div>
 
                        {/* <p className="image-text">0 EXAMPLES</p> */}
                    <div className="image-box">
                        {this.state.images.map(i => {
                            return (
                                <div id="image-wrapper" key={i}>
                                    <img src={i} className="slide-in-fwd-center image hover"></img>
                                    <p className="text" onClick={() => this.props.handleRemoveImage(i, this.state.name)}>x</p>
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
