// @ts-ignore
import React from 'react';
import './App.css';
import arrow from'./images/arrow.png';

class Label extends React.Component {
    render () {
        return (
            <div className="label-border scale-in-center">
                <div className="label-box">
                    <div className="title-box">
                        <p className="name-p">{this.props.name}</p>
                        <p className="count-p">{this.props.images.length + (this.props.images.length === 1 ? " example" : " examples")}</p>
                    </div>
                    <div className="vertical-line">
                        <img src={arrow} alt="remove label" width={10} style={{margin: "auto", cursor:"pointer"}} onClick={() => this.props.handleRemoveLabel(this.props.name)}></img>
                    </div>
                    <div className="image-box">
                        {this.props.images.map(i => {
                            return (
                                <div id="image-wrapper" key={i}>
                                    <img src={i} alt="remove pic from label" className="image hover"></img>
                                    <p className="text" onClick={() => this.props.handleRemoveImage(i, this.props.name)}>x</p>
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
