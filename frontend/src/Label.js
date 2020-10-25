// @ts-ignore
import React from 'react';
import './App.css';
import arrow from'./images/arrow.png';

class Label extends React.Component {
    constructor(props){
        super(props)
        this.handleDragOver = this.handleDragOver.bind(this);
        this.handleDragExit = this.handleDragExit.bind(this);
        this.handleDragLeave = this.handleDragLeave.bind(this);
        this.handleDrop = this.handleDrop.bind(this);
        this.state = {
            names: "label-border scale-in-center"
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
        this.setState({names: "label-border scale-in-center droppable"});
    }
    handleDragExit(e) {
        this.setState({names: "label-border scale-in-center"});
    }
    handleDragLeave(e) {
        this.setState({names: "label-border scale-in-center"});
    }
    handleDrop(e) {
        for (let i = 0; i < e.dataTransfer.files.length; i++) {
            let file = e.dataTransfer.files[i];
            const name = file.name;
            let reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                this.props.handleNewImage(reader.result, this.props.name);
            };
            reader.onerror = () => {
                console.log("Failed to load" + name);
            } 
          }
          this.setState({names: "label-border scale-in-center"});
          e.preventDefault();
    }

    render () {
        return (
            <div className={this.state.names} onDragOver={this.handleDragOver} onDragExit={this.handleDragExit} onDragLeave={this.handleDragLeave} onDrop={this.handleDrop}>
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
