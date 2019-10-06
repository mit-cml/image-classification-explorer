// @ts-ignore
import React from 'react';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Popover from 'react-bootstrap/Popover'
import './App.css';
import Label from './Label.js';
import Audio from './Audio.js';
import plus from'./images/plus.png';

class LabelView extends React.Component {
    constructor(props) {
        super(props)
        this.handleNewImage = this.handleNewImage.bind(this)
        this.handleRemoveImage = this.handleRemoveImage.bind(this)
        this.handleLabelKeyPress = this.handleLabelKeyPress.bind(this)

        this.state = { 
            imageMap: {

  
            }
        }
    }

    handleNewImage(image, currentLabel) {
        this.setState({
            imageMap: {
                ...this.state.imageMap,
                [currentLabel]: [...this.state.imageMap[currentLabel], image]
            }
        })
    }

    handleRemoveImage(i, name) {
        console.log(name)
        console.log(this.state.imageMap)
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

    handleLabelKeyPress(e) {
        console.log(e.keyCode)
        if(e.keyCode == 13) {
            console.log(e.target.value)
        }
    }

    render () {
        return (
            <div className="view-all">
                <div className="all-labels">
                    {Object.keys(this.state.imageMap).map(k => {
                        return (
                            <Label 
                                name={k}
                                images={this.state.imageMap[k]}
                                handleRemoveImage={this.handleRemoveImage}
                                key={k}/>
                        )
                    })}
                    <div className="plus-wrapper">
                    <OverlayTrigger
                        trigger="click"
                        placement={"left"}
                        overlay={
                            <Popover>
                                <Popover.Title as="h3">{"Create New Label"}</Popover.Title>
                                <Popover.Content>
                                <input type="text" name="name" onKeyPress={this.handleLabelKeyPress}/>
                                </Popover.Content>
                            </Popover>
                        }
                        >
                        <img src={plus} className="plus"></img>
                    </OverlayTrigger>
                    </div>
                </div>
                <Audio 
                    handleNewImage={this.handleNewImage}
                    allLabels={Object.keys(this.state.imageMap)}/>
            </div>
        )
    }
}

export default LabelView;
