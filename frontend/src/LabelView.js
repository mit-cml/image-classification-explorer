// @ts-ignore
import React from 'react';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Popover from 'react-bootstrap/Popover'
import './App.css';
import Label from './Label.js';
import Audio from './Audio.js';
import Button from 'react-bootstrap/Button';

import plus from'./images/plus.png';
import { Link } from 'react-router-dom';


class LabelView extends React.Component {
    constructor(props) {
        super(props)
        this.handleNewImage = this.handleNewImage.bind(this)
        this.handleRemoveImage = this.handleRemoveImage.bind(this)
        this.handleLabelKeyDown = this.handleLabelKeyDown.bind(this)
        this.createNewLabel = this.createNewLabel.bind(this)

        this.state = { 
            imageMap: {
            "Hello": [],
            "Bye": []
            }
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

    render () {
        console.log(this.state.imageMap)
        return (
            <div className="view-all">
                <Audio 
                    handleNewImage={this.handleNewImage}
                    allLabels={Object.keys(this.state.imageMap)}/>
                <div className="all-labels">
                    <div className="plus-wrapper">
                        <OverlayTrigger
                            trigger="click"
                            placement={"left"}
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
                <Link 
                    to={{
                        pathname: "/test",
                        state: {imageMap: this.state.imageMap}}}
                >
                    <Button variant="dark">Train</Button>
                </Link>
            </div>
        )
    }
}

export default LabelView;
