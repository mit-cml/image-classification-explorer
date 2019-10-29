import React from 'react';
import logo from './logo.svg';
import './App.css';
import Label from './Label.js';
import LabelView from './LabelView.js';
import Loading from './Loading.js';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import { BrowserRouter as Router, Route } from 'react-router-dom';
import TestView from './TestView';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <Navbar bg="dark" variant="dark">
          <Navbar.Brand href="#home">Spectrogram Audio Classifier</Navbar.Brand>
          <Nav className="mr-auto">
            <Nav.Link>Train</Nav.Link>
            <Nav.Link>Test</Nav.Link>
            <Nav.Link>Export</Nav.Link>
          </Nav>
        </Navbar>

        <Router>
          <Route exact path="/" component={LabelView} />
          <Route path="/test" component={TestView} />
          <Route path="/loading" component={Loading} />
        </Router>
        
        <div></div>
      </header>
    </div>
  );
}

export default App;
