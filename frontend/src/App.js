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
        <Router>
          <Route exact path="/" component={LabelView} />
          <Route path="/test" component={TestView} />
        </Router>
    </div>
  );
}

export default App;
