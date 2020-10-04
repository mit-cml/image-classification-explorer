import React from 'react';
import LabelView from './LabelView.js';
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
