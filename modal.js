import * as tf from '@tensorflow/tfjs';
import * as ui from './ui';

// Names for the titles of the analysis tools
const ANALYSIS_CORRECTNESS_TITLE = "Label Correctness";
const ANALYSIS_ERROR_TITLE = "Error per Label";
const ANALYSIS_CONFIDENCE_TITLE = "Confidence Graph";

// Names for the classes of the html elements in the modal
const ANALYSIS_TABLE_CLASS = "analysis-table";
const ANALYSIS_TABLE_ROW_CLASS = "analysis-table-row";
const ANALYSIS_TABLE_HEADER_CLASS = "analysis-table-header";
const ANALYSIS_TABLE_CELL_CLASS = "analysis-table-cell";
const ANALYSIS_TABLE_CELL_INNER_CLASS = "analysis-table-cell-inner";
const ANALYSIS_TABLE_CELL_CANVAS_CLASS = "analysis-table-cell-canvas";

const ANALYSIS_TABLE_PREDICTIONS_POPUP_CLASS = "analysis-table-predictions-popup";
const ANALYSIS_TABLE_PREDICTION_CLASS = "analysis-table-prediction";

// Html elements in the modal that we want to refer to
const modal = document.getElementsByClassName("modal")[0];
const modalContent = document.getElementsByClassName("modal-content")[0];
const modalCloseButton = document.getElementsByClassName("modal-close-button")[0];
const modalHeaderText = document.getElementsByClassName("modal-header-text")[0];
const modalBody = document.getElementsByClassName("modal-body")[0];
const modalFooter = document.getElementsByClassName("modal-footer")[0];

const modalCompareClearButton1 = document.getElementById("modal-compare-clear-button-1");
const modalCompareClearButton2 = document.getElementById("modal-compare-clear-button-2");
const modalCompareCanvas1 = document.getElementById("modal-compare-canvas-1");
const modalCompareCanvas2 = document.getElementById("modal-compare-canvas-2");
const modalCompareResults1 = document.getElementById("modal-compare-results-1");
const modalCompareResults2 = document.getElementById("modal-compare-results-2");

const modalCompareElements = {};
modalCompareElements[1] = [modalCompareCanvas1, modalCompareResults1];
modalCompareElements[2] = [modalCompareCanvas2, modalCompareResults2];

let currentModalCompareElement = 1;
let secondModalCompareElementOn = false;

// Maps analysis button names to their corresponding handlers
let toolTitleToContentFunction = {};

export function init() {
	toolTitleToContentFunction[ANALYSIS_CORRECTNESS_TITLE] = setModalContentCorrectness;
	toolTitleToContentFunction[ANALYSIS_ERROR_TITLE] = setModalContentError;
  toolTitleToContentFunction[ANALYSIS_CONFIDENCE_TITLE] = setModalContentConfidence;

  modalCloseButton.addEventListener('click', () => {
    modal.style.display = "none";
    modalCompareClearButton1.click();
    modalCompareClearButton2.click();
  });

  window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
      modalCompareClearButton1.click();
      modalCompareClearButton2.click();
    }
  }

  const analysisToolsButtons = document.getElementsByClassName('analysis-tools-button');
  for (let i = 0; i < analysisToolsButtons.length; i++) {
    analysisToolsButtons[i].addEventListener('click', (event) => {
      const toolButton = event.target;
      const toolButtonTitle = toolButton.firstChild.textContent;

      setModalHeader(toolButtonTitle);
      toolTitleToContentFunction[toolButtonTitle](getResultsHandler());

      modal.style.display = "block";
    });
  }

  modalCompareClearButton1.addEventListener("click", () => {
    clearCompareElements(1);
    currentModalCompareElement = 1;
  });

  modalCompareClearButton2.addEventListener("click", () => {
    clearCompareElements(2);
    secondModalCompareElementOn = false;
    if (currentModalCompareElement > 1) {
      currentModalCompareElement = 2;
    }
  });
}

// This is set in index.js
export let getResultsHandler;
export function setGetResultsHandler(handler) {
  getResultsHandler = handler;
}

// General helper methods for populating the modal
function removeBodyContent() {
  while (modalBody.firstChild) {
    modalBody.removeChild(modalBody.firstChild);
  }
}

function setModalHeader(title) {
  modalHeaderText.innerHTML = title;
}

function createTableCellCanvas(img) {
  const labelInner = document.createElement("div");
  labelInner.setAttribute("class", ANALYSIS_TABLE_CELL_INNER_CLASS);
  const labelCanvas = document.createElement("canvas");
  labelCanvas.setAttribute("class", ANALYSIS_TABLE_CELL_CANVAS_CLASS);
  labelCanvas.setAttribute("width", 224);
  labelCanvas.setAttribute("height", 224);
  labelInner.appendChild(labelCanvas);
  ui.draw(img, labelCanvas);

  return labelInner;
}

// Methods for adding functionality to the compare divs
function clearCompareElements(i) {
  const currentModalCompareElements = modalCompareElements[i];

  const currentCanvas = currentModalCompareElements[0];
  const canvasContext = currentCanvas.getContext('2d');
  canvasContext.clearRect(0, 0, currentCanvas.width, currentCanvas.height);

  const currentResults = currentModalCompareElements[1];
  while (currentResults.firstChild) {
    currentResults.removeChild(currentResults.firstChild);
  }
}

function setCompareEventListeners(cell, result) {
  let mouseoverElement = 0;

  cell.addEventListener("mouseover", () => {
    if (currentModalCompareElement < 3) {
      mouseoverElement = currentModalCompareElement;
      const currentModalCompareElements = modalCompareElements[currentModalCompareElement];

      ui.draw(result.img, currentModalCompareElements[0]);

      for (let i = 0; i < result.predictedLabels.length; i++) {
        const currentLabel = result.predictedLabels[i];
        const currentValue = result.predictedValues[i];

        const resultPredictionSpan = document.createElement("span");
        resultPredictionSpan.innerText = currentLabel + ": " + currentValue.toFixed(5);

        if (currentLabel === result.actualLabel) {
          resultPredictionSpan.setAttribute("class", ANALYSIS_TABLE_PREDICTION_CLASS + " correct");
        } else {
          resultPredictionSpan.setAttribute("class", ANALYSIS_TABLE_PREDICTION_CLASS + " incorrect");
        }

        currentModalCompareElements[1].appendChild(resultPredictionSpan);
      }
    }
  });

  cell.addEventListener("mouseout", () => {
    if (mouseoverElement === currentModalCompareElement) {
      clearCompareElements(currentModalCompareElement);
    }
  });

  cell.addEventListener("click", () => {
    let toIncrement = 1;

    if (currentModalCompareElement == 1) {
      if (secondModalCompareElementOn) {
        toIncrement = 2;
      }
    } else if (currentModalCompareElement == 2) {
      secondModalCompareElementOn = true;
    } else {
      toIncrement = 0;
    }

    currentModalCompareElement += toIncrement;
  });
}

// Methods for setting up the correctness table
function buildCorrectnessTable(labelNamesMapString) {
  const correctnessTableCellOuterClass = "analysis-table-cell-correctness-outer";
  const labelNamesMap = JSON.parse(labelNamesMapString);

  const table = document.createElement("table");
  const headerRow = document.createElement("tr");
  const headerFiller = document.createElement("th");
  const headerCorrect = document.createElement("th");
  const headerIncorrect = document.createElement("th");

  table.setAttribute("class", ANALYSIS_TABLE_CLASS);
  headerRow.setAttribute("class", ANALYSIS_TABLE_ROW_CLASS);
  headerFiller.setAttribute("class", ANALYSIS_TABLE_HEADER_CLASS);
  headerCorrect.setAttribute("class", ANALYSIS_TABLE_HEADER_CLASS);
  headerIncorrect.setAttribute("class", ANALYSIS_TABLE_HEADER_CLASS);

  headerCorrect.textContent = "Labelled Correctly";
  headerIncorrect.textContent = "Labelled Incorrectly";

  headerRow.appendChild(headerFiller);
  headerRow.appendChild(headerCorrect);
  headerRow.appendChild(headerIncorrect);
  table.appendChild(headerRow);

  for (let modelIndex in labelNamesMap) {
    if (labelNamesMap.hasOwnProperty(modelIndex)) {
      const labelName = labelNamesMap[modelIndex];

      const labelRow = document.createElement("tr");
      const labelHeader = document.createElement("th");
      labelHeader.textContent = labelName;

      const labelCorrect = document.createElement("td");
      const labelIncorrect = document.createElement("td");
      const labelCorrectOuter = document.createElement("div");
      const labelIncorrectOuter = document.createElement("div");

      labelRow.setAttribute("class", ANALYSIS_TABLE_ROW_CLASS);
      labelHeader.setAttribute("class", ANALYSIS_TABLE_HEADER_CLASS);
      labelCorrect.setAttribute("class", ANALYSIS_TABLE_CELL_CLASS);
      labelIncorrect.setAttribute("class", ANALYSIS_TABLE_CELL_CLASS);
      labelCorrectOuter.setAttribute("class", correctnessTableCellOuterClass);
      labelCorrectOuter.setAttribute("id", labelName + "-correct");
      labelIncorrectOuter.setAttribute("class", correctnessTableCellOuterClass);
      labelIncorrectOuter.setAttribute("id", labelName + "-incorrect");

      labelCorrect.appendChild(labelCorrectOuter);
      labelIncorrect.appendChild(labelIncorrectOuter);
      labelRow.appendChild(labelHeader);
      labelRow.appendChild(labelCorrect);
      labelRow.appendChild(labelIncorrect);

      table.appendChild(labelRow);
    };
  }

  modalBody.appendChild(table);
}

function populateCorrectnessTable(resultsArray) {
  for (let i = 0; i < resultsArray.length; i++) {
    const currentResult = resultsArray[i];
    const tableCellInner = createTableCellCanvas(currentResult.img);
    setCompareEventListeners(tableCellInner, currentResult);

    const correctnessSuffix = currentResult.predictedLabels[0] === currentResult.actualLabel ? "-correct" : "-incorrect";
    document.getElementById(currentResult.actualLabel + correctnessSuffix).appendChild(tableCellInner);
  }
}

function setModalContentCorrectness(results) {
  removeBodyContent();
  buildCorrectnessTable(results.getLabelNamesMap());
  populateCorrectnessTable(results.getAllResults());
}

// Methods for setting up the error table
function buildErrorTable(labelNamesMapString) {
  const errorTableCellOuterClass = "analysis-table-cell-error-outer";
  const labelNamesMap = JSON.parse(labelNamesMapString);

  const table = document.createElement("table");
  const headerRow = document.createElement("tr");
  const headerFiller = document.createElement("th");

  table.setAttribute("class", ANALYSIS_TABLE_CLASS);
  headerRow.setAttribute("class", ANALYSIS_TABLE_ROW_CLASS);
  headerFiller.setAttribute("class", ANALYSIS_TABLE_HEADER_CLASS);

  headerRow.appendChild(headerFiller);

  const validLabels = [];

  for (let modelIndex in labelNamesMap) {
    if (labelNamesMap.hasOwnProperty(modelIndex)) {
      const labelName = labelNamesMap[modelIndex];
      
      const headerLabel = document.createElement("th");
      headerLabel.setAttribute("class", ANALYSIS_TABLE_HEADER_CLASS);
      headerLabel.textContent = labelName;

      headerRow.appendChild(headerLabel);
      validLabels.push(labelName);
    }
  }

  table.appendChild(headerRow);

  for (let i = 0; i < validLabels.length; i++) {
    const rowLabelName = validLabels[i];

    const labelRow = document.createElement("tr");
    const labelHeader = document.createElement("th");

    labelRow.setAttribute("class", ANALYSIS_TABLE_ROW_CLASS);
    labelHeader.setAttribute("class", ANALYSIS_TABLE_HEADER_CLASS);

    labelHeader.textContent = rowLabelName;
    labelRow.appendChild(labelHeader);

    for (let j = 0; j < validLabels.length; j++) {
      const colLabelName = validLabels[j];

      const labelErrorCell = document.createElement("td");
      const labelErrorCellOuter = document.createElement("div");

      labelErrorCell.setAttribute("class", ANALYSIS_TABLE_CELL_CLASS);
      labelErrorCellOuter.setAttribute("class", errorTableCellOuterClass);
      labelErrorCellOuter.setAttribute("id", rowLabelName + "-" + colLabelName);

      labelErrorCell.appendChild(labelErrorCellOuter);
      labelRow.appendChild(labelErrorCell);
    }

    table.appendChild(labelRow);
  }

  modalBody.appendChild(table);
}

function populateErrorTable(resultsArray) {
  for (let i = 0; i < resultsArray.length; i++) {
    const currentResult = resultsArray[i];

    const actualLabel = currentResult.actualLabel;
    const predictedLabel = currentResult.predictedLabels[0];

    if (actualLabel != predictedLabel) {
      const tableCellInner = createTableCellCanvas(currentResult.img);
      setCompareEventListeners(tableCellInner, currentResult);

      document.getElementById(actualLabel + "-" + predictedLabel).appendChild(tableCellInner);
    }
  }
}

function setModalContentError(results) {
  removeBodyContent();
  buildErrorTable(results.getLabelNamesMap());
  populateErrorTable(results.getAllResults());
}

// Methods for setting up the confidence graph
function setModalContentConfidence(results) {
  modalBody.innerHTML = "CONFIDENCE: " + results.getLabelNamesMap();
}
