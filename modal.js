import * as tf from '@tensorflow/tfjs';
import * as ui from './ui';

const ANALYSIS_CORRECTNESS_TITLE = "Label Correctness";
const ANALYSIS_ERROR_TITLE = "Error per Label";
const ANALYSIS_CONFIDENCE_TITLE = "Confidence Graph";

const ANALYSIS_TABLE_CLASS = "analysis-table";
const ANALYSIS_TABLE_ROW_CLASS = "analysis-table-row";
const ANALYSIS_TABLE_HEADER_CLASS = "analysis-table-header";
const ANALYSIS_TABLE_CELL_CLASS = "analysis-table-cell";
const ANALYSIS_TABLE_CELL_INNER_CLASS = "analysis-table-cell-inner";
const ANALYSIS_TABLE_CELL_CANVAS_CLASS = "analysis-table-cell-canvas";

const ANALYSIS_TABLE_PREDICTIONS_POPUP_CLASS = "analysis-table-predictions-popup";
const ANALYSIS_TABLE_PREDICTION_CLASS = "analysis-table-prediction";

const modal = document.getElementsByClassName("modal")[0];
const modalContent = document.getElementsByClassName("modal-content")[0];
const modalCloseButton = document.getElementsByClassName("modal-close-button")[0];
const modalHeaderText = document.getElementsByClassName("modal-header-text")[0];
const modalBody = document.getElementsByClassName("modal-body")[0];

// Maps button names to correct content function
let toolTitleToContentFunction = {};

export function init() {
	toolTitleToContentFunction[ANALYSIS_CORRECTNESS_TITLE] = setModalContentCorrectness;
	toolTitleToContentFunction[ANALYSIS_ERROR_TITLE] = setModalContentError;
  toolTitleToContentFunction[ANALYSIS_CONFIDENCE_TITLE] = setModalContentConfidence;

  modalCloseButton.addEventListener('click', () => {
    modal.style.display = "none";
  });

  window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
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
}

export let getResultsHandler;
export function setGetResultsHandler(handler) {
  getResultsHandler = handler;
}

function removeBodyContent() {
  while (modalBody.firstChild) {
    modalBody.removeChild(modalBody.firstChild);
  }
}

function setModalHeader(title) {
  modalHeaderText.innerHTML = title;
}

// General helper methods for populating the modal

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

function createPredictionsPopup(result) {
  const predictionsPopupDiv = document.createElement("div");
  predictionsPopupDiv.setAttribute("class", ANALYSIS_TABLE_PREDICTIONS_POPUP_CLASS);

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

    predictionsPopupDiv.appendChild(resultPredictionSpan);
  }

  return predictionsPopupDiv;
}

// Methods for the correctness table

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
    const predictionsPopup = createPredictionsPopup(currentResult);
    tableCellInner.appendChild(predictionsPopup);

    const correctnessSuffix = currentResult.predictedLabels[0] === currentResult.actualLabel ? "-correct" : "-incorrect";
    document.getElementById(currentResult.actualLabel + correctnessSuffix).appendChild(tableCellInner);
  }
}

function setModalContentCorrectness(results) {
  removeBodyContent();
  buildCorrectnessTable(results.getLabelNamesMap());
  populateCorrectnessTable(results.getAllResults());
}

// Methods for the error table

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
      const predictionsPopup = createPredictionsPopup(currentResult);
      tableCellInner.appendChild(predictionsPopup);

      document.getElementById(actualLabel + "-" + predictedLabel).appendChild(tableCellInner);
    }
  }
}

function setModalContentError(results) {
  removeBodyContent();
  buildErrorTable(results.getLabelNamesMap());
  populateErrorTable(results.getAllResults());
}

// Methods for the confidence graph

function setModalContentConfidence(results) {
  modalBody.innerHTML = "CONFIDENCE: " + results.getLabelNamesMap();
}
