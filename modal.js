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
const modalCompareSaliency1 = document.getElementById("modal-compare-saliency-1");
const modalCompareSaliency2 = document.getElementById("modal-compare-saliency-2");
const modalCompareResults1 = document.getElementById("modal-compare-results-1");
const modalCompareResults2 = document.getElementById("modal-compare-results-2");

const modalSaliencyBox = document.getElementsByClassName("modal-compare-saliency-outer")[0];
const modalSaliencyButton = document.getElementsByClassName("modal-compare-saliency-button")[0];

const modalCompareElements = {};
modalCompareElements[1] = [modalCompareCanvas1, modalCompareSaliency1, modalCompareResults1];
modalCompareElements[2] = [modalCompareCanvas2, modalCompareSaliency2, modalCompareResults2];

let currentModalCompareElement = 1;
let secondModalCompareElementOn = false;

const currentCompareImgs = [null, null];

// Constants for the confidence graph
const CONFIDENCE_START = 40;
const CONFIDENCE_INTERVAL = 20;

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

    modalSaliencyBox.style.display = 'none';
  });

  modalCompareClearButton2.addEventListener("click", () => {
    clearCompareElements(2);
    secondModalCompareElementOn = false;
    if (currentModalCompareElement > 1) {
      currentModalCompareElement = 2;
    }

    modalSaliencyBox.style.display = 'none';
  });

  modalSaliencyButton.addEventListener("click", async () => {
    const saliency1 = await getSaliencyHandler(currentCompareImgs[0]);
    const saliency2 = await getSaliencyHandler(currentCompareImgs[1]);

    await tf.toPixels(saliency1, modalCompareSaliency1);
    await tf.toPixels(saliency2, modalCompareSaliency2);
  });
}

// This is set in index.js
let getResultsHandler;
export function setGetResultsHandler(handler) {
  getResultsHandler = handler;
}

let getSaliencyHandler;
export function setGetSaliencyHandler(handler) {
  getSaliencyHandler = handler;
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

  const currentSaliency = currentModalCompareElements[1];
  const saliencyContext = currentSaliency.getContext('2d');
  saliencyContext.clearRect(0, 0, currentSaliency.width, currentSaliency.height);

  const currentResults = currentModalCompareElements[2];
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
      currentCompareImgs[currentModalCompareElement - 1] = result.img;

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

        currentModalCompareElements[2].appendChild(resultPredictionSpan);
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

    if (currentModalCompareElement > 2) {
      modalSaliencyBox.style.display = '';
    }

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
function buildConfidenceGraph(labelNamesMapString, populateConfidenceGraph) {

  // First, create the dropdown
  const labelNamesMap = JSON.parse(labelNamesMapString);

  const dropdownContainer = document.createElement("div");
  const dropdownTitleSpan = document.createElement("span");
  const dropdown = document.createElement("select");

  dropdownContainer.setAttribute("class", "analysis-table-confidence-dropdown-outer");
  dropdownTitleSpan.setAttribute("class", "analysis-table-confidence-dropdown-title");
  dropdown.setAttribute("class", "analysis-table-confidence-dropdown");

  dropdownTitleSpan.textContent = "Select a label to view: ";
  dropdownContainer.appendChild(dropdownTitleSpan);
  dropdownContainer.appendChild(dropdown);

  for (let modelIndex in labelNamesMap) {
    if (labelNamesMap.hasOwnProperty(modelIndex)) {
      const labelName = labelNamesMap[modelIndex];

      const dropdownOption = document.createElement("option");
      dropdownOption.text = labelName;
      dropdownOption.value = labelName;

      dropdown.options.add(dropdownOption);
    };
  }

  // Next, create the table
  const confidenceGraphCellOuterClass = "analysis-table-cell-confidence-outer";

  const table = document.createElement("table");
  const contentRow = document.createElement("tr");
  const headerRow = document.createElement("tr");

  table.setAttribute("class", ANALYSIS_TABLE_CLASS);
  contentRow.setAttribute("class", ANALYSIS_TABLE_ROW_CLASS);
  contentRow.setAttribute("id", ANALYSIS_TABLE_ROW_CLASS + "-confidence-content");

  for (let i = CONFIDENCE_START; i <= 100; i += CONFIDENCE_INTERVAL) {
    // First, fill contentRow
    const confidenceGraphCell = document.createElement("td");
    const confidenceGraphCellOuter = document.createElement("div");

    confidenceGraphCell.setAttribute("class", ANALYSIS_TABLE_CELL_CLASS);
    confidenceGraphCellOuter.setAttribute("class", confidenceGraphCellOuterClass);
    confidenceGraphCellOuter.setAttribute("id", "confidence-" + i);

    confidenceGraphCell.appendChild(confidenceGraphCellOuter);
    contentRow.appendChild(confidenceGraphCell);

    // Next, fill headerRow
    const confidenceGraphHeader = document.createElement("th");
    confidenceGraphHeader.setAttribute("class", ANALYSIS_TABLE_HEADER_CLASS);
    confidenceGraphHeader.textContent = i + "%";

    headerRow.appendChild(confidenceGraphHeader);
  }

  table.appendChild(contentRow);
  table.appendChild(headerRow);

  modalBody.appendChild(dropdownContainer);
  modalBody.appendChild(table);

  // Finally, add functionality to the dropdown once all the elements are in place
  dropdown.addEventListener("change", (event) => {
    populateConfidenceGraph(event.target.value);
  });

  populateConfidenceGraph(dropdown.value);
}

function populateConfidenceGraphHelper(resultsArray) {

  // hard coded values for now
  function calculateBucket(value) {
    if (value >= 0.9) {
      return 100;
    } else if (value >= 0.7) {
      return 80;
    } else if (value >= 0.5) {
      return 60;
    } else if (value >= 0.4) {
      return 40;
    } else {
      return 0;
    }
  }

  function clearBuckets() {
    for (let i = CONFIDENCE_START; i <= 100; i += CONFIDENCE_INTERVAL) {
      const currentBucketCell = document.getElementById("confidence-" + i);

      while (currentBucketCell.firstChild) {
        currentBucketCell.removeChild(currentBucketCell.firstChild);
      }
    }
  }

  return function(label) {
    clearBuckets();

    for (let i = 0; i < resultsArray.length; i++) {
      const currentResult = resultsArray[i];
      const labelIndex = currentResult.predictedLabels.indexOf(label);

      if (labelIndex > -1) {
        const predictedValue = currentResult.predictedValues[labelIndex];
        const valueBucket = calculateBucket(predictedValue);

        if (valueBucket > 0) {
          const tableCellInner = createTableCellCanvas(currentResult.img);
          setCompareEventListeners(tableCellInner, currentResult);

          document.getElementById("confidence-" + valueBucket).appendChild(tableCellInner);
        }
      }
    }
  }
}

function setModalContentConfidence(results) {
  removeBodyContent();
  const populateConfidenceGraph = populateConfidenceGraphHelper(results.getAllResults());
  buildConfidenceGraph(results.getLabelNamesMap(), populateConfidenceGraph);
}
