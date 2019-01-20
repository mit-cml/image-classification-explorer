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

// Variables for keeping the state of the image compare divs
const modalCompareElements = {};
modalCompareElements[1] = [modalCompareCanvas1, modalCompareSaliency1, modalCompareResults1];
modalCompareElements[2] = [modalCompareCanvas2, modalCompareSaliency2, modalCompareResults2];

let currentModalCompareElement = 1;
let secondModalCompareElementOn = false;

const currentCompareImgs = [null, null];

// Variables for the confidence graph
const CONFIDENCE_START = 40;
const CONFIDENCE_END = 80;
const CONFIDENCE_INTERVAL = 20;

const confidenceColumnMap = {40: "Medium", 60: "High", 80: "Very High"};

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

  // Adds functionality to the analysis tool buttons on the interface
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

  // Methods for clearing the image compare divs
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

  // Method for getting the saliencies of the images in the image compare divs
  modalSaliencyButton.addEventListener("click", async () => {
    const saliency1 = await getSaliencyHandler(currentCompareImgs[0]);
    const saliency2 = await getSaliencyHandler(currentCompareImgs[1]);

    await tf.toPixels(saliency1, modalCompareSaliency1);
    await tf.toPixels(saliency2, modalCompareSaliency2);
  });
}

// Methods to set in index.js that will allow it to pass data to the modal.
let getResultsHandler;
export function setGetResultsHandler(handler) {
  getResultsHandler = handler;
}

let getSaliencyHandler;
export function setGetSaliencyHandler(handler) {
  getSaliencyHandler = handler;
}

// General helper methods for populating and cleaning up the modal
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

  // Keeps track of and draws the image into the compare cell and write its corresponding results
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

  // If we haven't clicked on the cell, remove the image from the compare div
  cell.addEventListener("mouseout", () => {
    if (mouseoverElement === currentModalCompareElement) {
      clearCompareElements(currentModalCompareElement);
    }
  });

  // If we click on the cell, keep the image and update the state of the compare divs
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

  // First, create the header row for the table
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

  // Then, we can create a row for each label that we have
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
  // For the correctness table, we just iterate through the results and place
  // all images in a cell depending on if they were classified correctly or not
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

  // First, create the header row, with one column per label
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

  // Then, create the rest of the rows, with one row per label
  for (let i = 0; i < validLabels.length; i++) {
    const rowLabelName = validLabels[i];

    const labelRow = document.createElement("tr");
    const labelHeader = document.createElement("th");

    labelRow.setAttribute("class", ANALYSIS_TABLE_ROW_CLASS);
    labelHeader.setAttribute("class", ANALYSIS_TABLE_HEADER_CLASS);

    labelHeader.textContent = rowLabelName;
    labelRow.appendChild(labelHeader);

    // Inner loop since each cell in this table respresents a label-label pair
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
  // For the error table, we iterate through the results and only add images to
  // the table if they were incorrectly classified
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
  // Overarching container for the dropdown and table
  const confidenceGraphContainer = document.createElement("div");
  confidenceGraphContainer.setAttribute("class", "analysis-table-confidence-graph-container");

  // First, create the dropdown for switching between labels
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

  for (let i = CONFIDENCE_START; i <= CONFIDENCE_END; i += CONFIDENCE_INTERVAL) {
    // First, create the row that will actually contain all the images
    const confidenceGraphCell = document.createElement("td");
    const confidenceGraphCellOuter = document.createElement("div");

    confidenceGraphCell.setAttribute("class", ANALYSIS_TABLE_CELL_CLASS);
    confidenceGraphCellOuter.setAttribute("class", confidenceGraphCellOuterClass);
    confidenceGraphCellOuter.setAttribute("id", "confidence-" + i);

    confidenceGraphCell.appendChild(confidenceGraphCellOuter);
    contentRow.appendChild(confidenceGraphCell);

    // Next, create the header row (the x-axis of the graph)
    const confidenceGraphHeader = document.createElement("th");
    confidenceGraphHeader.setAttribute("class", ANALYSIS_TABLE_HEADER_CLASS);
    confidenceGraphHeader.textContent = confidenceColumnMap[i];

    headerRow.appendChild(confidenceGraphHeader);
  }

  table.appendChild(contentRow);
  table.appendChild(headerRow);

  confidenceGraphContainer.appendChild(dropdownContainer);
  confidenceGraphContainer.appendChild(table);

  modalBody.appendChild(confidenceGraphContainer);

  // Finally, add functionality to the dropdown once all the elements are in place
  // and initialize the graph with the first label
  dropdown.addEventListener("change", (event) => {
    populateConfidenceGraph(event.target.value);
  });

  populateConfidenceGraph(dropdown.value);
}

function populateConfidenceGraphHelper(resultsArray) {
  // Unlike for the correctness and error tables, this method actually returns another
  // method that has access to all of the methods/data within the scope of this one.
  // The returned method is the one actually responsible for filling the graph and is
  // intended to be called whenever a label is selected from the dropdown.

  // Calculates the correct bucket to place the image in. Values are hard coded for now.
  function calculateBucket(value) {
    if (value >= 0.8) {
      return 80;
    } else if (value >= 0.6) {
      return 60;
    } else if (value >= 0.4) {
      return 40;
    } else {
      return 0;
    }
  }

  // Clears the graph when labels are being switched.
  function clearBuckets() {
    for (let i = CONFIDENCE_START; i <= CONFIDENCE_END; i += CONFIDENCE_INTERVAL) {
      const currentBucketCell = document.getElementById("confidence-" + i);

      while (currentBucketCell.firstChild) {
        currentBucketCell.removeChild(currentBucketCell.firstChild);
      }
    }
  }

  // Given a label, fills the confidence graph with images that were classified as
  // that label. Images are placed in buckets based on classification confidence.
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
