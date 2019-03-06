// TODO: 
// DONE: store Layer instances sequentially in list 
  // must be updated upon: Add Layer, Remove Layer 
// checkValidModel function that checks model validity (both layers and parameters) 
  // call onchange/onclick all the time 
// updateLayerDims function that computes and updates layer parameters starting at Layer 

// Checks if n is an integer 
function isInt(n) {
  return n % 1 == 0;
}

/**
 * A class that represents a layer node. 
 */
export class LayerNode {
  /**
   * @param {String} i The layer id suffix (i.e. "0", "1", "2", ..., "final")
   * @param {LayerList} linkedLayerList Reference to the linked layer list. 
   * @param {Boolean} updateDims true if dimensions should be updated, false otherwise 
   * @param {String} layerType String indicating which layer type to initialize to 
   */
  constructor(i, linkedLayerList, updateDims, layerType = "fc") {
    this.id = i; 
    if (i=="final") {
      this.id_val = -1;
    } else {
      this.id_val = Number(i);
    }

    this.layerList = linkedLayerList; 

    this.previous = null; 
    this.next = null; 

    this.isFirst = (i == "0");
    this.isFinal = (i == "final"); 

    if (i == "0") {
      this.inputDims = [7, 7, 256];
    } else {
      this.inputDims = null; 
    }
    
    if (i == "final") {
      this.outputDims = "Number of Labels"; 
    } else {
      this.outputDims = null; 
    }

    // this.removeButton = null;

    // call function to create layer/parameter input(s) and associated div elements  
    this.createLayer(updateDims, layerType);
  }

  /**
   * Creates layer select input, layer parameter inputs, and associated div elements 
   * 
   * @param {Boolean} updateDims true if dimensions should be updated, false otherwise 
   * @param {String} layerType String indicating which layer type to initialize to 
   */
  createLayer(updateDims, layerType = "fc") {
    let modelWrapper;
    if (this.isFinal) {
      modelWrapper = document.getElementById("model-pt2");
    } else {
      modelWrapper = document.getElementById("model-pt1");
    }
    
    const inputWrapper = document.createElement('div');
    inputWrapper.id = `inputWrapper-${this.id}` ;

    let dropdown_text;
    let dropdown_values;
    if (this.isFirst) {
      dropdown_text = ["Convolution", "Flatten"];
      dropdown_values = ["conv-0", "flat-0"];
    } else if (this.isFinal) {
      dropdown_text = ["Fully Connected"];
      dropdown_values = ["fc-final"];
    } else {
      dropdown_text = ["Fully Connected", "Convolution", "Max Pool", "Flatten"];
      dropdown_values = ["fc", "conv", "maxpool", "flat"];
    }

    const input = document.createElement('select');
    input.id = `select-${this.id}` ;
    
    // create and append options 
    for (let j = 0; j < dropdown_text.length; j++) {
      let option = document.createElement("option");
      option.value = dropdown_values[j];
      option.text = dropdown_text[j];
      input.appendChild(option); 
    }

    // initialize to default value 
    if (layerType == "fc" | layerType == "conv" | layerType == "maxpool" | layerType == "flat" | layerType == "conv-0" | layerType == "flat-0" | layerType == "fc-final") {
      input.value = layerType;
    } 
    
    inputWrapper.appendChild(input);
    
    let removeButton;
    if (!this.isFirst && !this.isFinal) {
      removeButton = document.createElement('button');
      removeButton.innerHTML = 'Remove Layer';
      let that = this;
      removeButton.onclick = () => { 
        // removing from linked layer list 
        that.layerList.removeLayer(that); 

        modelWrapper.removeChild(inputWrapper);
      }
    }
  
    // add layer input options 
    this.addFc(inputWrapper, this.id);
    this.addConv(inputWrapper, this.id);
    this.addMaxPool(inputWrapper, this.id);
    
    if (!this.isFirst && !this.isFinal) {
      inputWrapper.appendChild(removeButton);
    }

    modelWrapper.appendChild(inputWrapper);
  
    // add span for layer input/output display
    let layerDimsDisplay = document.createElement('span');
    layerDimsDisplay.id = `dimensions-${this.id}`;
    layerDimsDisplay.innerHTML = "[input] --> [output]";
    inputWrapper.appendChild(layerDimsDisplay);
    
    // display fully connected inputs only 
    if (this.isFinal) {
      document.getElementById(`fcn-units-${this.id}`).style.display = "none"; 
      document.getElementById(`conv-kernel-size-${this.id}`).style.display = "none"; 
      document.getElementById(`conv-filters-${this.id}`).style.display = "none"; 
      document.getElementById(`conv-strides-${this.id}`).style.display = "none"; 
      document.getElementById(`max-pool-size-${this.id}`).style.display = "none"; 
      document.getElementById(`max-strides-${this.id}`).style.display = "none"; 
    } else if (!this.isFirst) {
      document.getElementById(`fcn-units-${this.id}`).style.display = "inline"; 
      document.getElementById(`conv-kernel-size-${this.id}`).style.display = "none"; 
      document.getElementById(`conv-filters-${this.id}`).style.display = "none"; 
      document.getElementById(`conv-strides-${this.id}`).style.display = "none"; 
      document.getElementById(`max-pool-size-${this.id}`).style.display = "none"; 
      document.getElementById(`max-strides-${this.id}`).style.display = "none"; 
    } else {
      document.getElementById(`fcn-units-${this.id}`).style.display = "none"; 
      document.getElementById(`conv-kernel-size-${this.id}`).style.display = "inline"; 
      document.getElementById(`conv-filters-${this.id}`).style.display = "inline"; 
      document.getElementById(`conv-strides-${this.id}`).style.display = "inline"; 
      document.getElementById(`max-pool-size-${this.id}`).style.display = "none"; 
      document.getElementById(`max-strides-${this.id}`).style.display = "none"; 
    }

    
    let self = this; 
    let id_ref = self.id;
    input.onchange = self.layerSelectCheck(id_ref);
  }

  // Methods for adding user input for layer parameters 
  addFc(inputWrapper, i) {
    let self = this; 

    const unit_input = document.createElement("input");
    unit_input.type = "number"; 
    unit_input.id = `fcn-units-${i}`;
    unit_input.onchange = self.layerSelectCheck(self.id);
    unit_input.min = 1;
    unit_input.max = 300;
    unit_input.step = 1;
    unit_input.value = 100;
    inputWrapper.appendChild(unit_input); 
  }

  addConv(inputWrapper, i) {
    let self = this; 

    const kernel_input = document.createElement("input");
    const filter_input = document.createElement("input");
    const stride_input = document.createElement("input");
    kernel_input.type = filter_input.type = stride_input.type = "number"; 
    kernel_input.id = `conv-kernel-size-${i}`;
    filter_input.id = `conv-filters-${i}`;
    stride_input.id = `conv-strides-${i}`;
    kernel_input.onchange = self.layerSelectCheck(self.id);
    filter_input.onchange = self.layerSelectCheck(self.id);
    stride_input.onchange = self.layerSelectCheck(self.id);
    kernel_input.min = filter_input.min = stride_input.min = 1;
    kernel_input.max = filter_input.max = stride_input.max = 100;
    kernel_input.step = filter_input.step = stride_input.step = 1;
    kernel_input.value = filter_input.value = 5; 
    stride_input.value = 1;
    inputWrapper.appendChild(kernel_input);
    inputWrapper.appendChild(filter_input);
    inputWrapper.appendChild(stride_input);
  }

  addMaxPool(inputWrapper, i) {
    let self = this; 

    const pool_input = document.createElement("input");
    const stride_input = document.createElement("input");
    pool_input.type = stride_input.type = "number"; 
    pool_input.id = `max-pool-size-${i}`;
    stride_input.id = `max-strides-${i}`;
    pool_input.onchange = self.layerSelectCheck(self.id);
    stride_input.onchange = self.layerSelectCheck(self.id);
    pool_input.min = stride_input.min = 1;
    pool_input.max = stride_input.max = 20;
    pool_input.step = stride_input.step = 1;
    pool_input.value = stride_input.value = 5; 
    inputWrapper.appendChild(pool_input);
    inputWrapper.appendChild(stride_input);
  }

  // Checks selected layer and displays corresponding input boxes accordingly 
  layerSelectCheck(i) {
    let self = this;

    return function() {
      let selectedLayer = document.getElementById(`select-${i}`).value;

      if (selectedLayer == "fc") { 
        document.getElementById(`fcn-units-${i}`).style.display = "inline"; 
        document.getElementById(`conv-kernel-size-${i}`).style.display = "none"; 
        document.getElementById(`conv-filters-${i}`).style.display = "none"; 
        document.getElementById(`conv-strides-${i}`).style.display = "none"; 
        document.getElementById(`max-pool-size-${i}`).style.display = "none"; 
        document.getElementById(`max-strides-${i}`).style.display = "none"; 
      } else if (selectedLayer == "conv") {
        document.getElementById(`fcn-units-${i}`).style.display = "none"; 
        document.getElementById(`conv-kernel-size-${i}`).style.display = "inline"; 
        document.getElementById(`conv-filters-${i}`).style.display = "inline"; 
        document.getElementById(`conv-strides-${i}`).style.display = "inline"; 
        document.getElementById(`max-pool-size-${i}`).style.display = "none"; 
        document.getElementById(`max-strides-${i}`).style.display = "none"; 
      } else if (selectedLayer == "maxpool") {
        document.getElementById(`fcn-units-${i}`).style.display = "none"; 
        document.getElementById(`conv-kernel-size-${i}`).style.display = "none"; 
        document.getElementById(`conv-filters-${i}`).style.display = "none"; 
        document.getElementById(`conv-strides-${i}`).style.display = "none"; 
        document.getElementById(`max-pool-size-${i}`).style.display = "inline"; 
        document.getElementById(`max-strides-${i}`).style.display = "inline"; 
      } else if (selectedLayer == "conv-0") {
        document.getElementById("conv-kernel-size-0").style.display = "inline";
        document.getElementById("conv-filters-0").style.display = "inline";
        document.getElementById("conv-strides-0").style.display = "inline";
      } else if (selectedLayer == "flat-0") {
        document.getElementById("conv-kernel-size-0").style.display = "none";
        document.getElementById("conv-filters-0").style.display = "none";
        document.getElementById("conv-strides-0").style.display = "none";
      } else {
        document.getElementById(`fcn-units-${i}`).style.display = "none"; 
        document.getElementById(`conv-kernel-size-${i}`).style.display = "none"; 
        document.getElementById(`conv-filters-${i}`).style.display = "none"; 
        document.getElementById(`conv-strides-${i}`).style.display = "none"; 
        document.getElementById(`max-pool-size-${i}`).style.display = "none"; 
        document.getElementById(`max-strides-${i}`).style.display = "none"; 
      }

      // update dimensions 
      document.getElementById("model-error").innerHTML = "";
      document.getElementById("dim-error").innerHTML = "";
      // this.updateDimensions();
      self.layerList.updateDimensions(self);
    }
  }
}


/**
 * A class that represents a collection of layers in our editable model. 
 * Implemented like a doubly linked list. 
 */
export class LayerList {
  /**
   * @param {LayerNode} head The first layer 
   * @param {LayerNode} tail The last layer 
   */
  constructor(head, tail) {
    // Note that for the purposes of this application, we always have a head and tail 
    this.head = head; 
    this.tail = tail; 

    head.next = tail;
    tail.previous = head; 
  }

  /**
   * Adds head and tail in case they were never initialized when instantiated. 
   * List must be empty. 
   * 
   * @param {LayerNode} head The first layer 
   * @param {LayerNode} tail The last layer 
   */
  addHeadTail(head, tail) {
    this.head = head; 
    this.tail = tail; 

    head.next = tail;
    tail.previous = head; 
  }

  /**
   * Removes LayerNode from the doubly linked layer list 
   * 
   * @param {LayerNode} remove The layer to remove 
   */
  removeLayer(remove) {
    let oldPrev = remove.previous; 
    let oldNext = remove.next; 

    oldPrev.next = oldNext;
    oldNext.previous = oldPrev;

    // TODO: implement removal method that removes layer node entirely 
    remove.previous = remove; 
    remove.next = remove; 

    // TODO: update layer parameters 
    this.updateDimensions(oldPrev);
  }

  /**
   * Adds layer node before the tail 
   * 
   * @param {LayerNode} add The layer to add 
   * @param {Boolean} updateDims true if dimensions should be updated, false otherwise 
   */
  addLayer(add, updateDims=true) {
    let tailPrev = this.tail.previous; 

    add.next = this.tail;
    add.previous = tailPrev; 

    tailPrev.next = add; 
    this.tail.previous = add;

    // TODO: update layer parameters 
    if (updateDims) {
      this.updateDimensions(add);
    }
  }

  /**
   * Checks if dimensions are valid (positive integer).
   * 
   * @param {Array} dimensions 
   */
  checkIfValidDimension(dimensions) {
    if (dimensions != "Number of Labels") {
      for (let j=0; j<dimensions.length; j++) {
        let d = dimensions[j];
        if (d < 0 || !isInt(d)){
          document.getElementById("dim-error").innerHTML = "Invalid Dimensions! Fix layer parameters.";
          console.error("Invalid Dimensions! Fix layer parameters.");
        }
      }
    }
  }

  /**
   * Computes and updates output dimension for layer. 
   * 
   * @param {LayerNode} currentLayer The layer we want to compute the output dimension for. 
   * @param {Array} previousDimension Input dimension. 
   */
  computeDimension(currentLayer, previousDimension) {
    // this is layer the user has selected from dropdown 
    let layerValue = document.getElementById(`select-${currentLayer.id}`).value; 
    let dimension;

    // get layer parameters and set parameters 
    if (layerValue == "fc") {
      // if input is not a 1D tensor, raise error 
      if (previousDimension.length != 1) {
        document.getElementById("model-error").innerHTML = "Invalid Model! Must have flatten before fully connected.";
        console.error("Invalid Model! Must have flatten before fully connected.");
      }
  
      let fcnUnits = Number(document.getElementById(`fcn-units-${currentLayer.id}`).value);
  
      // compute and update output dimensions, check if valid 
      dimension = [fcnUnits];
      currentLayer.outputDims = dimension;
  
    } else if (layerValue == "maxpool") {
      // if input is not a 3D image, raise error 
      if (previousDimension.length != 3) {
        document.getElementById("model-error").innerHTML = "Invalid Model! Cannot have max pool after flatten.";
        console.error("Invalid Model! Cannot have max pool after flatten.");
      }
      let maxPoolSize = Number(document.getElementById(`max-pool-size-${currentLayer.id}`).value);
      let maxStrides = Number(document.getElementById(`max-strides-${currentLayer.id}`).value);
  
      // compute and update output dimensions, check if valid 
      let width = (previousDimension[0]-maxPoolSize)/maxStrides+1;
      let height = (previousDimension[1]-maxPoolSize)/maxStrides+1; 
      let depth = previousDimension[2];
      dimension = [width, height, depth];
      currentLayer.outputDims = dimension;
  
    } else if (layerValue == "conv" || layerValue == "conv-0") {
      // if input is not a 3D image, raise error 
      if (previousDimension.length != 3) {
        document.getElementById("model-error").innerHTML = "Invalid Model! Cannot have convolution after flatten.";
        console.error("Invalid Model! Cannot have convolution after flatten.");
      }
      let convKernelSize = Number(document.getElementById(`conv-kernel-size-${currentLayer.id}`).value);
      let convFilters = Number(document.getElementById(`conv-filters-${currentLayer.id}`).value); 
      let convStrides = Number(document.getElementById(`conv-strides-${currentLayer.id}`).value);
      
      // compute and update output dimensions, check if valid 
      let width = (previousDimension[0]-convKernelSize)/convStrides+1;
      let height = (previousDimension[1]-convKernelSize)/convStrides+1;
      let depth = convFilters;
      dimension = [width, height, depth];
      currentLayer.outputDims = dimension;
  
    } else if (layerValue == "fc-final") {
      dimension = "Number of Labels";

      // if input is not a 1D tensor, raise error 
      if (previousDimension.length != 1) {
        document.getElementById("model-error").innerHTML = "Invalid Model! Must have flatten before fully connected.";
        console.error("Invalid Model! Must have flatten before fully connected.");
      }
    } else {
      // if input is not a 3D tensor, raise error 
      if (previousDimension.length != 3) {
        document.getElementById("model-error").innerHTML = "Invalid Model! Cannot have multiple flatten layers.";
        console.error("Invalid Model! Cannot have multiple flatten layers.");
      }
  
      // compute and update output dimensions, check if valid 
      let flattenedDims = previousDimension[0] * previousDimension[1] * previousDimension[2]; 
      dimension = [flattenedDims];
      currentLayer.outputDims = dimension;
  
    }

    // check for invalid dimensions 
    this.checkIfValidDimension(dimension); 
    return dimension;
  }
  
  /**
   * Updates input/output dimension attributes starting at startingLayer. 
   * 
   * @param {LayerNode} startingLayer The layer to start updating from. 
   */
  updateDimensions(startingLayer) {
    let previousLayer = startingLayer.previous;
    let previousDimension; 
    if (previousLayer == null) {
      previousDimension = [7, 7, 256];
    } else {
      previousDimension = previousLayer.outputDims;
    }
    startingLayer.inputDims = previousDimension;

    // while we haven't reached the end, compute and update dimension attributes 
    let currentLayer = startingLayer;
    let nextDimension; 
    while (currentLayer != null) {
      currentLayer.inputDims = previousDimension;

      // compute next dimension 
      nextDimension = this.computeDimension(currentLayer, previousDimension);
      
      // update previousDimension 
      previousDimension = nextDimension;

      // update layer 
      currentLayer = currentLayer.next;
    }

    // update displays 
    this.updateDimensionDisplay(startingLayer);
  }

  /**
   * Updates input/output dimension display starting at startingLayer. 
   * Uses input and output dimension attributes for each layer. 
   * 
   * @param {LayerNode} startingLayer The layer to start updating from. 
   */
  updateDimensionDisplay(startingLayer) {
    let currentLayer = startingLayer;

    // while we haven't reached the end, update display 
    while (currentLayer != null) {
      let layerDimsDisplay = document.getElementById(`dimensions-${currentLayer.id}`);
      layerDimsDisplay.innerHTML = currentLayer.inputDims + " --> " + currentLayer.outputDims;

      currentLayer = currentLayer.next;
    }
  }
}