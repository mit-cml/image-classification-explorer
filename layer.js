// TODO: 
// store Layer instances sequentially in list 
  // must be updated upon: Add Layer, Remove Layer 
    // delete 1 thing at idx: layerList.splice(idx,1);
    // insert item at idx: layerList.splice(idx, 0, item);
// checkValidModel function that checks model validity (both layers and parameters)
  // call onchange/onclick all the time 
// updateLayerDims function that computes and updates layer parameters starting at Layer 

// Checks if n is an integer 
function isInt(n) {
  return n % 1 == 0;
}

// TODO: create Layer subclass for first layer (different add method)
// TODO: create Layer subclass for final layer 

/**
 * A class that represents a layer node. 
 */
export class LayerNode {
  /**
   * @param {String} i The layer id suffix (i.e. "0", "1", "2", ..., "final")
   */
  constructor(i) {
    console.log("INSIDE LAYERNODE CONSTRUCTOR!!");
    console.log(i == "final");
    this.id = i; 
    if (i=="final") {
      this.id_val = -1;
    } else{
      this.id_val = Number(i);
    }
    

    this.previous = null; 
    this.next = null; 

    this.isFirst = (i == "0");
    this.isFinal = (i == "final"); 

    console.log("Creating Layer");
    console.log(i);
    console.log((i == "0"));
    console.log((i == "final"));

    this.inputDims = null; 
    this.outputDims = null; 

    // call function to create layer/parameter input(s) and associated div elements  
    this.createLayer();
  }

  /**
   * Creates layer select input, layer parameter inputs, and associated div elements 
   */
  createLayer() {
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
      dropdown_values = ["conv", "flat"];
    } else if (this.isFinal) {
      dropdown_text = ["Fully Connected"];
      dropdown_values = ["fc"];
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
    
    inputWrapper.appendChild(input);
    
    if (!this.isFirst || !this.isFinal) {
      const removeButton = document.createElement('button');
      removeButton.innerHTML = 'Remove Layer';
      removeButton.onclick = () => { 
        // TODO: remove from linkedLayerList ? 
        modelWrapper.removeChild(inputWrapper)
      }
    }
  
    // add layer input options 
    this.addFc(inputWrapper, this.id);
    this.addConv(inputWrapper, this.id);
    this.addMaxPool(inputWrapper, this.id);
    
    // if (!this.isFirst || !this.isFinal) {
    //   inputWrapper.appendChild(removeButton);
    // }
    try {
      inputWrapper.appendChild(removeButton);
    } catch (e) {
      console.log("could not append remove button!");
    }

    modelWrapper.appendChild(inputWrapper);
  
    // add span for layer input/output display
    var layerDimsDisplay = document.createElement('span')
    layerDimsDisplay.id = `dimensions-${this.id}`;
    layerDimsDisplay.innerHTML = "[input] --> [output]";
    inputWrapper.appendChild(layerDimsDisplay);
    
    // display fully connected inputs only 
    document.getElementById(`fcn-units-${this.id}`).style.display = "inline"; 
    document.getElementById(`conv-kernel-size-${this.id}`).style.display = "none"; 
    document.getElementById(`conv-filters-${this.id}`).style.display = "none"; 
    document.getElementById(`conv-strides-${this.id}`).style.display = "none"; 
    document.getElementById(`max-pool-size-${this.id}`).style.display = "none"; 
    document.getElementById(`max-strides-${this.id}`).style.display = "none"; 
    
    // this.updateDimensions();
    if (!this.isFirst) {
      this.updateDimensions();
    }
    input.onchange = this.layerSelectCheck(this.id);
  }

  updateDimensions(){
    // get all select id's inside model-editor 
    let modelLayers = document.querySelectorAll("#model-editor select");
    let dimList = [[7,7,256]]; 

    for (let j = 0; j < modelLayers.length; j++) {
      let layerValue = document.getElementById(modelLayers[j].id).value;

      console.log("Layer!");
      console.log(layerValue);
      
      let idx = Number(modelLayers[j].id.substr(-1));

      // get layer parameters and set parameters 
      if (layerValue == "fc") {
        // if input is not a 1D tensor, raise error 
        if (dimList[dimList.length-1].length != 1) {
          document.getElementById("model-error").innerHTML = "Invalid Model! Must have flatten before fully connected.";
          throw new Error("Invalid Model! Must have flatten before fully connected.");
        }
        let fcnUnits = Number(document.getElementById(`fcn-units-${idx}`).value);
        
        // compute and push output dimensions 
        let nextDims = [];
        nextDims.push(fcnUnits);
        dimList.push(nextDims);
      } else if (layerValue == "maxpool") {
        // if input is not a 3D image, raise error 
        if (dimList[dimList.length-1].length != 3) {
          document.getElementById("model-error").innerHTML = "Invalid Model! Cannot have max pool after flatten.";
          throw new Error("Invalid Model! Cannot have max pool after flatten.");
        }
        let maxPoolSize = Number(document.getElementById(`max-pool-size-${idx}`).value);
        let maxStrides = Number(document.getElementById(`max-strides-${idx}`).value);

        // compute and push output dimensions 
        let lastDims = dimList[dimList.length-1];
        let nextDims = [];
        nextDims.push((lastDims[0]-maxPoolSize)/maxStrides+1);
        nextDims.push((lastDims[1]-maxPoolSize)/maxStrides+1);
        nextDims.push(lastDims[2]);
        dimList.push(nextDims);
      } else if (layerValue == "conv" || layerValue == "conv-0") {
        // if input is not a 3D image, raise error 
        if (dimList[dimList.length-1].length != 3) {
          document.getElementById("model-error").innerHTML = "Invalid Model! Cannot have convolution after flatten.";
          throw new Error("Invalid Model! Cannot have convolution after flatten.");
        }
        let convKernelSize = Number(document.getElementById(`conv-kernel-size-${idx}`).value);
        let convFilters = Number(document.getElementById(`conv-filters-${idx}`).value); 
        let convStrides = Number(document.getElementById(`conv-strides-${idx}`).value);
        
        // compute and push output dimensions 
        let lastDims = dimList[dimList.length-1];
        let nextDims = [];
        nextDims.push((lastDims[0]-convKernelSize)/convStrides+1);
        nextDims.push((lastDims[1]-convKernelSize)/convStrides+1);
        nextDims.push(convFilters);
        dimList.push(nextDims);
      } else if (layerValue == "fc-final") {
        // if input is not a 1D tensor, raise error 
        if (dimList[dimList.length-1].length != 1) {
          document.getElementById("model-error").innerHTML = "Invalid Model! Must have flatten before fully connected.";
          throw new Error("Invalid Model! Must have flatten before fully connected.");
        }
      } else {
        // if input is not a 3D tensor, raise error 
        if (dimList[dimList.length-1].length != 3) {
          document.getElementById("model-error").innerHTML = "Invalid Model! Cannot have multiple flatten layers.";
          throw new Error("Invalid Model! Cannot have multiple flatten layers.");
        }

        // compute and push output dimensions 
        let lastDims = dimList[dimList.length-1];
        let nextDims = [];
        nextDims.push(lastDims[0]*lastDims[1]*lastDims[2]);
        dimList.push(nextDims);
      }

      // if (i != modelLayers.length-1) {
      if (!this.isFinal) {
        document.getElementById(`dimensions-${idx}`).innerHTML = dimList[dimList.length-2] + " --> " + dimList[dimList.length-1];
      } else {
        document.getElementById("dimensions-final").innerHTML = dimList[dimList.length-1] + " --> " + ["Number of Labels"];
      }
      
      // if it's not the final layer and we're on the matching layer, update the input and output dimensions correspondingly 
      if (!this.isFinal && idx == Number(this.id)) {
        this.inputDims = dimList[dimList.length-2];
        this.outputDims = dimList[dimList.length-1];
      }
    }; 

    dimList.push(["Number of Labels"]);

    console.log("DIMENSIONS LIST: ");
    console.log(dimList);

    // check for invalid dimensions 
    for (let k=0; k<dimList.length-1; k++) {
      let dim = dimList[k];
      for (let j=0; j<dim.length; j++) {
        let d = dim[j];
        if (d < 0 || !isInt(d)){
          document.getElementById("dim-error").innerHTML = "Invalid Dimensions! Fix layer parameters.";
          throw new Error("Invalid Dimensions! Fix layer parameters.");
        }
      }
    }
  }

  // Methods for adding user input for layer parameters 
  addFc(inputWrapper, i) {
    const unit_input = document.createElement("input");
    unit_input.type = "number"; 
    unit_input.id = `fcn-units-${i}`;
    unit_input.onchange = this.layerSelectCheck(i);
    unit_input.min = 1;
    unit_input.max = 300;
    unit_input.step = 1;
    unit_input.value = 100;
    inputWrapper.appendChild(unit_input); 
  }

  addConv(inputWrapper, i) {
    const kernel_input = document.createElement("input");
    const filter_input = document.createElement("input");
    const stride_input = document.createElement("input");
    kernel_input.type = filter_input.type = stride_input.type = "number"; 
    kernel_input.id = `conv-kernel-size-${i}`;
    filter_input.id = `conv-filters-${i}`;
    stride_input.id = `conv-strides-${i}`;
    kernel_input.onchange = this.layerSelectCheck(i);
    filter_input.onchange = this.layerSelectCheck(i);
    stride_input.onchange = this.layerSelectCheck(i);
    kernel_input.min = filter_input.min = stride_input.min = 1;
    kernel_input.max = filter_input.max = stride_input.max = 100;
    kernel_input.step = filter_input.step = stride_input.step = 1;
    kernel_input.value = filter_input.value = stride_input.value = 5;
    inputWrapper.appendChild(kernel_input);
    inputWrapper.appendChild(filter_input);
    inputWrapper.appendChild(stride_input);
  }

  addMaxPool(inputWrapper, i) {
    const pool_input = document.createElement("input");
    const stride_input = document.createElement("input");
    pool_input.type = stride_input.type = "number"; 
    pool_input.id = `max-pool-size-${i}`;
    stride_input.id = `max-strides-${i}`;
    pool_input.onchange = this.layerSelectCheck(i);
    stride_input.onchange = this.layerSelectCheck(i);
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
      console.log("Select ID");
      console.log(`select-${i}`);
      let selectedLayer = document.getElementById(`select-${i}`).value;
      console.log("Selected layer");
      console.log(selectedLayer);

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
      self.updateDimensions();
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
   * Removes LayerNode from the doubly linked layer list 
   * 
   * @param {LayerNode} remove The layer to remove 
   */
  removeLayer(remove) {
    oldPrev = remove.previous; 
    oldNext = remove.next; 

    oldPrev.next = oldNext;
    oldNext.previous = oldPrev;

    // TODO: implement removal method that removes layer node entirely 
    remove.previous = remove; 
    remove.next = remove; 
  }

  /**
   * Adds layer node before the tail 
   * 
   * @param {LayerNode} add The layer to add 
   */
  addLayer(add) {
    tailPrev = this.tail.previous; 

    add.next = this.tail;
    add.previous = tailPrev; 

    tailPrev.next = add; 
    this.tail.previous = add;
  }
}
