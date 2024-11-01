# size_measurement

### 1. Introduction

This program measures the size of objects detected by an object detector. For a visual overview of the workflow, open the `app.pbfsm` file using the [OpenWorkflow web service](https://cmusatyalab.github.io/OpenWorkflow/) to see the specific step where object size measurement occurs.

### 2. Package Installation

To set up the server, install the following packages:

```bash
pip install opencv-contrib-python
pip install pandas
pip install tensorflow
pip install tensorflow-object-detection-api

### 3. How to Use

1. **Set Up the Object Detector**  
   Build an object detector model for the target object.

2. **Create Dummy Classifier (Temporary Requirement)**  
   A dummy classifier is also required as this program extends the `TwoStageOWF` module. This requirement will be removed in the next update of `server.py` once the measurement processor is fully integrated with OpenWorkFlow.

3. **Build the Workflow**  
   Use the OpenWorkflow service [here](https://cmusatyalab.github.io/OpenWorkflow/) to create a workflow, referencing the provided example `app.pbfsm`.

4. **Organize Files on the Server**  
   Arrange files on the server as follows:

   ```plaintext
   Workspace/
   ├── Size_Measurement/
   │   ├── server/
   │   │   ├── server.py
   │   │   └── measure_object_size.py
   ├── app.pbfsm
   ├── object-detector_model/
   └── dummy_classifier/
