# size_measurement


**1. Introduction**
   
This program measures the size of objects detected by an object detector. For a visual overview of the workflow, open the app.pbfsm file using the OpenWorkflow web service (https://cmusatyalab.github.io/OpenWorkflow/) to see the specific step where object size measurement occurs.

**2. Package Installation**

Server side:
pip install opencv-contrib-python
pip install pandas
pip install tensorflow
pip install tensorflow-object-detection-api

**3. How to use**

1. Set Up the Object Detector
Build an object detector model for the target object.

2. Create Dummy Classifier (Temporary Requirement)
Currently, a dummy classifier is also needed as this program extends the TwoStageOWF module. This requirement will be removed in the next update of server.py after the measurement processor is fully integrated with OpenWorkFlow.

3. Build the Workflow
Use the OpenWorkflow service [here](https://cmusatyalab.github.io/OpenWorkflow/) to create a workflow, referencing the provided example app.pbfsm.

4. Organize Files on the Server


Workspace/
├── Size_Measurement/
│   ├── server/
│   │   ├── server.py
│   │   └── measure_object_size.py
├── app.pbfsm
├── object-detector_model/
└── dummy_classifier/

5. Run the Server
python server.py ../../app.pbfsm
6. Connect the Client
Build and run the Android client app to connect to the server and execute the predefined workflow.
