size_measurement
1. Introduction
This program measures the size of objects detected by an object detector. For a visual overview of the workflow, open the app.pbfsm file using the OpenWorkflow web service to see the specific step where object size measurement occurs.

2. Package Installation
To set up the server, install the following packages:

bash
Copy code
pip install opencv-contrib-python
pip install pandas
pip install tensorflow
pip install tensorflow-object-detection-api
3. How to Use
Set Up the Object Detector
Build an object detector model for the target object.

Create Dummy Classifier (Temporary Requirement)
A dummy classifier is also required as this program extends the TwoStageOWF module. This requirement will be removed in the next update of server.py once the measurement processor is fully integrated with OpenWorkFlow.

Build the Workflow
Use the OpenWorkflow service here to create a workflow, referencing the provided example app.pbfsm.

Organize Files on the Server
Arrange files on the server as follows:

plaintext
Copy code
Workspace/
├── Size_Measurement/
│   ├── server/
│   │   ├── server.py
│   │   └── measure_object_size.py
├── app.pbfsm
├── object-detector_model/
└── dummy_classifier/
Run the Server
Start the server with the command:

bash
Copy code
python server.py ../../app.pbfsm
Connect the Client
Build and run the Android client app to connect to the server and execute the predefined workflow.
