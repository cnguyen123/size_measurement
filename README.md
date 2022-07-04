# size_measurement


**1. Introduction**

This program helps measure the size of an object returned by the object detector. Please see the workflow file (i.e., open the file app.pbfsm with the OWF web service https://cmusatyalab.github.io/OpenWorkflow/) to see the logic of one step where an object is measured its scale.   

**2. Package Installation**

Server side:
- opencv: pip install opencv-contrib-python
- pandas: pip install pandas
- tensorflow: pip install tensorflow
- tensorflow-object-detection: pip install tensorflow-object-detection-api

**3. How to use**

- Build an object detector for the object which requires to estimate the size.
- As the current moment, it also needs a dummy classifier (because it extended the TwoStageOWF). This will be removed next version of server.py after this measurement is added to the OpenWorkFlow.
- Access the OWF service at https://cmusatyalab.github.io/OpenWorkflow/ to build a workflow, as the example attached (i.e., app.pbfsm). 
- Upload all to a server running Tensorflow API:
- The file is ordered in the server as:

-----Workspace

-----------Size_Measurement

---------------server

----------------------server.py

----------------------measure_object_size.py

------------app.pbfsm

------------object-detector_model

------------dummy_classifier

- Run the server as: python server.py ../../app.pbfsm
- Connect the client to server to go through the predefined workflow.

