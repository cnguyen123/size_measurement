import ast
import argparse
import json
import logging
import io
import os
from collections import namedtuple
from multiprocessing import Process, Pipe
import measure_object_size as ms
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from object_detection.utils import label_map_util

from torchvision import transforms
import torch
import torch.nn.functional as nnf
from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2

import credentials
import http_server
import mpncov
import owf_pb2
import wca_state_machine_pb2


SOURCE = 'owf_client'
INPUT_QUEUE_MAXSIZE = 60
PORT = 9099
NUM_TOKENS = 1

DETECTOR_ONES_SIZE = (1, 480, 640, 3)


ALWAYS = 'Always'
HAS_OBJECT_CLASS = 'HasObjectClass'
CLASS_NAME = 'class_name'
TWO_STAGE_PROCESSOR = 'TwoStageProcessor'
DUMMY_PROCESSOR ='DummyProcessor'
CLASSIFIER_PATH = 'classifier_path'
DETECTOR_PATH = 'detector_path'
DETECTOR_CLASS_NAME = 'detector_class_name'
CONF_THRESHOLD = 'conf_threshold'

LABELS_FILENAME = 'classes.txt'
CLASSIFIER_FILENAME = 'model_best.pth.tar'
LABEL_MAP_FILENAME = 'label_map.pbtxt'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR='../../classifier/data/carbody_chassis/faultNegativeResult'

_State = namedtuple('_State', ['always_transition', 'has_class_transitions', 'processors'])
_Classifier = namedtuple('_Classifier', ['model', 'labels'])
_Detector = namedtuple('_Detector', ['detector', 'category_index'])


def _result_wrapper_for_transition(transition):
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)

    logger.info('sending %s', transition.instruction.audio)

    result = gabriel_pb2.ResultWrapper.Result()
    result.payload_type = gabriel_pb2.PayloadType.TEXT
    result.payload = transition.instruction.audio.encode()
    result_wrapper.results.append(result)

    if len(transition.instruction.image) > 0:
        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = transition.instruction.image
        result_wrapper.results.append(result)

    if len(transition.instruction.video) > 0:
        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.VIDEO
        result.payload = transition.instruction.video
        result_wrapper.results.append(result)

    to_client_extras = owf_pb2.ToClientExtras()
    to_client_extras.step = transition.next_state
    to_client_extras.zoom_result = owf_pb2.ToClientExtras.ZoomResult.NO_CALL

    result_wrapper.extras.Pack(to_client_extras)
    return result_wrapper


def _result_wrapper_for(step, zoom_result):
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)
    to_client_extras = owf_pb2.ToClientExtras()
    to_client_extras.step = step
    to_client_extras.zoom_result = zoom_result

    result_wrapper.extras.Pack(to_client_extras)
    return result_wrapper


def _start_zoom():
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)
    to_client_extras = owf_pb2.ToClientExtras()
    to_client_extras.zoom_result = owf_pb2.ToClientExtras.ZoomResult.CALL_START

    zoom_info = owf_pb2.ZoomInfo()
    zoom_info.app_key = credentials.ANDROID_KEY
    zoom_info.app_secret = credentials.ANDROID_SECRET
    zoom_info.meeting_number = credentials.MEETING_NUMBER
    zoom_info.meeting_password = credentials.MEETING_PASSWORD

    to_client_extras.zoom_info.CopyFrom(zoom_info)

    result_wrapper.extras.Pack(to_client_extras)
    return result_wrapper


class _StatesModels:
    def __init__(self, fsm_file_path):
        self._states = {}
        self._classifiers = {}
        self._object_detectors = {}

        self._classifier_representation = {
            'function': mpncov.MPNCOV,
            'iterNum': 5,
            'is_sqrt': True,
            'is_vec': True,
            'input_dim': 2048,
            'dimension_reduction': None,
        }

        pb_fsm = wca_state_machine_pb2.StateMachine()
        with open(fsm_file_path, 'rb') as f:
            pb_fsm.ParseFromString(f.read())

        for state in pb_fsm.states:
            for processor in state.processors:
                self._load_models(processor)

            assert (state.name not in self._states), 'duplicate state name'
            always_transition = None
            has_class_transitions = {}

            for transition in state.transitions:
                assert (len(transition.predicates) == 1), 'bad transition'

                predicate = transition.predicates[0]
                if predicate.callable_name == ALWAYS:
                    always_transition = transition
                    break

                assert predicate.callable_name == HAS_OBJECT_CLASS, (
                    'bad callable')
                callable_args = json.loads(predicate.callable_args)
                class_name = callable_args[CLASS_NAME]

                has_class_transitions[class_name] = transition

            self._states[state.name] = _State(
                always_transition=always_transition,
                has_class_transitions=has_class_transitions,
                processors=state.processors)

        self._start_state = self._states[pb_fsm.start_state]

    def _load_models(self, processor):
        assert processor.callable_name == TWO_STAGE_PROCESSOR or processor.callable_name == DUMMY_PROCESSOR, 'bad processor'

        callable_args = json.loads(processor.callable_args)
    
        if processor.callable_name == TWO_STAGE_PROCESSOR:

            classifier_dir = callable_args[CLASSIFIER_PATH]

            if classifier_dir not in self._classifiers:
        
                #print(os.listdir(classifier_dir))
                print(classifier_dir)
                labels_file = open(os.path.join(classifier_dir, LABELS_FILENAME))
                labels = ast.literal_eval(labels_file.read())

                freezed_layer = 0
                model = mpncov.Newmodel(self._classifier_representation,
                                    len(labels), freezed_layer)
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
                trained_model = torch.load(os.path.join(classifier_dir,
                                                    CLASSIFIER_FILENAME))
                model.load_state_dict(trained_model['state_dict'])
                model.eval()

                self._classifiers[classifier_dir] = _Classifier(
                    model=model, labels=labels)

            detector_dir = callable_args[DETECTOR_PATH]

            if detector_dir not in self._object_detectors:
                detector = tf.saved_model.load(detector_dir)
                ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)
                detector(ones)

                label_map_path = os.path.join(detector_dir, LABEL_MAP_FILENAME)
                label_map = label_map_util.load_labelmap(label_map_path)
                categories = label_map_util.convert_label_map_to_categories(
                    label_map,
                    max_num_classes=label_map_util.get_max_label_map_index(
                        label_map),
                    use_display_name=True)
                category_index = label_map_util.create_category_index(categories)

                self._object_detectors[detector_dir] = _Detector(
                    detector=detector, category_index=category_index)

    def get_classifier(self, path):
        return self._classifiers[path]

    def get_object_detector(self, path):
        return self._object_detectors[path]

    def get_state(self, name):
        return self._states[name]

    def get_start_state(self):
        return self._start_state


class _StatesForExpertCall:
    def __init__(self, transition, states_models):
        self._added_states = set()
        self._state_names = []
        self._transition_to_state = {}

        self._states_models = states_models

        if not os.path.exists(http_server.IMAGES_DIR):
            os.mkdir(http_server.IMAGES_DIR)

        self._add_descendants(transition)

    def _add_descendants(self, transition):
        if transition.next_state in self._added_states:
            return

        self._added_states.add(transition.next_state)
        self._state_names.append(transition.next_state)
        self._transition_to_state[transition.next_state] = transition

        img_filename = os.path.join(
            http_server.IMAGES_DIR, '{}.jpg'.format(transition.next_state))
        with open(img_filename, 'wb') as f:
            f.write(transition.instruction.image)

        next_state = self._states_models.get_state(transition.next_state)
        if next_state.always_transition is not None:
            self._add_descendants(next_state.always_transition)
            return

        for transition in next_state.has_class_transitions.values():
            self._add_descendants(transition)

    def get_state_names(self):
        return self._state_names

    def get_transition(self, name):
        return self._transition_to_state[name]


class InferenceEngine(cognitive_engine.Engine):
   #image_count = 0
  
    def __init__(self, fsm_file_path):
        #print("In INIT INFERENCE ENGINE")
        self.image_count = 0
        self.count_ = 0
        self.error_count = 0
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self._transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            normalize,
        ])
        self._states_models = _StatesModels(fsm_file_path)

        start_state = self._states_models.get_start_state()
        assert start_state.always_transition is not None, 'bad start state'
        self._states_for_expert_call = _StatesForExpertCall(
            start_state.always_transition,
            self._states_models)
        state_names = self._states_for_expert_call.get_state_names()

        http_server_conn, self._engine_conn = Pipe()
        self._http_server_process = Process(
            target=http_server.start_http_server,
            args=(http_server_conn, state_names))
        self._http_server_process.start()

        self._on_zoom_call = False

    def handle(self, input_frame):
    
        to_server_extras = cognitive_engine.unpack_extras(
            owf_pb2.ToServerExtras, input_frame)

        if (to_server_extras.zoom_status ==
                owf_pb2.ToServerExtras.ZoomStatus.STOP):
            msg = {
                'zoom_action': 'stop'
            }
            self._engine_conn.send(msg)
            pipe_output = self._engine_conn.recv()
            new_step = pipe_output.get('step')
            logger.info('Zoom Stopped. New step: %s', new_step)
            transition = self._states_for_expert_call.get_transition(new_step)
            return _result_wrapper_for_transition(transition)

        step = to_server_extras.step
        if step == '':
            state = self._states_models.get_start_state()
        elif (to_server_extras.zoom_status ==
              owf_pb2.ToServerExtras.ZoomStatus.START):
            if self._on_zoom_call:
                return _result_wrapper_for(
                    step, owf_pb2.ToClientExtras.ZoomResult.EXPERT_BUSY)

            msg = {
                'zoom_action': 'start',
                'step': step
            }
            self._engine_conn.send(msg)
            logger.info('Zoom Started')
            return _start_zoom()
        else:
            state = self._states_models.get_state(step)

        if state.always_transition is not None:
            return _result_wrapper_for_transition(state.always_transition)

        if len(state.processors) == 0:
            return _result_wrapper_for(
                step, owf_pb2.ToClientExtras.ZoomResult.NO_CALL)

        assert len(state.processors) == 1, 'wrong number of processors'
        processor = state.processors[0]
        callable_args = json.loads(processor.callable_args)
        detector_dir = callable_args[DETECTOR_PATH]
        detector = self._states_models.get_object_detector(detector_dir)

        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        #07/02/2022
        #img_size_measure = img.copy()
        #
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = detector.detector(np.expand_dims(img, 0))

        scores = detections['detection_scores'][0].numpy()
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)

        im_height, im_width = img.shape[:2]

        classifier_dir = callable_args[CLASSIFIER_PATH]
        classifier = self._states_models.get_classifier(classifier_dir)

        pil_img = Image.open(io.BytesIO(input_frame.payloads[0]))

        conf_threshold = float(callable_args[CONF_THRESHOLD])
        detector_class_name = callable_args[DETECTOR_CLASS_NAME]
        for score, box, class_id in zip(scores, boxes, classes):
            class_name = detector.category_index[class_id]['name']
            if (score < conf_threshold) or (class_name != detector_class_name):
               continue
            logger.debug('found object')
            #print('score is ', score, 'but threshold is ', conf_threshold)
            #print('class_name is ', class_name)
            #print('detector_class_name is ', detector_class_name)




            # from https://github.com/tensorflow/models/blob/39f98e30e7fb51c8b7ad58b0fc65ee5312829deb/research/object_detection/utils/visualization_utils.py#L1232
            ymin, xmin, ymax, xmax = box

            # from https://github.com/tensorflow/models/blob/39f98e30e7fb51c8b7ad58b0fc65ee5312829deb/official/vision/detection/utils/object_detection/visualization_utils.py#L192
            (left, right, top, bottom) = (
                xmin * im_width, xmax * im_width,
                ymin * im_height, ymax * im_height)

           
            #size measurement
            if detector_class_name == "bolt":
                #print("Start to estimate size of the object...")
                (xmin, ymin, xmax, ymax) = (xmin * im_width, ymin * im_height,
                                          xmax * im_width, ymax * im_height)
                img, re1, size_ob = ms.size_measuring2(xmin, ymin, xmax, ymax, img)
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                img_to_send = img
                img_to_send = cv2.cvtColor(img_to_send, cv2.COLOR_BGR2RGB)
                _, jpeg_img = cv2.imencode('.jpg', img_to_send)
                img_data = jpeg_img.tobytes()   
                result = gabriel_pb2.ResultWrapper.Result()
                result.payload_type = gabriel_pb2.PayloadType.IMAGE
                result.payload = img_data
                
                label_name = ""
                if re1 == -2:
                    continue # Do not care this error, since it is a fault positive result
                elif re1 == -1:
                    #print("Something wrong!the marker is not fullly showed...")
                    label_name = "aruco_error"
                    self.count_ += 1
                    #print("Self.count ", self.count_)
                    if self.count_ == 3:
                        self.count_ = 0
                        print("label is", label_name)
                        #transition = state.has_class_transitions.get(label_name)
                    elif self.count_ < 3:
                        continue

                else:

                    size_ob = round(size_ob * 10, 0) #from cm to mm
                    print("Object size is ", size_ob)
                    if size_ob in range(12 - 2, 12 + 7):
                        label_name = "bolt12"
                        self.count_ += 1
                        print("Self.count ", self.count_)
                        
                    else: #different bolt size
                        label_name = "error"
                        self.error_count += 1
                        if self.error_count == 7:
                            self.error_count = 0
                        elif self.error_count < 7:
                            continue
                transition = state.has_class_transitions.get(label_name)
                if transition is None:
                    continue

                return _result_wrapper_for_transition(transition)

            # end size measurement

            cropped_pil = pil_img.crop((left, top, right, bottom))

            transformed = self._transform(cropped_pil).cuda()

            output = classifier.model(transformed[None, ...])
            
            _, pred = output.topk(1, 1, True, True)
            classId = pred.t()
        
            label_name = classifier.labels[classId]
            logger.info('Found label: %s', label_name)
  
            transition = state.has_class_transitions.get(label_name)
           
            if transition is None:
                continue
            
            return _result_wrapper_for_transition(transition)


        return _result_wrapper_for(
            step, owf_pb2.ToClientExtras.ZoomResult.NO_CALL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fsm_file_path', type=str)
    args = parser.parse_args()

    def engine_factory():
        return InferenceEngine(args.fsm_file_path)

    local_engine.run(
        engine_factory, SOURCE, INPUT_QUEUE_MAXSIZE, PORT, NUM_TOKENS)


if __name__ == '__main__':
    main()
