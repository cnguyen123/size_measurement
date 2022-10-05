import json
import requests
import hmac
import hashlib
import credentials
import logging
import get_zoom_recordings

logger = logging.getLogger(__name__)

API_SERVER = "https://api.zoom.us/v2"
ACCESS_TOKEN_SUFFIX = "?access_token=" + credentials.JWT_TOKEN
ZOOM_CLIENT_USER_NAME = "Gabriel User"

in_meeting_control_api = "/live_meetings/" + credentials.MEETING_NUMBER + "/events"
in_meeting_control_url = API_SERVER + in_meeting_control_api + ACCESS_TOKEN_SUFFIX
start_record_payload = {'method': "recording.start"}
stop_record_payload = {'method': "recording.stop"}

gabriel_user_list = []


# Referring to https://marketplace.zoom.us/docs/api-reference/webhook-reference/#verify-webhook-events
def verify(headers, body):
    msg = ":".join(["v0", headers.get('x-zm-request-timestamp'), body])
    signature = hmac.new(bytes(credentials.SECRET_TOKEN, 'utf-8'), msg=bytes(msg, 'utf-8'),
                         digestmod=hashlib.sha256).hexdigest()
    return "v0=" + signature == headers.get('x-zm-signature')


def handle(headers, body):
    body_text = body.decode('utf-8')
    if verify(headers, body_text):
        logger.info("Request received from Zoom webhook")
        content = json.loads(body_text)
        if content.get('event') == "recording.completed":
            get_zoom_recordings.main()

        elif (content.get('event') == "meeting.participant_joined" and
                content.get('payload').get('object').get('participant').get('user_name') == ZOOM_CLIENT_USER_NAME):
            if len(gabriel_user_list) == 0:
                response = requests.patch(in_meeting_control_url, json=start_record_payload)
                if not response:
                    logger.warning("Starting cloud recording failed. " + str(response))
                else:
                    logger.info("Cloud recording started.")
            gabriel_user_list.append(content.get('payload').get('object').get('participant').get('user_id'))

        elif (content.get('event') == "meeting.participant_left" and
              content.get('payload').get('object').get('participant').get('user_name') == ZOOM_CLIENT_USER_NAME):
            gabriel_user_list.remove(content.get('payload').get('object').get('participant').get('user_id'))
            if len(gabriel_user_list) == 0:
                response = requests.patch(in_meeting_control_url, json=stop_record_payload)
                if not response:
                    logger.warning("Stopping cloud recording failed. " + str(response))
                else:
                    logger.info("Cloud recording stopped.")
