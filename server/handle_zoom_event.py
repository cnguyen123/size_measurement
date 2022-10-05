import json
import requests
import credentials
import logging

logger = logging.getLogger(__name__)

API_SERVER = "https://api.zoom.us/v2"
ACCESS_TOKEN_SUFFIX = "?access_token=" + credentials.JWT_TOKEN
ZOOM_CLIENT_USER_NAME = "Gabriel User"

in_meeting_control_api = "/live_meetings/" + credentials.MEETING_NUMBER + "/events"
in_meeting_control_url = API_SERVER + in_meeting_control_api + ACCESS_TOKEN_SUFFIX
payload = {'method': "recording.start"}


# Referring to https://marketplace.zoom.us/docs/api-reference/webhook-reference/#verify-webhook-events
def verify(headers, obj):
    return True


def handle(headers, body):
    logger.info("Request received from Zoom webhook")
    logger.info(str(headers['x-zm-request-timestamp']))
    logger.info(str(headers['x-zm-signature']))
    obj = json.loads(body.decode('UTF8'))
    logger.info(str(obj))
    if verify(headers, obj):
        pass


if __name__ == "__main__":
    response = requests.patch(in_meeting_control_url, json=payload)
    print(response)
