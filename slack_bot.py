import requests, logging, re
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from config import config

logging.basicConfig(level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

QA_ENDPOINT = config.QA_ENDPOINT

app = App(token=config.SLACK_BOT_TOKEN)

# Log every incoming envelope/body
@app.middleware
def log_request(logger, body, next):
    logger.debug(f"[INCOMING] {body}")
    return next()

# Catch app mentions (what we want)
@app.event("app_mention")
def on_app_mention(event, say):
    logger.info(f"app_mention event: {event}")
    user = event.get("user")
    text = event.get("text","")
    # strip <@U...> mentions
    q = " ".join(t for t in text.split() if not t.startswith("<@")).strip()
    msg = say(f"Got it <@{user}> — searching…")
    try:
        r = requests.post(QA_ENDPOINT, json={"question": q}, timeout=60)
        r.raise_for_status()
        answer = r.json().get("answer", "No answer.")
        app.client.chat_update(channel=event["channel"], ts=msg["ts"], text=answer)
    except Exception as e:
        logger.exception("QA call failed")
        say(f"Error: `{e}`")

# Broad net: log ANY message events (helps confirm delivery)
@app.event("message")
def on_message_events(event, logger):
    logger.info(f"[message event seen] subtype={event.get('subtype')} text={event.get('text')}")

# Even broader: regex listener (won’t fire in threads without extra opts)
@app.message(re.compile(".*"))
def on_any_message(message, logger):
    logger.info(f"[message listener] {message.get('text')}")

if __name__ == "__main__":
    logger.info("Starting Slack bot (Socket Mode)…")
    SocketModeHandler(app, config.SLACK_APP_TOKEN).start()
