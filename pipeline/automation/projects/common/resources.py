import os

from dagster_slack import SlackResource

SlackResource(token=os.getenv("SLACK_API_TOKEN"))

common_resources = {
    "slack": SlackResource,
}
