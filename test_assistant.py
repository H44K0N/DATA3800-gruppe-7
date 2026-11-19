import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
from google.genai import types

from data3800_gruppe_7.assistant import Assistant, build_system_prompt, build_tool
from data3800_gruppe_7.cleaning import FEATURES
from data3800_gruppe_7.risk import CrashScorer


class ScriptedClient:
    """Stands in for the Gemini API and gives back prepared replies, one per message."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.sent = []
        # The assistant asks client.chats.create(...) for a conversation.
        self.chats = SimpleNamespace(create=self.start_conversation)

    def start_conversation(self, **settings):
        self.settings = settings
        return self

    def send_message(self, message):
        self.sent.append(message)
        reply = next(self.replies)
        # A prepared error stands for a request the API failed on.
        if isinstance(reply, Exception):
            raise reply
        return reply


def tool_request(arguments):
    """Return a reply in which the language model asks for the tool."""
    call = types.FunctionCall(id="call-1", name="score_crash", args=arguments)
    return SimpleNamespace(function_calls=[call], text=None)


def final_answer(text):
    return SimpleNamespace(function_calls=None, text=text)


scorer = CrashScorer()

# The tool description covers every feature of the model
declaration = build_tool(scorer).function_declarations[0]
properties = declaration.parameters_json_schema["properties"]
assert set(properties) == set(FEATURES)

# The language model cannot send UNKNOWN for a detail the caller does not know
assert "CLEAR" in properties["WEATHER_CONDITION"]["enum"]
for description in properties.values():
    assert "UNKNOWN" not in description.get("enum", [])

# A described crash is scored by our model before the language model answers
client = ScriptedClient(
    [tool_request({"FIRST_CRASH_TYPE": "PEDESTRIAN"}), final_answer("explained")]
)
assistant = Assistant(client, "test-model", scorer)
assert assistant.answer("A car hit a pedestrian.") == "explained"
tool_result = client.sent[1][0].function_response
assert tool_result.id == "call-1"
assert tool_result.response["risk_score"] > 0.9

# The result files are sent with the instructions
instructions = client.settings["config"].system_instruction
assert "=== output/evaluation/metrics.csv ===" in instructions

# A value the model cannot score goes back to the language model as an error
client = ScriptedClient(
    [tool_request({"WEATHER_CONDITION": "SUNNY"}), final_answer("asked the user")]
)
assistant = Assistant(client, "test-model", scorer)
assert assistant.answer("It was sunny.") == "asked the user"
assert "error" in client.sent[1][0].function_response.response

# A question about the results needs no tool call
client = ScriptedClient([final_answer("logistic regression")])
assistant = Assistant(client, "test-model", scorer)
assert assistant.answer("Which model is best?") == "logistic regression"
assert len(client.sent) == 1

# The chat goes on after a question the API failed on
client = ScriptedClient(
    [httpx2.ConnectError("no connection"), final_answer("answered")]
)
assistant = Assistant(client, "test-model", scorer)
typed = iter(["first question", "second question", ""])
printed = io.StringIO()
with patch("builtins.input", lambda prompt: next(typed)), redirect_stdout(printed):
    assistant.chat()
assert client.sent == ["first question", "second question"]
assert "No answer to that question: no connection" in printed.getvalue()
assert "answered" in printed.getvalue()

# A missing result file stops the assistant with a message that says what to run
missing = ["output/evaluation/no_such_table.csv"]
with patch("data3800_gruppe_7.assistant.RESULT_FILES", missing):
    try:
        build_system_prompt()
    except SystemExit as stop:
        assert "Run the evaluation first" in str(stop)
    else:
        raise AssertionError("A missing result file should stop the assistant")

print("ALL TESTS PASSED!")
