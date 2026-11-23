import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
import pytest
from google.genai import types

from data3800_gruppe_7.assistant import Assistant, build_system_prompt, build_tool
from data3800_gruppe_7.cleaning import FEATURES
from data3800_gruppe_7.modeling import MODELS_DIR, PREDICTIONS_PATH, TYPICAL_CRASH_PATH
from data3800_gruppe_7.risk import MODEL_NAME, CrashScorer

needs_outputs = pytest.mark.skipif(
    not (
        (MODELS_DIR / f"{MODEL_NAME}.joblib").exists()
        and PREDICTIONS_PATH.exists()
        and TYPICAL_CRASH_PATH.exists()
        and Path("output/evaluation/metrics.csv").exists()
    ),
    reason="model and evaluation output missing, run modeling and evaluation first",
)


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


@pytest.fixture(scope="module")
def scorer():
    return CrashScorer()


@needs_outputs
def test_tool_covers_every_feature(scorer):
    # The tool description covers every feature of the model
    declaration = build_tool(scorer).function_declarations[0]
    properties = declaration.parameters_json_schema["properties"]
    assert set(properties) == set(FEATURES)

    # The language model cannot send UNKNOWN for a detail the caller does not know
    assert "CLEAR" in properties["WEATHER_CONDITION"]["enum"]
    for description in properties.values():
        assert "UNKNOWN" not in description.get("enum", [])


@needs_outputs
def test_described_crash_is_scored_first(scorer):
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


@needs_outputs
def test_unscorable_value_goes_back_as_error(scorer):
    # A value the model cannot score goes back to the language model as an error
    client = ScriptedClient(
        [tool_request({"WEATHER_CONDITION": "SUNNY"}), final_answer("asked the user")]
    )
    assistant = Assistant(client, "test-model", scorer)
    assert assistant.answer("It was sunny.") == "asked the user"
    assert "error" in client.sent[1][0].function_response.response


@needs_outputs
def test_results_question_needs_no_tool(scorer):
    # A question about the results needs no tool call
    client = ScriptedClient([final_answer("logistic regression")])
    assistant = Assistant(client, "test-model", scorer)
    assert assistant.answer("Which model is best?") == "logistic regression"
    assert len(client.sent) == 1


@needs_outputs
def test_chat_goes_on_after_api_error(scorer):
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


def test_missing_result_file_stops_assistant():
    # A missing result file stops the assistant with a message that says what to run
    missing = ["output/evaluation/no_such_table.csv"]
    with (
        patch("data3800_gruppe_7.assistant.RESULT_FILES", missing),
        pytest.raises(SystemExit, match="Run the evaluation first"),
    ):
        build_system_prompt()
