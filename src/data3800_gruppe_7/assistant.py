"""Explain our results and score described crashes with a Gemini model.

Run from the project root, after the modeling and evaluation modules:

    uv run python -m data3800_gruppe_7.assistant

Type a question or describe a crash, and press Enter on an empty line to stop.
One question can also be given directly:

    uv run python -m data3800_gruppe_7.assistant "Which model is best?"

The assistant sends the conversation and our result files to a Gemini model
through Google's API. When the user describes a crash, the language model asks
for the score_crash tool. The tool runs our trained model (see risk.py) and
returns the numbers, and the language model explains them. It needs an API key
from Google AI Studio in a file called .env in the project root:

    GEMINI_API_KEY=your-key

The free tier allows a limited number of requests a day for each model. To use
another model, add GEMINI_MODEL=<model id> to .env.
"""

import json
import os
import sys

import httpx2
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from data3800_gruppe_7.modeling import PROJECT_ROOT
from data3800_gruppe_7.risk import MODEL_NAME, NUMBER_LIMITS, CrashScorer

KEY_VARIABLE = "GEMINI_API_KEY"
MODEL_VARIABLE = "GEMINI_MODEL"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Google turns a request away when the model is busy or we have sent too many
# requests in a minute. The library only tries again when asked to. The waits
# between the attempts double each time and add up to about 15 seconds.
REQUEST_ATTEMPTS = 5
# A request that gets no answer is given up after this many seconds.
REQUEST_TIMEOUT_SECONDS = 30

# How long the language model thinks before it answers. With less thinking it
# guessed details the user never gave.
THINKING_LEVEL = "high"

# What can go wrong when we ask the Gemini API: an error answer or no connection.
# The library sends its requests with httpx2, so a lost connection is an httpx2 error.
API_ERRORS = (errors.APIError, httpx2.HTTPError)

# The files the assistant may answer from, relative to the project root.
RESULT_FILES = [
    "output/cleaning_summary.txt",
    "output/evaluation/metrics.csv",
    "output/evaluation/metric_intervals.csv",
    "output/evaluation/permutation_importance.csv",
    "output/evaluation/odds_ratios.csv",
]

TOOL_NAME = "score_crash"

# The language model may ask for the tool again, for example after a rejected
# value. This limit stops a model that never gets to an answer.
MAX_TOOL_ROUNDS = 5

# What each feature means, for the language model that fills in the tool call.
FEATURE_DESCRIPTIONS = {
    "POSTED_SPEED_LIMIT": "Posted speed limit in mph.",
    "WEATHER_CONDITION": "Weather at the time of the crash.",
    "LIGHTING_CONDITION": "Light at the time of the crash.",
    "ROADWAY_SURFACE_COND": "Condition of the road surface.",
    "CRASH_HOUR": "Hour of the day, from 0 to 23.",
    "TRAFFIC_CONTROL_DEVICE": "Traffic control at the place, such as traffic lights.",
    "TRAFFICWAY_TYPE": "Type of road or place, such as a one-way street.",
    "FIRST_CRASH_TYPE": "What the vehicle hit first, or how it crashed.",
    "NUM_UNITS": "Number of vehicles, pedestrians and cyclists involved.",
}

SYSTEM_PROMPT = """\
You are the assistant of a student project that predicts whether a
police-reported traffic crash in Chicago in 2025 led to an injury. The
project's result files follow these instructions, each under a line with its
file name.

How the results were made:
- The models were fitted on the crashes from January to August, and their
  settings were chosen on September and October. Every number in
  output/evaluation/ comes from the crashes in November and December, the
  test set, which the models never saw.
- metric_intervals.csv holds 95% bootstrap intervals from resamples of the
  test set. A row named "a - b" is the difference between two models on the
  same resamples. The two models differ when that interval excludes 0.
- permutation_importance.csv shows how much the PR-AUC of a model falls when
  one feature is shuffled.
- odds_ratios.csv comes from the logistic regression. Its crashes column
  counts the test crashes in each category. For POSTED_SPEED_LIMIT and
  NUM_UNITS the odds ratio is for one standard deviation.
- The group prefers the logistic regression, because it scores the same as
  the random forest and is easier to interpret. The score_crash tool uses it.

When the user asks about the project:
- Answer only from these notes and the result files. Do not fill gaps with
  outside knowledge, but you may explain a term such as recall or odds ratio
  in plain words.
- Say only what the numbers show. Before you say that something holds for
  several models, check the rows of each model.
- After a sentence with numbers from a file, give the file in brackets, for
  example [output/evaluation/metrics.csv].
- If the files cannot answer the question, say so, and say what data would be
  needed to answer it.

When the user describes a crash:
- Always call the score_crash tool first, with the details the user gave, and
  leave out the rest. The tool fills in typical values itself. Call it again
  when the user changes a detail. Never estimate the risk of a crash yourself.
- If a category is not allowed, use the allowed one that clearly fits, such
  as CLEAR for sunny. Never change a number. Leave out a number outside the
  limits, and tell the user the limits.
- Explain the result in everyday words, without the names of features and
  fields. Say how large a share of the test crashes with a similar risk score
  had an injury, compared with all test crashes, and which details matter
  most. Those test crashes can be of any kind, so do not call them crashes of
  the same kind. what_if holds the risk score the crash would get if one
  detail had its typical value instead. The detail raises the score when
  raises_risk_score_by is above 0.
- The risk score is not a probability. Use the injury share of the test
  crashes with a similar score to say how likely an injury is.
- Say which typical values the tool assumed.
- Give no file name for numbers from the tool. They come from our model.

Always:
- The results are associations in police-reported crashes. Never describe a
  detail as the cause of an injury.
- If the question is about something else than this project or a crash, say
  that you only answer about this project.
- Answer in the same language as the question, in at most six sentences."""


def main():
    # An answer can hold characters that Windows cannot write to a file by default.
    sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:])
    assistant = make_assistant()
    # shortcut: an error from the API ends the whole chat, catch it per question
    # when the chat has to survive a busy model
    try:
        if question:
            print(assistant.answer(question))
        else:
            assistant.chat()
    except API_ERRORS as error:
        sys.exit(
            f"Gemini did not answer: {error}\n"
            f"The free tier allows a limited number of requests a day for each "
            f"model. Try again later, or set {MODEL_VARIABLE} in .env to another model."
        )


def make_assistant():
    """Return an assistant with a new conversation, set up from .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    client = make_client(os.getenv(KEY_VARIABLE))
    model = os.getenv(MODEL_VARIABLE, DEFAULT_MODEL)
    return Assistant(client, model, load_scorer())


def make_client(api_key):
    """Return a client for the Gemini API, or stop with a message if the key is missing."""
    if not api_key:
        sys.exit(f"{KEY_VARIABLE} is missing. Add it to .env in the project root.")
    options = types.HttpOptions(
        # The library counts the timeout in milliseconds.
        timeout=REQUEST_TIMEOUT_SECONDS * 1000,
        retry_options=types.HttpRetryOptions(attempts=REQUEST_ATTEMPTS),
    )
    return genai.Client(api_key=api_key, http_options=options)


def load_scorer():
    """Return the crash scorer, or stop with a message if the models are missing."""
    try:
        return CrashScorer()
    except FileNotFoundError as error:
        sys.exit(
            f"{error}\nRun the modeling first: uv run python -m data3800_gruppe_7.modeling"
        )


class Assistant:
    """Keeps one conversation with the language model."""

    def __init__(self, client, model, scorer):
        # The client is kept because its connection closes when it is deleted.
        self.client = client
        self.model = model
        self.scorer = scorer
        settings = types.GenerateContentConfig(
            system_instruction=build_system_prompt(),
            tools=[build_tool(scorer)],
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        )
        # The conversation keeps the earlier questions, answers and tool calls.
        self.conversation = client.chats.create(model=model, config=settings)

    def chat(self):
        """Answer what the user types in the terminal, until an empty line."""
        print("Ask about our results or describe a crash. An empty line stops.")
        while question := input("> ").strip():
            print(self.answer(question))

    def answer(self, question):
        """Return the answer to one question, after running the tool calls it needs."""
        message = question
        for _ in range(MAX_TOOL_ROUNDS):
            reply = self.conversation.send_message(message)
            if not reply.function_calls:
                return reply.text
            message = [self.run_tool(call) for call in reply.function_calls]
        return "The language model kept asking for the tool without giving an answer."

    def run_tool(self, call):
        """Score the crash in a tool call and return the result for the language model.

        The call is printed, so the user sees what our model was asked. A
        value the scorer rejects goes back as an error, so the language model
        can correct it or ask the user.
        """
        described = call.args or {}
        print(f"{TOOL_NAME}: {json.dumps(described)}")
        try:
            result = self.scorer.score(described)
        except (TypeError, ValueError) as error:
            result = {"error": str(error)}
        # The id tells the language model which of its calls this result answers.
        response = types.FunctionResponse(id=call.id, name=call.name, response=result)
        return types.Part(function_response=response)


def build_tool(scorer):
    """Describe the score_crash tool, so the language model knows how to call it.

    The allowed values come from the trained model, so the description always
    matches what the model can score.
    """
    properties = {}
    for feature, categories in scorer.allowed_categories().items():
        properties[feature] = {
            "type": "string" if isinstance(categories[0], str) else "number",
            "description": FEATURE_DESCRIPTIONS[feature],
            "enum": categories,
        }
    for feature, (low, high) in NUMBER_LIMITS.items():
        properties[feature] = {
            "type": "number",
            "description": FEATURE_DESCRIPTIONS[feature],
            "minimum": low,
            "maximum": high,
        }
    declaration = types.FunctionDeclaration(
        name=TOOL_NAME,
        description=(
            f"Score one crash with the project's trained model, {MODEL_NAME}. "
            "Give only the details the user stated."
        ),
        parameters_json_schema={"type": "object", "properties": properties},
    )
    return types.Tool(function_declarations=[declaration])


def build_system_prompt():
    """Return the instructions followed by every result file."""
    files = [read_result_file(path) for path in RESULT_FILES]
    return SYSTEM_PROMPT + "\n\n" + "\n\n".join(files)


def read_result_file(relative_path):
    """Return a result file as text, under a line with its name."""
    text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    return f"=== {relative_path} ===\n{text}"


if __name__ == "__main__":
    main()
