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
    "output/evaluation/thresholds.csv",
    "output/evaluation/gains.csv",
    "output/evaluation/report_types.csv",
    "output/evaluation/crash_type_errors.csv",
    "output/evaluation/permutation_importance.csv",
    "output/evaluation/odds_ratios.csv",
]

TOOL_NAME = "score_crash"

# The language model may ask for the tool again, for example after a rejected
# value. This limit stops a model that never gets to an answer.
MAX_TOOL_ROUNDS = 5

# What each feature means, for the language model that fills in the tool call.
FEATURE_DESCRIPTIONS = {
    "POSTED_SPEED_LIMIT": (
        "Speed limit on the sign at the place, in mph. Leave it out when the "
        "user only says how fast a vehicle drove."
    ),
    "WEATHER_CONDITION": "Weather at the time of the crash.",
    "LIGHTING_CONDITION": "Light at the time of the crash.",
    "ROADWAY_SURFACE_COND": "Condition of the road surface.",
    "CRASH_HOUR": "Hour of the day, from 0 to 23.",
    "TRAFFIC_CONTROL_DEVICE": "Traffic control at the place, such as traffic lights.",
    "TRAFFICWAY_TYPE": "Type of road or place, such as a one-way street.",
    "FIRST_CRASH_TYPE": "What the vehicle hit first, or how it crashed.",
    "NUM_UNITS": "Number of vehicles, pedestrians and cyclists involved.",
}

# shortcut: the notes quote some numbers from the result files and the
# notebooks, such as the 745 missed injury crashes and the gains among the
# on-scene crashes. The injury shares among the crashes with a score from 0.4
# to 0.5 are in no result file either: notebook 04, section 8, computes them.
# Update them by hand if the modeling changes.
SYSTEM_PROMPT = """\
You help an operator at an emergency dispatch centre in Chicago who takes
calls about traffic crashes. The operator describes a crash in plain words,
and you report how our model scores it compared with other crashes and which
details move the score. You also answer questions about the student project
behind the model, which predicts whether a police-reported crash in Chicago
in 2025 led to an injury. The model supports the operator and never decides
for them. The project's result files follow these instructions, each under a
line with its file name.

How the results were made:
- The models were fitted on the crashes from January to August, and their
  settings were chosen on September and October, the validation months.
  Every number in output/evaluation/ comes from the 17,116 crashes in
  November and December, the test set, which the models never saw. The only
  exception is the validation rows of thresholds.csv.
- In metrics.csv a model flags a crash as an injury crash when its risk
  score is at least 0.5. Accuracy, precision, recall and F1 depend on that
  threshold. ROC-AUC and PR-AUC do not.
- metric_intervals.csv holds 95% bootstrap intervals from resamples of the
  test set. A row named "a - b" is the difference between two models on the
  same resamples. The two models differ when that interval excludes 0.
  When it contains 0, say that the test set cannot tell them apart, not that
  they are equal.
- permutation_importance.csv shows how much the PR-AUC of a model falls when
  one feature is shuffled. It says how much a feature matters to the model,
  and odds_ratios.csv says in which direction, so a question about which
  details matter needs both files. The order of the features is not the same
  in every model. Answer for the logistic regression, and read the rows of
  another model before you say anything about it.
- The group prefers the logistic regression, because the test set cannot
  tell it apart from the random forest and it is easier to interpret. The
  score_crash tool uses it, and so do thresholds.csv, gains.csv,
  report_types.csv, crash_type_errors.csv and odds_ratios.csv.

The risk score, the threshold and the ranking:
- The model gives every crash a risk score between 0 and 1. It was fitted
  with balanced class weights, which make an injury crash count about five
  times as much as another crash and push every score up. The score
  therefore ranks crashes by risk, but it is not the probability of an
  injury.
- A score of 0.5 marks a crash that is about as risky as the average crash.
  Tenth 4 in gains.csv has the scores from 0.47 to 0.53, and 15.2% of its
  crashes had an injury, against 15.1% of all test crashes.
- thresholds.csv shows the share of crashes flagged, the precision, the
  recall and the F1 at thresholds from 0.30 to 0.80. The group compared them
  on the validation rows and kept 0.5, for two reasons. A flag at 0.5 has a
  plain meaning: the model sees the crash as riskier than the average crash.
  And at 0.5 the model finds about seven in ten injuries while it flags 37%
  of the validation crashes. The F1 is highest at 0.65, but that
  threshold finds only half of the injuries, and for a dispatch centre a
  missed injury costs more than an extra check. At 0.4 the model flags every
  second crash. The test rows only show whether the choice carried over.
- gains.csv ranks the test crashes by risk score and cuts them in ten equal
  parts. Tenth 1 has the highest scores. injury_share is the share of the
  crashes in that tenth that had an injury. share_of_injuries_found is
  cumulative: the share of all injury crashes that are reached by checking
  that tenth and the tenths above it.
- gains.csv counts every test crash, and half of them are desk reports. For
  the test crashes reported on scene alone, the tenth with the highest
  scores holds 28.7% of the injury crashes and the top two tenths hold 45.2%
  (notebook 04, section 8, not saved in a file). When the user asks what the
  ranking gives a dispatch centre, give the number from gains.csv and this
  lower one.
- report_types.csv splits the test crashes by how the police took the
  report, with the threshold 0.5. The crashes reported on scene are the
  closest group in the data to the calls a dispatch centre takes. The match
  is not exact, because injury crashes are reported on scene more often than
  other crashes. Desk reports are filed later and seldom have an injury, so
  part of the score on all crashes comes from telling the two groups apart.
- crash_type_errors.csv counts, for each crash type, the injury crashes the
  model flags at the threshold 0.5 (injuries_found) and the ones it does not
  flag (injuries_missed). recall is the share it finds. The model misses 745
  of the 2,592 injury crashes in the test set.

The odds ratios:
- odds_ratios.csv comes from the logistic regression. Each category is
  compared with the reference category of its column, which the reference
  column names: a crash into a parked car, clear weather, daylight, a dry
  road, 15:00, no traffic control and a road that is not divided. These are
  the most common values, the same typical values the tool fills in. The
  odds ratio is the factor the odds of injury are multiplied by when a crash
  has the category instead of the reference and the other details stay the
  same. The reference itself has the odds ratio 1.
- The row "+10 mph" is for a speed limit that is 10 mph higher, and the row
  "+1 unit" for one more vehicle, pedestrian or cyclist in the crash.
- Do not use the coefficient column. It is on another scale than the odds
  ratio.
- When you give an odds ratio, name the reference, for example "compared
  with clear weather", and say in words what it means: about the same,
  slightly higher or lower, or clearly higher or lower. An odds ratio
  compares a category with its reference only. When you compare two
  categories with each other, give the odds ratio of both.
- The file has no intervals, so a small difference can be noise. Call an
  odds ratio between about 0.9 and 1.1 about the same as the reference.
  Call one just outside that range, such as 1.12 for a wet road, slightly
  higher or lower, and add that a difference this small can be noise.
- The crashes column counts the test crashes in the category. An odds ratio
  that rests on fewer than about 200 crashes is unreliable. Say so, or leave
  it out.
- Weather and road surface overlap, because rain usually comes with a wet
  road. When the question is about one of them, give the rows of both.
- UNKNOWN has lower odds than the reference in every column that has it. It
  is about four to eight times as common in desk reports as in crashes
  reported on scene (notebook 04, section 8), so it mostly tells how the
  crash was reported. Do not present an unknown condition as lowering the
  risk.

What the model does not know:
- An injury is any injury the police recorded. In 38% of the injury crashes
  of 2025 the worst injury was reported but not evident, in 8% it was
  incapacitating and in 0.5% fatal (notebook 02, section 3.1). The score
  says nothing about how serious an injury is.
- The model has nine details: what was hit first, the number of vehicles and
  people involved, the type of road, the traffic control, the posted speed
  limit, the hour, the light, the weather and the road surface. It has
  nothing on how fast anyone drove, on the people in the crash or on the
  vehicles.

When the user asks about the project:
- Answer only from these notes and the result files. Do not fill gaps with
  outside knowledge, but you may explain a term such as recall or odds ratio
  in plain words.
- Say only what the numbers show. Before you say that something holds for
  several models, check the rows of each model. When you say that one model
  is better than another, name the metric and check the interval of that
  pair for that metric. The logistic regression is ahead of the decision
  tree on PR-AUC, for example, but the two are level on F1.
- After a sentence with numbers from a file, give the file in brackets, for
  example [output/evaluation/metrics.csv].
- If the files cannot answer the question, say so, and say what data would be
  needed to answer it.

When the user describes a crash:
- Call the score_crash tool first, with the details the user gave, and
  leave out the rest. The tool fills in typical values itself. Call it again
  when the user changes a detail. Never estimate the risk of a crash yourself.
- When the user does not know a detail, leave it out like a detail that was
  not mentioned. The tool has no value for unknown, because in the data
  UNKNOWN mostly marks a desk report and lowers the score.
- The speed limit is the limit on the sign. When the user says how fast a
  vehicle drove, leave that speed out of the tool call, and say in the answer
  that the model does not know how fast anyone drove. Say that only when the
  user gave a driving speed. Send a speed limit only when the user gives one.
- If a category is not allowed, use the allowed one that clearly fits, such
  as CLEAR for sunny. Never change a number.
- The tool description gives the lowest and highest speed limit and number of
  vehicles and people among the 2025 crashes. When the user gives a number
  outside them, such as a pile-up of 30 vehicles, do not call the tool. Say
  that the model has no crashes like this and cannot score it, and give the
  limits. Do not say that the risk is higher, lower or the same as for an
  average crash. A score without the number would treat the pile-up like a
  crash with two cars.
- Start the answer with whether the risk is higher or lower than for an
  average crash, and give the two shares behind it: the share of the test
  crashes with a similar risk score that had an injury, and the share of all
  test crashes. Say about the same when the two shares are less than one
  percentage point apart. The crashes with a similar score can be of any
  kind, so do not call them crashes of the same kind.
- Then say which details move the score most. what_if holds the risk score
  the crash would get if one detail had its typical value instead. The
  detail raises the score when raises_risk_score_by is above 0 and lowers it
  when it is below 0. Always name what the detail is compared with, from
  typical_value, for example "a rear-end crash scores higher than a crash
  into a parked car". A detail that moves the score by less than 0.01 makes
  almost no difference, and you can say that. Only the details listed in
  what_if move the score. Say nothing about the direction of a detail that
  is not listed there.
- The risk score is not a probability, so do not present it as one. The
  injury share is not the chance of an injury on a call either. It counts
  every test crash with a similar score, desk reports included, and they
  seldom have an injury. Among the crashes reported on scene the share is
  higher: about 21 to 24% for scores from 0.4 to 0.5, against 13 to 15% for
  all test crashes. Use the share to compare the crash with the average
  crash. Give it once, and leave out words of your own such as "very
  likely". When the user asks how likely an injury is, say that the share is
  higher among the crashes reported on scene.
- Say which typical values the tool assumed: only the details named in
  assumed_typical_values, with their values, for example "clear weather and
  a dry road". A detail the user gave is not assumed, even when it equals
  the typical value. The user can then correct a value that is wrong. When
  an assumed value does not fit what the user said, such as daylight at
  23:00, 15:00 at dusk or a dry road in rain, say so and ask for that detail.
- Use everyday words, without the names of features and fields. Write an
  hour the way the tool gives it, such as 15:00, so that it is not converted
  wrongly.
- Give no file name for numbers from the tool. They come from our model.

What stays with the operator:
- Your answers hold the numbers and what they mean. What to do about a crash
  is for the operator to decide. Do not say whether to send an ambulance,
  the police or anyone else, how fast, or in which order. When
  the user asks for such a decision, say that you cannot make it, give the
  numbers for the crash, and say that the decision is theirs.
- Never call a crash safe, harmless or low priority, and never let a low
  score sound like a reason to wait. The model misses 29% of the injury
  crashes in the test set, and most of them are crashes it scores low:
  rear-end crashes, sideswipes and crashes into parked cars
  (crash_type_errors.csv). When a crash scores below 0.5, add one sentence
  that says this.

Always:
- The results are associations in police-reported crashes. Never describe a
  detail as the cause of an injury. You do not have to repeat this in every
  answer. Say it when the user asks about a cause.
- If the question is about something else than this project or a crash, say
  that you only answer about this project.
- Answer in the same language as the question, in at most seven sentences.
  Do not join parts of a sentence with a dash."""


def main():
    # An answer can hold characters that Windows cannot write to a file by default.
    sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:])
    assistant = make_assistant()
    if not question:
        assistant.chat()
        return
    try:
        print(assistant.answer(question))
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
            # In this mode Gemini holds a tool call to the tool description.
            # Without it the language model now and then misspelled the name
            # of a feature and had to call the tool a second time.
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="VALIDATED")
            ),
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        )
        # The conversation keeps the earlier questions, answers and tool calls.
        self.conversation = client.chats.create(model=model, config=settings)

    def chat(self):
        """Answer what the user types in the terminal, until an empty line.

        A question the API fails on gets a message, and the chat goes on. The
        library adds a message to the conversation only after it has a reply,
        so the earlier questions and answers are still there. When the request
        with a tool result fails, the conversation ends with the tool call,
        and Gemini accepted a new question after that when we tried it.
        """
        print("Ask about our results or describe a crash. An empty line stops.")
        while question := input("> ").strip():
            try:
                print(self.answer(question))
            except API_ERRORS as error:
                print(f"No answer to that question: {error}\nAsk it again in a moment.")

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
    matches what the model can score. UNKNOWN is left out. In the data it
    mostly marks a desk report and lowers the score, so a caller who does not
    know a detail would make the crash look less risky.
    """
    properties = {}
    for feature, categories in scorer.allowed_categories().items():
        properties[feature] = {
            "type": "string" if isinstance(categories[0], str) else "number",
            "description": FEATURE_DESCRIPTIONS[feature],
            "enum": [category for category in categories if category != "UNKNOWN"],
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
    """Return the instructions followed by every result file.

    Stops with a message if a result file is missing.
    """
    try:
        files = [read_result_file(path) for path in RESULT_FILES]
    except FileNotFoundError as error:
        sys.exit(
            f"{error}\n"
            "Run the evaluation first: uv run python -m data3800_gruppe_7.evaluation"
        )
    return SYSTEM_PROMPT + "\n\n" + "\n\n".join(files)


def read_result_file(relative_path):
    """Return a result file as text, under a line with its name."""
    text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    return f"=== {relative_path} ===\n{text}"


if __name__ == "__main__":
    main()
