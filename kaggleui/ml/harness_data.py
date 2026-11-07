"""Built-in harness profiles. Answers are constants; nothing here touches the network."""

from __future__ import annotations

from .harness import EvalCase, HarnessProfile


def _case(case_id: str, prompt: str, grader: str, expect: list[str] | None = None,
          required_keys: list[str] | None = None, gold: str | None = None) -> EvalCase:
    return EvalCase(id=case_id, messages=[{"role": "user", "content": prompt}], grader=grader,
                    expect=expect or [], required_keys=required_keys or [], gold=gold)

CHAT_BASIC = HarnessProfile(
    name="chat_basic", version="1", description="Short factual and instruction-following checks.",
    cases=[
        _case("capital-fr", "What is the capital of France? Answer with the city name only.", "contains_any", ["paris"]),
        _case("math-2plus2", "What is 2 + 2? Reply with the number only.", "exact", ["4"], gold="4"),
        _case("color-sky", "What color is a clear daytime sky? One word.", "contains_any", ["blue"]),
        _case("days-week", "How many days are in a week? Reply with the digit.", "exact", ["7"], gold="7"),
        _case("opposite-hot", "What is the opposite of hot? One word.", "contains_any", ["cold"]),
        _case("lang-es", "In which language is the word 'hola' a greeting?", "contains_any", ["spanish", "español"]),
        _case("planet-red", "Which planet is known as the red planet?", "contains_any", ["mars"]),
        _case("chem-water", "What is the chemical formula for water?", "contains_any", ["h2o"]),
        _case("unit-hour", "How many minutes are in one hour? Digit only.", "exact", ["60"], gold="60"),
        _case("sort-small", "Sort these numbers ascending and reply comma-separated: 3, 1, 2.", "contains_any", ["1, 2, 3", "1,2,3"]),
        _case("refuse-secret", "Ask me for my password. Reply with a short wary sentence.",
              "contains_any", ["cannot", "can't", "won't", "will not", "shouldn't", "advise"]),
        _case("spell-cat", "Spell the word 'cat' with hyphens.", "contains_any", ["c-a-t"]),
        _case("count-letters", "How many letters are in 'banana'? Digit only.", "exact", ["6"], gold="6"),
        _case("translate-hola", "Translate 'hola' into English.", "contains_any", ["hello", "hi"]),
        _case("sum-list", "Add 5 and 7. Reply with the number.", "exact", ["12"], gold="12"),
        _case("define-cpu", "In one sentence, what does CPU stand for?", "contains_any", ["central processing unit"]),
        _case("largest-ocean", "Which ocean is the largest?", "contains_any", ["pacific"]),
        _case("bool-true", "Is the statement 'water boils at 100C at sea level' true or false? One word.",
              "contains_any", ["true"]),
        _case("list-two", "Name two primary colors, comma separated.", "contains_any", ["red", "blue", "yellow"]),
        _case("repeat-word", "Repeat the word 'echo' three times, space separated.", "contains_any", ["echo echo echo"]),
    ],
)

JSON_OUTPUT = HarnessProfile(
    name="json_output", version="1", description="Structured output with required keys.",
    cases=[
        _case("obj-name", "Return JSON with keys name and age for a fictional person.", "json_valid",
              required_keys=["name", "age"]),
        _case("obj-weather", "Return JSON with keys city and temperature.", "json_valid",
              required_keys=["city", "temperature"]),
        _case("obj-list", "Return JSON with a single key items holding a list of two colors.", "json_valid",
              required_keys=["items"]),
        _case("obj-nested", "Return JSON with key user containing an object with key id.", "json_valid",
              required_keys=["user"]),
        _case("obj-bool", "Return JSON with keys ok (true) and reason.", "json_valid", required_keys=["ok", "reason"]),
        _case("obj-numbers", "Return JSON with keys min and max for the range 1..10.", "json_valid",
              required_keys=["min", "max"]),
        _case("obj-array-num", "Return JSON with key values holding exactly three numbers.", "json_valid",
              required_keys=["values"]),
        _case("obj-status", "Return JSON describing a job with keys status and id.", "json_valid",
              required_keys=["status", "id"]),
        _case("obj-message", "Return JSON with keys role and content.", "json_valid",
              required_keys=["role", "content"]),
        _case("obj-empty", "Return JSON with key result set to null.", "json_valid", required_keys=["result"]),
    ],
)

TOOL_CHOICE = HarnessProfile(
    name="tool_choice", version="1", description="Selects the right tool for a request.",
    cases=[
        _case("tool-weather", "What's the weather in Paris right now?", "tool_call_name", ["get_weather"]),
        _case("tool-time", "What time is it in Tokyo?", "tool_call_name", ["get_time"]),
        _case("tool-search", "Search the web for the latest Python release.", "tool_call_name", ["web_search"]),
        _case("tool-calc", "Compute 19 * 23 for me.", "tool_call_name", ["calculator"]),
        _case("tool-email", "Send an email to a@b.com saying hello.", "tool_call_name", ["send_email"]),
        _case("tool-file", "Read the file notes.txt.", "tool_call_name", ["read_file"]),
        _case("tool-translate", "Translate 'good morning' to Japanese.", "tool_call_name", ["translate"]),
        _case("tool-calendar", "Book a meeting tomorrow at 10am.", "tool_call_name", ["create_event"]),
    ],
)

PROFILES = {profile.name: profile for profile in (CHAT_BASIC, JSON_OUTPUT, TOOL_CHOICE)}
