from hard_prompt.nlp_prober_gen import PROMPT_ONE
from hard_prompt.nlp_prober_confi_comput import PROMPT_TWO
from hard_prompt.nlp_dev_doc_gen import PROMPT_THREE
from hard_prompt.nlp_dd_confi_comput import PROMPT_FOUR
from hard_prompt.agent_dd_implement import PROMPT_FIVE
from hard_prompt.agent_improve_commentor import PROMPT_SIX
from hard_prompt.agent_iterat_improver import PROMPT_SEVEN
from hard_prompt.agent_exception_catcher import PROMPT_EIGHT
import json
import subprocess
from pathlib import Path
from typing import Final
from codex_harness import CodexHarness

from Questions import (
    QUESTION_ZERO,
    QUESTION_ONE,
    QUESTION_TWO,
    QUESTION_THREE,
    QUESTION_FOUR,
    QUESTION_FIVE,
    QUESTION_SIX,
)
# config

NLP_MODEL = "gpt-5.4"
AGENT_MODEL = "claude-opus-4-7"
WROKING_SPACE: Final = "/home/xuanhe_linux_001/agentic_probe_rein/dummy_project"
RESPONSE_DIR = Path(__file__).parent / "response"

# ── Input helpers ─────────────────────────────────────────────────────────────

def get_input_placeholder(text: str) -> str:
    """Display a question and return the raw user input string."""
    print(f"\n{text}")
    return input(">> ")


def _ask_yn(text: str, default: bool | None = None) -> bool:
    """Loop until the user enters Y or N. Returns True for Y."""
    while True:
        raw = get_input_placeholder(text).strip().upper()
        if raw == "" and default is not None:
            return default
        if raw in ("Y", "N"):
            return raw == "Y"
        print("  Invalid input — please enter Y or N.")


def _ask_int_range(text: str, lo: int, hi: int) -> int:
    """Loop until the user enters an integer in [lo, hi]."""
    while True:
        raw = get_input_placeholder(text).strip()
        if raw.isdigit():
            value = int(raw)
            if lo <= value <= hi:
                return value
        print(f"  Invalid input — please enter a number between {lo} and {hi}.")


def _ask_pos_int(text: str, default: int = 3) -> int:
    """Loop until the user enters a positive integer, or Enter for default."""
    while True:
        raw = get_input_placeholder(text).strip()
        if raw == "":
            return default
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print(f"  Invalid input — please enter a positive integer, or press Enter for default ({default}).")
# ── State ─────────────────────────────────────────────────────────────────────

USER_ANSWER_ONE: str = ""
USER_ANSWER_TWO: int = 0
USER_ANSWER_THREE: int = 0
USER_ANSWER_FOUR: int = 3


# TODO below all action
probe_template = """
{
    "probe_designs": [
        { "probe_type": "string", "probe_name": "string", \
            "content": "string", "possible_sources": ["string"], \
                "confidence": "float between 0 and 1" },
        { "probe_type": "string", "probe_name": "string", \
            "content": "string", "possible_sources": ["string"], \
                "confidence": "float between 0 and 1" },
        ...
    ]
}"""

dev_plan_template = """
{
    "dev_plans": [
        { "content": "string", "metric": "string", "threshold": "string", "confidence": 0.0 },
        { "content": "string", "metric": "string", "threshold": "string", "confidence": 0.0 },
        { "content": "string", "metric": "string", "threshold": "string", "confidence": 0.0 }
    ]
}
"""


def _nlp_call(message: str) -> dict:
    harness = CodexHarness(model=NLP_MODEL, mode="chat")
    harness.start_conversation()
    response = harness.query(message, response_format={"type": "json_object"})
    harness.exit()
    return json.loads(response)


def _agent_call(prompt: str) -> None:
    subprocess.run(["claude", "-p", prompt], cwd=WROKING_SPACE, check=True)


def action_1_probe_generation_from_context():
    RESPONSE_DIR.mkdir(exist_ok=True)
    result = _nlp_call(f"{PROMPT_ONE}\n\n{USER_ANSWER_ONE}")
    out = RESPONSE_DIR / "probe_designs.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


def action_1a_probe_confidence():
    RESPONSE_DIR.mkdir(exist_ok=True)
    probe_designs = (RESPONSE_DIR / "probe_designs.json").read_text()
    result = _nlp_call(f"{PROMPT_TWO}\n\n{probe_designs}")
    out = RESPONSE_DIR / "probe_confidenced.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


def action_2_dev_doc_generation_from_probe():
    RESPONSE_DIR.mkdir(exist_ok=True)
    probe_data = json.loads((RESPONSE_DIR / "probe_confidenced.json").read_text())
    selected = probe_data["probe_designs"][USER_ANSWER_TWO - 1]
    result = _nlp_call(f"{PROMPT_THREE}\n\n{json.dumps(selected, indent=2)}")
    out = RESPONSE_DIR / "dev_doc.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


def action_2a_dev_doc_confidence():
    RESPONSE_DIR.mkdir(exist_ok=True)
    dev_doc = (RESPONSE_DIR / "dev_doc.json").read_text()
    result = _nlp_call(f"{PROMPT_FOUR}\n\n{dev_doc}")
    out = RESPONSE_DIR / "dev_doc_confidenced.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


def action_3_agent_implementation():
    dev_doc_data = json.loads((RESPONSE_DIR / "dev_doc_confidenced.json").read_text())
    selected = dev_doc_data["dev_plans"][USER_ANSWER_THREE - 1]
    _agent_call(f"{PROMPT_FIVE}\n\nWrite prober.py and integrate it into train.py.\n\n{selected}")


def action_4_agent_improvement():
    _agent_call(f"{PROMPT_SIX}\n\nTarget file: train.py")


def action_4_iterate():
    _agent_call(PROMPT_SEVEN)


def action_run_training() -> tuple[bool, str]:
    result = subprocess.run(
        ["python", "train.py"],
        cwd=WROKING_SPACE,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    return result.returncode == 0, result.stderr


MAX_FIX_RETRIES = 5

def action_x_agentic_exception_catcher():
    success, error = action_run_training()
    retries = 0
    while not success:
        if retries >= MAX_FIX_RETRIES:
            print(f"Could not fix after {MAX_FIX_RETRIES} attempts. Last error:\n{error}")
            raise RuntimeError("Exception catcher exceeded max retries.")
        retries += 1
        print(f"Error detected (attempt {retries}/{MAX_FIX_RETRIES}):\n{error}\nAsking agent to fix...")
        _agent_call(f"{PROMPT_EIGHT}\n\nError output:\n{error}")
        success, error = action_run_training()
    print("Training and probe ran successfully.")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global USER_ANSWER_ONE, USER_ANSWER_TWO, USER_ANSWER_THREE, USER_ANSWER_FOUR

    # Step 1: confirm setup
    if not _ask_yn(QUESTION_ZERO):
        print("Please complete the setup before proceeding. Exiting.")
        return

    # Step 2: project context (free text)
    USER_ANSWER_ONE = get_input_placeholder(QUESTION_ONE).strip()

    action_1_probe_generation_from_context()
    action_1a_probe_confidence()

    while True:
        # Step 3: show probes and let user select one
        print("\n" + (RESPONSE_DIR / "probe_confidenced.json").read_text())
        USER_ANSWER_TWO = _ask_int_range(QUESTION_TWO, lo=1, hi=10)

        action_2_dev_doc_generation_from_probe()
        action_2a_dev_doc_confidence()

        # Step 4: show dev docs and let user select one
        print("\n" + (RESPONSE_DIR / "dev_doc_confidenced.json").read_text())
        USER_ANSWER_THREE = _ask_int_range(QUESTION_THREE, lo=1, hi=3)

        action_3_agent_implementation()
        action_x_agentic_exception_catcher()

        # Step 5: optionally comment, then iterate
        if _ask_yn(QUESTION_SIX, default=False):
            action_4_agent_improvement()
            action_x_agentic_exception_catcher()
        USER_ANSWER_FOUR = _ask_pos_int(QUESTION_FOUR, default=3)
        while USER_ANSWER_FOUR > 0:
            action_4_iterate()
            action_x_agentic_exception_catcher()
            USER_ANSWER_FOUR -= 1

        # Step 6: continue or exit
        if not _ask_yn(QUESTION_FIVE):
            print("Goodbye!")
            return


if __name__ == "__main__":
    main()
