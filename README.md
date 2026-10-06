# The Complete Agentic AI Engineering Manual: companion code

The code for *The Complete Agentic AI Engineering Manual: 9 Battle-Tested Frameworks for Building Self-Improving
LLM Agents* by Daniel Ashgrove. Every program in the book is here, and every listing printed in the book is this
code, line for line. If the two ever differ, this repository is the authoritative copy.

## Getting started

You need Python 3.10 or newer. From this folder:

```
python3 -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
pip install -r setup/requirements.txt
python setup/check_setup.py
```

`check_setup.py` checks Python, the packages and your settings, and costs nothing. Most of the code needs no model
and no account. The programs that call a model need two environment variables: `ANTHROPIC_API_KEY`, a key from the
Claude Console, and `BOOK_MODEL`, a current model ID from platform.claude.com/docs/en/models/overview. The book's
Setup & Prerequisites section explains both.

## What is here

One folder per chapter that has code, and a `setup` folder. Every folder is complete on its own: it holds the
chapter's new files and, unchanged, every earlier file they import. Run a chapter's programs from inside its folder.

| Folder | What the chapter adds | Needs a model |
| --- | --- | --- |
| `setup` | `check_setup.py`, the environment check; `requirements.txt`, the tested package versions | Only with `--live` |
| `ch01` | `agent_vs_chatbot.py`; `three_countries.py`, the exercise solution | No |
| `ch03` | `react_loop.py` | Yes |
| `ch04` | `plan_and_execute.py` | Yes |
| `ch05` | `tool_contracts.py`; `containment_demo.py`; the exercise files `exercise_broken.py` and `ch3_calculate_probe.py` | No |
| `ch06` | `policy_retrieval.py`; `policy_agent.py`, the live script; `api_search_results.py` | Live script only |
| `ch07` | `agent_memory.py`; `memory_agent.py`, the live script | Live script only |
| `ch08` | `self_critique.py`; `critique_agent.py`, the live script | Live script only |
| `ch09` | `orchestration.py`; `orchestra_agent.py`, the live script | Live script only |
| `ch10` | `eval_harness.py`; `eval_agent.py`, the live script | Live script only |
| `ch11` | `guardrails.py`; `guarded_agent.py`, the live script | Live script only |
| `ch12` | `combined.py`; `combined_agent.py`, the live script | Live script only |
| `ch13` | `deploy.py`; `deploy_agent.py`, the live script; `dashboard.py`; `telemetry.py` | Live script only |

From Chapter 5 on, each chapter's main program runs a demonstration with scripted stand-ins in place of the model,
and prints exactly the output shown in the book. Each of those folders also has tests, which need no model and no
key:

```
cd ch06
python policy_retrieval.py
python -m unittest
```

## Changes since printing

None yet. If a library update changes how a program behaves, the fix is made here, and this section says what
changed and why.

## Licence

The code is released under the MIT licence; see `LICENSE`. The licence covers the code only. The book's text and
figures are not part of this repository and remain all rights reserved.
