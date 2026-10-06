"""
A self-critiquing agent: Section 6.7's agent and check, a critic that reads each
answer the check passes, revisions your code asks for, and lessons kept in
Chapter 7's store through a gate.
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from policy_agent import (SYSTEM_PROMPT, WITHHELD, ContractMiddleware,   # Section 6.7
                          as_langchain_tool, contracts as POLICY_CONTRACTS)
from self_critique import (Finding, LessonStore, attempt_history, check_findings,
                           lesson_report, make_lesson_contract, record_attempts,
                           refine, review_report)
from tool_contracts import ToolContract, ToolRegistry

EVIDENCE_TOOLS = {"look_up_flight", "calculate",              # not notes, not reviews
                  "search_policies"}
NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}

CRITIC_PROMPT = (
    "You review an answer to a traveller's question against the tool results it was "
    "based on. Report only problems that would change what the traveller does: a "
    "passage that does not support the sentence citing it, a rule applied to the wrong "
    "case, part of the question left unanswered. Write one problem per line, starting "
    "with '- '. If there are none, reply exactly: NO ISSUES"
)

REFLECT_PROMPT = (
    "Below are your attempts at one answer and what the application found. If a "
    "check failed on one attempt and a later attempt passed, you may call "
    "keep_lesson once per failed rule, describing what the revision changed. If "
    "nothing is worth keeping, reply without calling it."
)


def parse_critique(text: str) -> list[Finding]:
    """NO ISSUES means none. A reply that isn't a list counts as one objection:
    an answer is never approved by a review your code couldn't read."""
    if text.strip() == "NO ISSUES":
        return []
    lines = [line[2:].strip() for line in text.splitlines() if line.startswith("- ")]
    return [Finding("critic", "critic", line) for line in lines or [text.strip()]]


def _call(name: str, call_id: str) -> dict:
    return {"name": name, "args": {}, "id": call_id, "type": "tool_call"}


class CritiqueAgent:
    """One session: an answer is drafted, checked, critiqued and revised, and the
    attempts are recorded. Build a new one for every session."""

    def __init__(self, model, critic, store: LessonStore, session_id: str,
                 max_revisions: int = 2):
        self.critic, self.store, self.session_id = critic, store, session_id
        self.max_revisions = max_revisions
        contracts = POLICY_CONTRACTS + [
            ToolContract("recall_lessons", "Show the notes this agent kept about its "
                         "own earlier drafts.", NO_ARGS,
                         handler=lambda: lesson_report(store)),
            ToolContract("review_answer", "Reviews are requested by the application "
                         "after you answer; do not call this.", NO_ARGS,
                         handler=lambda: "Reviews are requested by the application."),
        ]
        self.agent = create_agent(
            model=model,
            tools=[as_langchain_tool(c) for c in contracts],
            system_prompt=SYSTEM_PROMPT,
            middleware=[ContractMiddleware(ToolRegistry(contracts))],
        )

    def answer(self, question: str) -> str:
        messages = [HumanMessage(question),                  # lessons recalled in code
                    AIMessage(content="",
                              tool_calls=[_call("recall_lessons", "lessons_0")]),
                    ToolMessage(lesson_report(self.store), tool_call_id="lessons_0",
                                name="recall_lessons")]
        messages = self.agent.invoke({"messages": messages})["messages"]

        def evidence() -> list[str]:                         # what the tools returned
            return [str(m.text) for m in messages if isinstance(m, ToolMessage)
                    and m.status == "success" and m.name in EVIDENCE_TOOLS]

        def critique(answer: str) -> list[Finding]:   # a fresh call, not this thread
            prompt = (f"Question: {question}\n\nTool results:\n"
                      + "\n\n".join(evidence()) + f"\n\nAnswer:\n{answer}")
            reply = self.critic.invoke([SystemMessage(CRITIC_PROMPT),
                                        HumanMessage(prompt)])
            return parse_critique(str(reply.text))

        # The review reaches the model as a tool result in the same conversation.
        def revise(answer: str, findings: list[Finding]) -> str:
            nonlocal messages
            call_id = f"review_{len(messages)}"
            messages = messages[:-1] + [
                AIMessage(content=answer, tool_calls=[_call("review_answer", call_id)]),
                ToolMessage(review_report(findings), tool_call_id=call_id,
                            name="review_answer")]
            messages = self.agent.invoke({"messages": messages})["messages"]
            return str(messages[-1].text)

        self.result = refine(str(messages[-1].text),
                             lambda answer: check_findings(answer, question,
                                                           evidence()),
                             critique, revise, self.max_revisions)
        self.episode = record_attempts(self.store, self.result, self.session_id)
        self.messages = messages
        return self.result.delivered.answer if self.result.delivered else WITHHELD

    def reflect(self, model) -> list[str]:
        """Reflexion's step, gated. Asked only when a check failed and a later attempt
        passed, since no other session can produce a lesson the gate accepts."""
        attempts = self.result.attempts
        first_fail = next((a.number for a in attempts if not a.passes), None)
        if first_fail is None or not any(a.passes for a in attempts[first_fail:]):
            return []
        contract = make_lesson_contract(self.store, self.result, self.session_id,
                                        self.episode.memory_id)
        reply = model.bind_tools([as_langchain_tool(contract)]).invoke(
            [SystemMessage(REFLECT_PROMPT), HumanMessage(attempt_history(self.result))])
        registry = ToolRegistry([contract])
        return [registry.execute(c["name"], c["args"]).content
                for c in reply.tool_calls]


if __name__ == "__main__":
    import os
    import sys
    from datetime import date

    from langchain_anthropic import ChatAnthropic
    from policy_retrieval import QUESTION

    MODEL = os.environ.get("BOOK_MODEL")                 # a current Claude model ID
    if not MODEL:
        sys.exit("BOOK_MODEL is not set. Set it to a current Claude model ID "
                 "and run this again.")
    model = ChatAnthropic(model=MODEL)
    agent = CritiqueAgent(model, critic=model,
                          store=LessonStore("lessons", "flight-agent"),
                          session_id=date.today().isoformat())
    print(agent.answer(QUESTION))
    print(agent.episode.text)
    for outcome in agent.reflect(model):
        print(outcome)
