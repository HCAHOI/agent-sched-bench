from scripts.evaluation.evaluate_pennylane_revocable_tool_lease import (
    Action,
    Program,
    simulate,
)


def test_revocable_lease_reactivates_on_later_lender_tool_phase() -> None:
    programs = [
        Program(
            "lender",
            (Action(4, "exec"), Action(1, "llm", 100), Action(4, "exec")),
        ),
        Program("base-1", (Action(20, "other"),)),
        Program(
            "borrower",
            (Action(1, "llm", 200), Action(1, "other"), Action(1, "llm", 400)),
        ),
    ]

    permanent = simulate(programs, "permanent_loan", base_concurrency=2, feedback_budget_s=2)
    lease = simulate(programs, "revocable_lease", base_concurrency=2, feedback_budget_s=2)

    assert lease["feedback_trigger_count"] == 2
    assert lease["paused_borrower_s"] > 0
    assert lease["task_completion_s"]["borrower"] == 8
    assert lease["llm_request_overlap_s"] < permanent["llm_request_overlap_s"]


def test_revocable_lease_promotes_paused_borrower_when_base_slot_opens() -> None:
    programs = [
        Program("lender", (Action(4, "exec"), Action(1, "other"))),
        Program("base-1", (Action(20, "other"),)),
        Program(
            "borrower",
            (Action(1, "llm", 200), Action(1, "other"), Action(10, "llm", 400)),
        ),
    ]

    lease = simulate(
        programs,
        "revocable_lease",
        base_concurrency=2,
        feedback_budget_s=2,
    )

    assert lease["borrower_promotions"] == 1
    assert lease["task_completion_s"]["borrower"] == 15
