import json

from agentic_talent_screener.workflow.requirements import get_job
from agentic_talent_screener.workflow.storage import WorkflowError, audit, encode, expect_revision, new_id, require, utc_now


def current_proposal(db, job_id, ranking_id):
    return db.execute(
        """SELECT * FROM shortlist_proposals WHERE (job_id, ranking_id) = (?, ?)
        ORDER BY proposal_no DESC LIMIT 1""", (job_id, ranking_id),
    ).fetchone()


def create_proposal(db, job_id, ranking_id, actor, *, previous_ranking_id=None) -> str:
    job = get_job(db, job_id)
    baseline = job["last_approved_shortlist_id"]
    last_size = db.execute(
        """SELECT size FROM shortlist_proposal_revisions WHERE job_id = ? AND size IS NOT NULL
        ORDER BY saved_at DESC, revision_no DESC LIMIT 1""", (job_id,),
    ).fetchone()
    comparison = current_proposal(db, job_id, previous_ranking_id) if not baseline and previous_ranking_id else None
    previous = current_proposal(db, job_id, ranking_id)
    if previous and previous["state"] == "draft":
        raise WorkflowError(409, "PROPOSAL_ALREADY_OPEN", "Review the existing proposal.")
    proposal_id = new_id()
    db.execute(
        """INSERT INTO shortlist_proposals VALUES (?, ?, ?, ?, 'draft', 1, ?, ?, ?, ?, ?)""",
        (job_id, proposal_id, ranking_id, previous["proposal_no"] + 1 if previous else 1, baseline,
         previous_ranking_id, comparison["proposal_id"] if comparison else None,
         comparison["current_revision_no"] if comparison else None, utc_now()),
    )
    db.execute("INSERT INTO shortlist_proposal_revisions VALUES (?, ?, 1, ?, '[]', '[]', ?, ?)",
               (job_id, proposal_id, last_size["size"] if last_size else None, actor, utc_now()))
    audit(db, job_id, actor, "shortlist_proposed", {"proposal_id": proposal_id, "ranking_id": ranking_id})
    return proposal_id


def proposal_record(db, job_id, proposal_id):
    return require(db.execute(
        """SELECT p.*, r.size, r.overrides_json, r.prior_override_decisions_json
        FROM shortlist_proposals p JOIN shortlist_proposal_revisions r ON
        (p.job_id, p.proposal_id, p.current_revision_no) = (r.job_id, r.proposal_id, r.revision_no)
        WHERE (p.job_id, p.proposal_id) = (?, ?)""", (job_id, proposal_id),
    ).fetchone())


def ranked_ids(db, job_id, ranking_id) -> dict[str, int]:
    return dict(db.execute(
        """SELECT candidate_id, rank FROM ranking_rows WHERE (job_id, ranking_id) = (?, ?)
        AND rank IS NOT NULL ORDER BY rank""", (job_id, ranking_id),
    ).fetchall())


def approval_record(db, job_id, shortlist_id):
    return require(db.execute(
        "SELECT * FROM shortlist_approvals WHERE (job_id, shortlist_id) = (?, ?)", (job_id, shortlist_id),
    ).fetchone())


def shortlist_cutoff(db, job_id, ranking_id, ranked, size) -> tuple[list | None, list | None]:
    if size is None:
        return None, None
    config = db.execute(
        """SELECT b.settings_json FROM rankings k JOIN scoring_runs r USING (job_id, run_id)
        JOIN scoring_bases b USING (job_id, basis_id) WHERE (k.job_id, k.ranking_id) = (?, ?)""", (job_id, ranking_id),
    ).fetchone()
    near = json.loads(config["settings_json"])["config"]["near_cutoff_count"]
    ids = list(ranked)
    return ids[:size], ids[size:size + near]


def member_ids(db, job_id, shortlist_id) -> list[str]:
    return [row[0] for row in db.execute(
        "SELECT candidate_id FROM shortlist_members WHERE (job_id, shortlist_id) = (?, ?) ORDER BY rank",
        (job_id, shortlist_id),
    )]


def compare_members(old, current, old_ranks, current_ranks) -> dict:
    old_members, new_members = set(old), set(current)

    def change(candidate_id):
        return {"candidate_id": candidate_id, "old_rank": old_ranks.get(candidate_id),
                "new_rank": current_ranks.get(candidate_id)}

    return {
        "entered": [change(cid) for cid in current if cid not in old_members],
        "exited": [change(cid) for cid in old if cid not in new_members],
        "rank_changes": [change(cid) for cid in current if cid in old_members
                         and old_ranks.get(cid) != current_ranks.get(cid)],
    }


def prior_overrides(db, proposal):
    if not proposal["approval_baseline_id"]:
        return []
    return json.loads(approval_record(db, proposal["job_id"], proposal["approval_baseline_id"])["effective_overrides_json"])


def proposal_view(db, job_id, proposal_id, current_ranking_id=None) -> dict:
    proposal = proposal_record(db, job_id, proposal_id)
    ranking_id, size = proposal["ranking_id"], proposal["size"]
    ranked = ranked_ids(db, job_id, ranking_id)
    top, near = shortlist_cutoff(db, job_id, ranking_id, ranked, size)
    decisions = json.loads(proposal["prior_override_decisions_json"])
    resolved = {decision["override_id"]: decision for decision in decisions}
    overrides = json.loads(proposal["overrides_json"])
    previous = prior_overrides(db, proposal)
    applied = [
        item for item in previous if resolved.get(item["override_id"], {}).get("decision") == "reconfirm"
    ] + overrides
    effective = None
    if top is not None:
        selected = set(top)
        for item in applied:
            if item["action"] == "include":
                selected.add(item["candidate_id"])
            else:
                selected.discard(item["candidate_id"])
        effective = [cid for cid in ranked if cid in selected]
    baseline = proposal["approval_baseline_id"]
    old_ranking_id, old_members = proposal["previous_ranking_id"], None
    if baseline:
        old_approval = approval_record(db, job_id, baseline)
        old_ranking_id, old_members = old_approval["ranking_id"], member_ids(db, job_id, baseline)
    elif proposal["comparison_proposal_id"]:
        old_revision = db.execute(
            """SELECT size FROM shortlist_proposal_revisions WHERE (job_id, proposal_id, revision_no) = (?, ?, ?)""",
            (job_id, proposal["comparison_proposal_id"], proposal["comparison_revision_no"]),
        ).fetchone()
        if old_revision["size"] is not None:
            old_members = list(ranked_ids(db, job_id, old_ranking_id))[:old_revision["size"]]
    comparison = {"ranking_id": old_ranking_id, "shortlist_id": baseline} if old_ranking_id else None
    if old_members is not None and effective is not None:
        old_ranks = ranked_ids(db, job_id, old_ranking_id)
        comparison["computed_top_n"] = compare_members(old_members, top, old_ranks, ranked)
        comparison["effective_membership"] = compare_members(old_members, effective, old_ranks, ranked)
    state = proposal["state"]
    if state == "draft" and ranking_id != current_ranking_id:
        state = "superseded"
    approved = db.execute("SELECT shortlist_id FROM shortlist_approvals WHERE (job_id, proposal_id) = (?, ?)",
                          (job_id, proposal_id)).fetchone()
    return {
        "proposal_id": proposal_id, "ranking_id": ranking_id, "revision": f"{proposal_id}:{proposal['current_revision_no']}",
        "status": state, "size": size, "top_n_ids": top, "near_cutoff_ids": near, "effective_member_ids": effective,
        "overrides": overrides, "prior_override_decisions": decisions,
        "prior_overrides": [
            {**item, "decision": resolved.get(item["override_id"], {}).get("decision"),
             "can_reconfirm": item["candidate_id"] in ranked} for item in previous
        ],
        "comparison": comparison, "shortlist_id": approved["shortlist_id"] if approved else None,
    }


def editable_proposal(db, job_id, proposal_id, current_ranking_id, expected_revision):
    proposal = proposal_record(db, job_id, proposal_id)
    expect_revision(f"{proposal_id}:{proposal['current_revision_no']}", expected_revision)
    if proposal["state"] != "draft" or proposal["ranking_id"] != current_ranking_id or (
        proposal["approval_baseline_id"] != get_job(db, job_id)["last_approved_shortlist_id"]
    ):
        raise WorkflowError(409, "PROPOSAL_NOT_CURRENT", "Review a proposal on the current ranking.")
    return proposal


def save_proposal(db, job_id, proposal_id, actor, body, current_ranking_id) -> dict:
    proposal = editable_proposal(db, job_id, proposal_id, current_ranking_id, body["expected_revision"])
    previous = {item["override_id"]: item for item in prior_overrides(db, proposal)}
    ranked = ranked_ids(db, job_id, proposal["ranking_id"])
    active_candidates, reviewed = set(), set()
    prior_edits = {item["override_id"]: item for item in json.loads(proposal["prior_override_decisions_json"])}
    old_overrides = {item["candidate_id"]: item for item in json.loads(proposal["overrides_json"])}
    decisions, overrides, now = [], [], utc_now()
    for item in body["prior_override_decisions"]:
        key = item["override_id"]
        if key in reviewed or key not in previous:
            raise WorkflowError(422, "INVALID_PRIOR_OVERRIDE", "Review each override from the baseline at most once.")
        reviewed.add(key)
        if item["decision"] == "reconfirm":
            candidate_id = previous[key]["candidate_id"]
            if candidate_id not in ranked:
                raise WorkflowError(422, "CANDIDATE_NOT_RANKED", "Drop prior overrides for unranked candidates.")
            active_candidates.add(candidate_id)
        old = prior_edits.get(key)
        decisions.append(old if old and old["decision"] == item["decision"] else
                         {**item, "actor_id": actor, "recorded_at": now})
    for item in body["overrides"]:
        candidate_id = item["candidate_id"]
        if candidate_id not in ranked or candidate_id in active_candidates:
            raise WorkflowError(422, "INVALID_OVERRIDE", "A ranked candidate can have only one effective override.")
        active_candidates.add(candidate_id)
        old = old_overrides.get(candidate_id)
        overrides.append(old if old and all(old[k] == v for k, v in item.items()) else
                         {**item, "override_id": new_id(), "actor_id": actor, "recorded_at": now})
    number = proposal["current_revision_no"] + 1
    db.execute("INSERT INTO shortlist_proposal_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
               (job_id, proposal_id, number, body["size"], encode(overrides), encode(decisions), actor, now))
    db.execute("UPDATE shortlist_proposals SET current_revision_no = ? WHERE (job_id, proposal_id) = (?, ?)",
               (number, job_id, proposal_id))
    audit(db, job_id, actor, "shortlist_proposal_edited", {"proposal_id": proposal_id, "ranking_id": proposal["ranking_id"]},
          {"revision_no": number, "removed_override_ids": sorted(
              {item["override_id"] for item in old_overrides.values()} - {item["override_id"] for item in overrides}
          )})
    return proposal_view(db, job_id, proposal_id, current_ranking_id)


def approve_proposal(db, job_id, proposal_id, actor, revision, current_ranking_id) -> dict:
    proposal = editable_proposal(db, job_id, proposal_id, current_ranking_id, revision)
    view = proposal_view(db, job_id, proposal_id, current_ranking_id)
    if not view["top_n_ids"] or any(item["decision"] is None for item in view["prior_overrides"]):
        raise WorkflowError(409, "PROPOSAL_REVIEW_INCOMPLETE",
                            "Select a size, ensure candidates are ranked, and resolve all prior overrides.")
    shortlist_id, now = new_id(), utc_now()
    reviewed = {item["override_id"]: item for item in view["prior_override_decisions"]}
    saved = list(view["overrides"])
    for prior in prior_overrides(db, proposal):
        decision = reviewed[prior["override_id"]]
        if decision["decision"] == "reconfirm":
            saved.append({
                "candidate_id": prior["candidate_id"], "action": prior["action"], "reason": prior["reason"],
                "override_id": new_id(), "actor_id": decision["actor_id"], "recorded_at": decision["recorded_at"],
                "prior_override_id": prior["override_id"],
            })
    ranking_id = proposal["ranking_id"]
    db.execute("INSERT INTO shortlist_approvals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
               (job_id, shortlist_id, ranking_id, proposal_id, proposal["current_revision_no"],
                proposal["approval_baseline_id"], encode(saved), actor, now))
    ranks = ranked_ids(db, job_id, ranking_id)
    db.executemany("INSERT INTO shortlist_members VALUES (?, ?, ?, ?, ?)",
                   [(job_id, shortlist_id, cid, ranking_id, ranks[cid]) for cid in view["effective_member_ids"]])
    db.execute("UPDATE shortlist_proposals SET state = 'approved' WHERE (job_id, proposal_id) = (?, ?)",
               (job_id, proposal_id))
    db.execute("UPDATE jobs SET last_approved_shortlist_id = ? WHERE job_id = ?", (shortlist_id, job_id))
    audit(db, job_id, actor, "shortlist_approved",
          {"shortlist_id": shortlist_id, "ranking_id": ranking_id, "proposal_id": proposal_id},
          {"revision_no": proposal["current_revision_no"], "member_count": len(view["effective_member_ids"])})
    return approved_shortlist(db, job_id, shortlist_id)


def approved_shortlist(db, job_id, shortlist_id) -> dict:
    approval = dict(approval_record(db, job_id, shortlist_id))
    revision = db.execute(
        """SELECT size, prior_override_decisions_json FROM shortlist_proposal_revisions
        WHERE (job_id, proposal_id, revision_no) = (?, ?, ?)""",
        (job_id, approval["proposal_id"], approval["revision_no"]),
    ).fetchone()
    profile = db.execute(
        """SELECT b.profile_version, j.profile_id FROM rankings k JOIN scoring_runs r USING (job_id, run_id)
        JOIN scoring_bases b USING (job_id, basis_id) JOIN jobs j USING (job_id)
        WHERE (k.job_id, k.ranking_id) = (?, ?)""", (job_id, approval["ranking_id"]),
    ).fetchone()
    ranked = ranked_ids(db, job_id, approval["ranking_id"])
    size = revision["size"]
    top, near = shortlist_cutoff(db, job_id, approval["ranking_id"], ranked, size)
    overrides = json.loads(approval.pop("effective_overrides_json"))
    return {
        **approval, **dict(profile), "size": size, "top_n_ids": top, "near_cutoff_ids": near,
        "approved_member_ids": member_ids(db, job_id, shortlist_id),
        "overrides": overrides,
        "prior_override_decisions": json.loads(revision["prior_override_decisions_json"]),
    }
