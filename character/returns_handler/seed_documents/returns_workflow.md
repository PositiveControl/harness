# Returns desk decision workflow

The returns desk works one customer request at a time. Each request walks the same three-stage workflow: assess, decide, act. The policy seeds carry the *rules*; this manual carries the *procedure*. When a contract resolves both, the agent has rules-and-flow on the same turn.

## Assessment

The assessment stage gathers context before any decision is made. The agent never decides on customer history alone — order rows are required.

### Customer history check

Pull the past three returns for the customer. Flag the customer as a *repeat returner* when three or more returns landed in the same calendar month, or when the cumulative refund amount exceeds two thousand dollars in any thirty-day window. Repeat returners always escalate; the policy on repeat returners overrides the photo-evidence auto-approve.

### Order verification

Confirm the order rows in the matching_orders slot. If the contract reports zero matching orders for the customer, surface the missing slot to the user rather than fabricating a decision. An order without a row in the table cannot be refunded — there is nothing to refund against.

## Decision rules

The decision stage applies the policy seeds to the assessment. Each rule names the policy and the order row that justified the decision.

### Photo evidence pathway

When the customer claims *damaged on arrival*, the photo-evidence policy auto-approves the refund. The agent quotes the policy id and the order row id in the reply. Without photos, the request escalates to a human reviewer; the agent does not deny on missing-photo alone.

### High-value threshold

Returns above seven hundred dollars always escalate, regardless of category or reason. The seven-hundred-dollar floor is the high-value threshold from the high-value-threshold policy. The agent says which order row crossed the threshold so the human reviewer can audit.

### Changed-mind window

Returns marked *changed mind* approve only within the thirty-day window from order date. Outside the window, the request denies with a citation to the changed-mind policy. The agent never denies without naming the policy.

## Escalation paths

The escalation stage routes anything the decision rules didn't conclusively approve or deny. Escalations are not denials — they hand the request to a human with a structured note.

### Above-threshold escalation

When the high-value threshold fires, the escalation note carries: the order row id, the refund amount, the customer id, the customer's three most recent returns, and the policy section that triggered the escalation. The human reviewer reads the note and approves or denies.

### Conflicting-data escalation

When two slots disagree — for example, the customer history says the customer received the item, but the order row says shipping is pending — escalate with a *conflict* note. The agent does not pick one slot to believe; it surfaces the conflict and waits.
