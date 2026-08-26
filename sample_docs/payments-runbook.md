# Payments Runbook

## Error E-4471

Error E-4471 means the settlement file was rejected by the clearing house because the checksum footer was malformed. Re-queue the batch with `payctl requeue --batch <id>` and page the payments on-call immediately.

## Error E-2210

Error E-2210 is a benign duplicate-submission warning emitted when the upstream retries. No action is required unless it repeats more than five times in an hour.

## Error E-9003

Error E-9003 indicates the HSM signing key has expired. Do not retry — rotate the key first using the documented ceremony, then replay the batch.

## Escalation

If a batch is still stuck 20 minutes after a requeue, escalate to the Treasury Systems lead. Out of hours, use the #payments-sev channel rather than direct messages.

## Nightly cutoff

The nightly settlement cutoff is 22:45 UTC. Batches submitted after the cutoff land in the following business day's window.
