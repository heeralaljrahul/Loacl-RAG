# Infrastructure Notes

## Database backups

Postgres is backed up every six hours to object storage with a 35-day retention. Restores are rehearsed quarterly; the last rehearsal took 41 minutes for the primary cluster.

## Deployment

Deployments run through the release pipeline and require two approvals. Friday afternoon deploys are blocked by policy after 14:00 local time.

## Monitoring

Alerts route to PagerDuty. The p99 latency budget for the checkout API is 800 milliseconds; sustained breaches page the service owner.

## Capacity

The staging cluster runs at roughly one-third of production capacity and is not a valid load-testing target.
