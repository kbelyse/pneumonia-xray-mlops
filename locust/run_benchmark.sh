#!/bin/sh
# Runs the same Locust test against 1, 2 and 4 api containers and saves the csv files.
# Usage: sh locust/run_benchmark.sh [users] [seconds]
USERS=${1:-50}
SECONDS_RUN=${2:-90}
mkdir -p locust/results

for N in 1 2 4; do
  echo "== $N container(s) =="
  docker compose up -d --scale api=$N
  # wait until every container answers
  until [ "$(docker compose ps api --format '{{.Health}}' | grep -c healthy)" -eq "$N" ]; do sleep 3; done
  docker compose restart nginx >/dev/null
  sleep 5
  locust -f locust/locustfile.py --host http://localhost:8080 --headless \
    -u "$USERS" -r 5 -t "${SECONDS_RUN}s" --csv "locust/results/${N}_containers_${USERS}u" --only-summary
done
python locust/summarize.py
