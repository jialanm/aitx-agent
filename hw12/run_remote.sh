#!/usr/bin/env bash
# Run a GPU job on the EC2 instance from the laptop, copy its results back, then stop the instance.
#
# The job runs under nohup on the instance, so a dropped SSH connection does not kill it; this
# script polls for a done-marker, copies the result folder and the log, and stops the instance
# (stop keeps the disk; the results also stay on the instance).
#
# Usage:
#   HOST=ubuntu@1.2.3.4 KEY=~/.ssh/cmu_llm_11967.pem hw12/run_remote.sh lora_qwen3_4b \
#     "uv run --group hw12 python hw12/finetune_lora.py --epochs 3 --out data/hw12/lora_qwen3_4b" \
#     data/hw12/lora_qwen3_4b
#
# Arguments: NAME (log name), JOB (command run inside ~/aitx-agent), RESULT_DIR (relative to ~/aitx-agent).
# Environment: HOST (user@address), KEY (ssh key file), INSTANCE (default g6e), REGION (default us-east-1),
#              LOCAL_DIR (default data/hw12), POLL (seconds between checks, default 60).
set -euo pipefail

NAME=${1:?name}; JOB=${2:?job command}; RESULT_DIR=${3:?result dir on the instance}
HOST=${HOST:?set HOST=user@address}; KEY=${KEY:?set KEY=path/to/key.pem}
INSTANCE=${INSTANCE:-i-03256ddf79df2a182}; REGION=${REGION:-us-east-1}
LOCAL_DIR=${LOCAL_DIR:-data/hw12}; POLL=${POLL:-60}
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$HOST")

echo "$(date +%H:%M:%S) starting $NAME on $HOST"
"${SSH[@]}" "cd ~/aitx-agent && mkdir -p logs && rm -f logs/$NAME.done && \
  nohup bash -c '$JOB; echo \$? > logs/$NAME.done' > logs/$NAME.log 2>&1 &"

while ! "${SSH[@]}" "test -f ~/aitx-agent/logs/$NAME.done" 2>/dev/null; do
  echo "$(date +%H:%M:%S) running... $("${SSH[@]}" "tail -n 1 ~/aitx-agent/logs/$NAME.log" 2>/dev/null | cut -c1-110)"
  sleep "$POLL"
done
EXIT=$("${SSH[@]}" "cat ~/aitx-agent/logs/$NAME.done")
echo "$(date +%H:%M:%S) job finished with exit code $EXIT"

mkdir -p "$LOCAL_DIR"
scp -i "$KEY" -r "$HOST:~/aitx-agent/$RESULT_DIR" "$LOCAL_DIR/"
scp -i "$KEY" "$HOST:~/aitx-agent/logs/$NAME.log" "$LOCAL_DIR/"
echo "$(date +%H:%M:%S) copied $RESULT_DIR and logs/$NAME.log to $LOCAL_DIR/"

aws ec2 stop-instances --instance-ids "$INSTANCE" --region "$REGION" >/dev/null
echo "$(date +%H:%M:%S) stop requested for $INSTANCE"
exit "$EXIT"
