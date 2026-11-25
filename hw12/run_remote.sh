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
# Environment: KEY (ssh key file), INSTANCE (default g6e), REGION (default us-east-1), USER_NAME (default ubuntu),
#              LOCAL_DIR (default data/hw12), POLL (seconds between checks, default 60).
# The instance is started first, retrying every 15 seconds while AWS reports no capacity, and its
# address is read from AWS, so HOST need not be given; set HOST=user@address to skip the start.
# Set ATTACH=1 (with HOST) to skip starting the job and only poll, copy and stop, for a job
# that is already running on the instance.
set -euo pipefail

NAME=${1:?name}; JOB=${2:?job command}; RESULT_DIR=${3:?result dir on the instance}
KEY=${KEY:?set KEY=path/to/key.pem}
INSTANCE=${INSTANCE:-i-03256ddf79df2a182}; REGION=${REGION:-us-east-1}; USER_NAME=${USER_NAME:-ubuntu}
LOCAL_DIR=${LOCAL_DIR:-data/hw12}; POLL=${POLL:-60}

if [ -z "${HOST:-}" ]; then
  until aws ec2 start-instances --instance-ids "$INSTANCE" --region "$REGION" >/dev/null 2>/tmp/start_err.txt; do
    if grep -q "InsufficientInstanceCapacity" /tmp/start_err.txt; then
      echo "$(date +%H:%M:%S) no capacity yet, retrying in 15 s"
      sleep 15
    else
      echo "start refused for another reason:"; cat /tmp/start_err.txt; exit 1
    fi
  done
  echo "$(date +%H:%M:%S) start accepted, waiting for the instance to run"
  aws ec2 wait instance-running --instance-ids "$INSTANCE" --region "$REGION"
  ADDRESS=$(aws ec2 describe-instances --instance-ids "$INSTANCE" --region "$REGION" \
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
  HOST="$USER_NAME@$ADDRESS"
  echo "$(date +%H:%M:%S) instance running at $ADDRESS; waiting for ssh"
  until ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$HOST" true 2>/dev/null; do sleep 10; done
fi
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$HOST")

echo "$(date +%H:%M:%S) starting $NAME on $HOST"
# A non-interactive shell skips .bashrc, so uv and the NCBI key are not there by default:
# put uv's install dirs on the PATH and import the key line from .bashrc before the job.
SETUP='export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; eval "$(grep -E "^export NCBI_API_KEY=" ~/.bashrc 2>/dev/null)"; cd ~/aitx-agent'
if [ -z "${ATTACH:-}" ]; then
  # setsid detaches the job from the ssh session and </dev/null closes its input, so this
  # ssh call returns at once instead of holding the connection open for the job's lifetime.
  "${SSH[@]}" "$SETUP && mkdir -p logs && rm -f logs/$NAME.done && \
    setsid nohup bash -c '$SETUP; $JOB; echo \$? > logs/$NAME.done' > logs/$NAME.log 2>&1 < /dev/null &"
else
  echo "$(date +%H:%M:%S) ATTACH set: not starting the job, polling for logs/$NAME.done"
fi

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

if [ "$EXIT" = "0" ]; then
  aws ec2 stop-instances --instance-ids "$INSTANCE" --region "$REGION" >/dev/null
  echo "$(date +%H:%M:%S) stop requested for $INSTANCE"
else
  echo "$(date +%H:%M:%S) job failed (exit $EXIT); instance left running for a retry. Last log lines:"
  tail -n 5 "$LOCAL_DIR/$NAME.log"
fi
exit "$EXIT"
