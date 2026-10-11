#!/bin/zsh
# GPU queue for the remaining seeds, each followed by its test evaluation. One job at a time
# (16 GB Macs). Safe to rerun after an interruption: finished steps are skipped and
# interrupted ones resume from their last epoch. Waits for any training already running.
#
#   nohup caffeinate -is zsh scripts/run_queue.sh > runs/queue.log 2>&1 &            # all steps
#   nohup caffeinate -is zsh scripts/run_queue.sh frcnn:1 frcnn:2 > runs/queue.log 2>&1 &
cd "${0:A:h}/.."
P=${PYTHON:-$(command -v python)}
YOLO=${P:h}/yolo
log() { print "[$(date '+%a %H:%M')] $*" }

while pgrep -f "train_frcnn.py|yolo detect train" > /dev/null; do sleep 60; done

yolo_seed() {  # same settings as runs/detect/full (seed 0); Ultralytics nests the output dir
  local d=runs/detect/yolo_s$1 real=runs/detect/runs/detect/yolo_s$1
  [[ -e $d ]] || ln -sfn runs/detect/yolo_s$1 $d
  if [[ -f $real/results.csv && $(($(wc -l < $real/results.csv) - 1)) -ge 100 ]]; then
    log "yolo seed $1 already done"
  elif [[ -f $real/weights/last.pt ]]; then
    log "yolo seed $1 resuming"; $YOLO detect train resume model=$real/weights/last.pt
  else
    log "yolo seed $1 start"
    $YOLO detect train model=yolov8s.pt data=caries.yaml epochs=100 patience=100 batch=16 \
      imgsz=640 device=mps workers=8 seed=$1 deterministic=True project=runs/detect name=yolo_s$1
  fi
  evaluate yolo $real/weights/best.pt yolov8s_s$1
}

frcnn_seed() {
  local d=runs/frcnn/frcnn_s$1
  mkdir -p runs/frcnn
  if grep -qs "^done" runs/frcnn/frcnn_s$1.log; then
    log "frcnn seed $1 already done"
  elif [[ -f $d/weights/last.pt ]]; then
    log "frcnn seed $1 resuming"
    $P scripts/train_frcnn.py --name frcnn_s$1 --seed $1 --batch 1 --resume >> runs/frcnn/frcnn_s$1.log 2>&1
  else
    log "frcnn seed $1 start"
    $P scripts/train_frcnn.py --name frcnn_s$1 --seed $1 --batch 1 > runs/frcnn/frcnn_s$1.log 2>&1
  fi
  evaluate frcnn $d/weights/best.pt frcnn_s$1
}

evaluate() {  # $1 model, $2 weights, $3 name
  [[ -f runs/eval/$3_test.json ]] && return
  log "evaluating $3"; $P scripts/evaluate.py $1 $2 --name $3 2>&1 | grep -E '"(mAP50|f1)"'
}

if (( $# )); then steps=($@); else steps=(yolo:1 yolo:2 frcnn:1 frcnn:2); fi
for s in $steps; do
  case ${s%%:*} in
    yolo) yolo_seed ${s#*:} ;;
    frcnn) frcnn_seed ${s#*:} ;;
  esac
done
log "queue complete: $steps"
