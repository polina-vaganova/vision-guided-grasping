#!/bin/bash
# Запуск эксперимента N сценариев с ПЕРИОДИЧЕСКОЙ ПОЛНОЙ ЧИСТКОЙ Gazebo.
# Каждый запуск контейнера = BATCH_SIZE сценариев в одном sim.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOTAL_SCENARIOS=${1:-30}
BATCH_SIZE=${2:-2}     # 2 за один sim (быстрее, меньше шанс залипания)

YAML="$ROOT/ros2_ws/src/pick_place_bringup/config/pick_place_params.yaml"
CSV="$ROOT/experiments/scenarios.csv"

echo "=== Эксперимент: $TOTAL_SCENARIOS сценариев, по $BATCH_SIZE на инстанс ==="

# Установить n_scenarios = BATCH_SIZE
sed -i '' "s/^    n_scenarios:.*/    n_scenarios:       $BATCH_SIZE/" "$YAML"

initial_count=$(wc -l < "$CSV")
target_count=$((initial_count + TOTAL_SCENARIOS))

batch_n=0
while true; do
  cur=$(wc -l < "$CSV")
  done_count=$((cur - initial_count))
  if [ $done_count -ge $TOTAL_SCENARIOS ]; then
    break
  fi
  batch_n=$((batch_n + 1))
  echo ""
  echo "--- Batch $batch_n: всего сделано $done_count / $TOTAL_SCENARIOS ---"

  # Полная пересборка контейнера
  bash "$ROOT/run_ros2.sh" > /dev/null 2>&1

  # Ждём готовности pipeline (макс 90 сек)
  for i in $(seq 1 30); do
    if docker exec ros2-gazebo bash -c "source /opt/ros/jazzy/setup.bash && ros2 service list 2>/dev/null | grep -q attach_object" 2>/dev/null; then
      echo "  pipeline ready"
      break
    fi
    sleep 3
  done

  # Ждём пока добавится BATCH_SIZE строк ИЛИ 6 минут таймаут
  before_batch=$(wc -l < "$CSV")
  for i in $(seq 1 120); do
    sleep 3
    cur=$(wc -l < "$CSV")
    delta=$((cur - before_batch))
    if [ $delta -ge $BATCH_SIZE ]; then
      echo "  batch $batch_n: $delta scenarios done"
      break
    fi
  done
done

final=$(wc -l < "$CSV")
echo ""
echo "=== Готово: $((final - initial_count)) сценариев в этой сессии ==="
