# E1 — task Agile

## Động lực từ E0

E0 cho thấy policy alpha đi được 0.2291 m/s ở lệnh tiến 0.8, gần như không
khác lệnh 0.4; đây là bão hoà năng lực chứ không phải giới hạn của filter.
Vì vậy Agile bắt đầu với command range vừa năng lực và mở rộng theo curriculum.
Điều này bị bác bỏ nếu các stage nhỏ vẫn bão hoà ở cùng mức hoặc stage lớn không
cải thiện tốc độ đạt được.

Đi ngang chỉ đạt 0.0217 m/s ở lệnh 0.4, và cả bốn nhóm push làm policy ngã đều
ở trục ngang. Agile giảm biên độ `lin_vel_y` còn 0.6 lần biên độ tiến trong
curriculum. Điều này bị bác bỏ nếu lateral tracking vẫn xấp xỉ bằng không sau
khi đã được học với command phù hợp.

Gait đã lọt vào cấu trúc 94.3% single support, 3.0% flight, 1.698 Hz; thân
cũng bị ràng buộc bởi upright mạnh. Agile thay cửa sổ air-time cố định bằng
`air_time_adaptive`, giảm upright từ 2.0 xuống 1.0 và giảm các motion blocker.
Điều này bị bác bỏ nếu gait vẫn giữ đúng cấu trúc cũ hoặc adaptive timing làm
gait mất ổn định.

Ở E0, torque saturation là 0% với filter runtime và 9.3% khi tắt filter;
slip p95 cũng tăng từ 0.1702 lên 0.3275 trong bài đi 0.4 m/s. Agile đưa EMA
runtime vào target trong training: alpha 0.7 cho mười joint chân và 0.5 cho
bốn joint đầu/cổ. `last_action` trong observation vẫn là output raw của
policy. Điều này bị bác bỏ nếu policy train có EMA nhưng battery chạy với
filter runtime 0.7/0.5 vẫn tái tạo mismatch actuator hoặc không chuyển giao.

## Cách chạy

Chưa có gì được train. Session này không có GPU; các lệnh dưới đây chạy trên
máy GPU.

Smoke test bắt buộc:

```bash
uv run train Mjlab-Agile-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5
```

Run dài:

```bash
uv run train Mjlab-Agile-Flat-MicroDuck \
  --env.scene.num-envs 4096
```

Export checkpoint đã train (thay placeholder bằng run thật):

```bash
uv run scripts/export.py Mjlab-Agile-Flat-MicroDuck \
  --wandb-run-path <entity/project/run_id>
```

Chạy lại cùng E0 battery trên ONNX đã export:

```bash
python scripts/eval_battery.py \
  --policy <path/to/exported_agile.onnx> \
  --legs-lowpass 0.7 \
  --head-lowpass 0.5 \
  --out e1-agile-filtered.json
```

Battery phải chạy **có filter** `--legs-lowpass 0.7 --head-lowpass 0.5`, vì
policy Agile được train với transfer function này. So sánh cùng sáu metric
E0: M1 speed envelope, M2 step response, M3 disturbance, M4 gait dynamics,
M5 cost và M6 idle stability.
