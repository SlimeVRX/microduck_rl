# E0 — Đo baseline trước khi train: policy walking hiện tại, A/B low-pass runtime

Kết quả đo, không phải kế hoạch. Mọi con số dưới đây sinh ra từ
`scripts/eval_battery.py` trên `alpha_walking.onnx` (ONNX đang deploy, lấy từ repo
`microduck`), chạy headless MuJoCo trên CPU.

Đọc tài liệu này cùng `docs/agility-breakthrough-plan.md`: E0 là bước "đo
trước, không train" của kế hoạch đó, và mục §Quyết định ở cuối chốt lại trụ 2
(delay-aware) bằng bằng chứng chứ không bằng suy luận.

## 1. Thiết lập — và tại sao nó được coi là "giống deploy"

| Hạng mục | Giá trị | Nguồn |
| --- | --- | --- |
| Policy | `alpha_walking.onnx`, `obs[1,61] → actions[1,14]` | repo `microduck`, `policies/` |
| Scene | `src/mjlab_microduck/robot/microduck/scene.xml` | repo này |
| Vòng điều khiển | 50 Hz, decimation 4 | AGENTS.md; `robotd` |
| `action_scale` | 0.9 (walking) | `robotd/src/control.rs`, `Tuning::default()` |
| Arm "filtered" | `legs_lowpass=0.7`, `head_lowpass=0.5` | mặc định runtime |
| Arm "unfiltered" | filter tắt hoàn toàn | `alpha >= 1.0` ⇒ `None` |

Harness bám đúng đường điều khiển của runtime ở bốn điểm dễ sai nhất:
`last_action` đưa vào observation là **action thô** (không phải action đã lọc);
EMA áp lên **target vị trí** với `previous` là target **đã lọc** của tick trước;
thứ tự 61D giữ nguyên; và mouth không nằm trong 14 action.

**Hai giới hạn phải nói trước.** (a) `scene.xml` dùng actuator `position` kp=0.55,
không phải BAM/backlash như lúc train — nên **giá trị tuyệt đối không phải cam kết
trên robot thật**, chỉ có *delta giữa hai arm* là đáng tin. (b) `alpha_stand.onnx`
**ngã trong 2,6–5,5 s** ở scene này với mọi tổ hợp `standing_gain_ratio` ∈ {0.8, 1.0}
× `action_scale` ∈ {0.9, 1.0} × filter on/off. Harness có sẵn `--stand-policy` để tái
hiện scheduler thật (đổi net + gain khi `|twist| <= 0.05`), nhưng vì stand net không
đứng nổi trên model đơn giản này, toàn bộ số dưới đây **chỉ dùng walking net**. Đây là
khoảng cách của harness, không phải kết luận về stand net.

## 2. Số đo

Lệnh so sánh: `filtered` (mặc định runtime) → `unfiltered` (tắt filter).

### M1 — Trần tốc độ

| Lệnh | filtered | unfiltered |
| --- | --- | --- |
| vx 0.2 | 0.155 | 0.199 |
| vx 0.4 | 0.180 | 0.219 |
| vx 0.8 | 0.229 | 0.269 |
| vy 0.4 | **0.022** | **0.013** |
| wz 1.0 | 0.357 | 0.385 |
| wz 2.0 | 0.413 | 0.355 |

(m/s và rad/s, đo bằng **dịch chuyển/thời gian** chứ không phải vận tốc tức thời.)

Ba điều, theo thứ tự quan trọng:

1. **Trần tốc độ không nằm ở filter.** Tắt filter chỉ mua thêm ~17% (0.23 → 0.27 m/s).
   Cả hai arm đều **bão hoà quanh 0.2–0.27 m/s** dù lệnh lên tới 0.8: lệnh 0.8 và lệnh
   0.4 cho ra gần như cùng một tốc độ. Trần này là của **policy + cơ thể**, không phải
   của bộ lọc.
2. **Đi ngang gần như không tồn tại.** Lệnh 0.4 m/s ngang cho ra 0.01–0.02 m/s ở cả hai
   arm — dưới cả nhiễu. Slot `vy` trong obs đang gần như là input chết ở hành vi thật.
3. **Xoay bão hoà ở ~0.4 rad/s** trong khi lệnh tới 2.0 rad/s. Không ngã, chỉ đơn giản
   là không xoay nhanh hơn.

### M2 — Thời gian đáp ứng

| | filtered | unfiltered |
| --- | --- | --- |
| bắt đầu đi (từ nghỉ) | *không đo được* | 0.20 s |
| đảo chiều 0.4 → −0.4 | *không đo được* | *không đo được* |
| bắt đầu xoay | ≤ 0.04 s | ≤ 0.04 s |

`rise_window_s = 0.5` là **sàn phân giải**, nên 0.04 s chỉ có nghĩa "nhanh hơn mức
harness phân giải được", không phải "đo được là 40 ms".

*Không đo được* là kết quả có ý nghĩa, không phải lỗi: walking net **không đứng yên khi
lệnh bằng 0** — nó bước tại chỗ và trôi (xem M6), nên chuyển tiếp nghỉ→đi và
0.4→−0.4 không có mốc "trước" sạch để so. Phiên bản đầu của harness báo 0.0 s cho
những ca này; đó là artifact của việc đo vận tốc tức thời của thân đang lắc, và tôi đã
thay bằng `nan` có chủ đích thay vì để nó đọc như "đáp ứng tức thời". Muốn đo đúng
rise-from-rest thì cần stand→walk hot-swap chạy được — tức là cần model actuator trung
thực hơn (§1b).

### M3 — Chịu nhiễu (impulse vận tốc lúc đang đi 0.3 m/s)

| | filtered | unfiltered |
| --- | --- | --- |
| survival rate | 0.70 | 0.70 |
| mọi hướng đều sống | ≤ 0.4 | ≤ 0.4 |
| ngã đầu tiên tại | 0.6 (ngang) | 0.6 (ngang) |

Tổng thể **bằng nhau**, nhưng *chỗ* nó chết thì khác: filtered ngã ở **mọi** cú đẩy
ngang ≥ 0.6; unfiltered sống được đẩy ngang +0.6 nhưng lại ngã ở đẩy trước 1.0. Trục
ngang là trục yếu của cả hai — cùng một trục mà M1 nói robot không đi ngang được.
`recovery_tilt_max` của các ca sống đều 3–17°, tức là khi sống thì phục hồi gọn.

### M4 — Gait động hay quasi-static

| vx = 0.4 | filtered | unfiltered |
| --- | --- | --- |
| double support | 2.7% | 3.3% |
| single support | 94.3% | 86.7% |
| **flight** | **3.0%** | **10.0%** |
| step freq | 1.70 Hz | 2.60 Hz |
| foot clearance | 16.6 mm | 15.8 mm |

Đây là chỗ filter thể hiện rõ nhất: bỏ filter làm **flight fraction ×3.3** và **nhịp
bước +53%**. Nghĩa là dải tần bị EMA cắt đúng là dải tạo ra pha bay và nhịp nhanh.

### M5 — Giá phải trả về vật lý (đi 0.4 m/s)

| | filtered | unfiltered |
| --- | --- | --- |
| torque saturation | **0.0%** | **9.3%** |
| slip p95 | 0.17 | 0.33 |
| \|a_z\| p95 | 32.4 | 35.6 |
| action rate | 13.5 | 16.3 |

Cùng xu hướng ở xoay (sat 1% → 10%) và idle (0.3% → 9.7%). Tắt filter mua tốc
độ/nhịp bằng cách **đẩy servo vào bão hoà ~10% số tick** và **gấp đôi trượt chân**.

### M6 — Đứng yên (lệnh 0)

| | filtered | unfiltered |
| --- | --- | --- |
| trôi | 0.111 m/s | 0.137 m/s |
| trôi yaw | 0.287 rad/s | 0.373 rad/s |
| tilt trung bình | 2.9° | 2.9° |
| ngã | không | không |

Không ngã, tilt rất nhỏ (~3°) — nhưng **trôi 0.11 m/s và 0.29 rad/s khi lệnh bằng 0**,
tức là gần một nửa tốc độ đi tối đa của chính nó. Trên robot thật runtime che việc này
bằng cách chuyển sang stand net dưới ngưỡng 0.05; nhưng đây là bằng chứng số cho thấy
walking net **không có trạng thái nghỉ**.

## 3. Quyết định — chốt trụ 2 (delay-aware)

Mâu thuẫn mà tài liệu chẩn đoán nêu ra (AGENTS.md: *"Policies are UNFILTERED"* vs
runtime mặc định 0.7/0.5) **đã có câu trả lời từ chính source runtime**:

> `robotd/src/control.rs`: *"The alpha policies are trained with 0.5 — it must match
> training or transfer degrades."* … *"Same, for the ten leg joints. Trained with 0.7."*

Vậy `alpha_*` **được train CÓ filter**. Suy ra hai kết luận, và chúng đổi hướng trụ 2:

1. **Không phải bug ở stack alpha.** Với `alpha_*`, arm "unfiltered" trong E0 mới chính
   là arm **lệch** train/deploy — và giá của việc lệch đó đo được: saturation 0% → 9.3%,
   slip ×2. Nói cách khác, mismatch *nhanh hơn* nhưng trả bằng bão hoà servo.
2. **Nhưng nó là bẫy đang mở cho repo này.** Bất kỳ policy train mới ở `microduck_rl`
   hôm nay đều **không có filter**, trong khi runtime **mặc định có**. Deploy một policy
   mới với mặc định hiện tại là tái tạo đúng cái mismatch trên.

⇒ **Chốt: đưa EMA của runtime vào training loop** (α=0.7 chân / 0.5 đầu, cùng với delay
đã có), thay vì tắt filter ở runtime. Lý do là bằng chứng M5: trên model này filter mua
biên an toàn actuator rất lớn với chi phí ~17% tốc độ; và alpha đã chứng minh đường
"train có filter" chạy được trên hardware. Kèm theo: `legs_lowpass`/`head_lowpass` phải
trở thành **thuộc tính của từng policy** (ghi cạnh file ONNX), không phải một mặc định
toàn cục — vì hai họ policy trong repo hiện có hai transfer function khác nhau.

## 4. E0 đổi gì trong kế hoạch

Ba số làm thay đổi thứ tự ưu tiên:

- **Trần tốc độ/xoay là bão hoà, không phải trần bị lọc** (M1). Nới filter không mở được
  trần này ⇒ trụ 3 (priorless: curriculum theo năng lực, bỏ trần tốc độ do người đặt)
  quan trọng hơn tôi ước lượng ban đầu, và nên đo lại M1 sau mỗi stage curriculum.
- **Đi ngang ≈ 0 và trục ngang là trục ngã** (M1 + M3). Hai bằng chứng độc lập trỏ về
  cùng một chỗ: thiếu DOF ngang (ankle roll) và support polygon hẹp. Đây là hạng mục
  **phần cứng**, không phải reward — và nó xác nhận phần "RL không mua được" của tài liệu
  chẩn đoán.
- **Walking net không có trạng thái nghỉ** (M6 + M2). Trôi 0.11 m/s ở lệnh 0 là dấu hiệu
  policy không biết mình đang ở trạng thái nào — đúng lập luận cho trụ 1 (memory/GRU), và
  cho thấy `zero_command_prob` hiện tại chưa đủ để tạo một trạng thái nghỉ thật.

**Không** có bằng chứng nào ở E0 biện minh cho việc đổi contract 61D, thêm obs
current/contact, hay bắt đầu train dài. Bước tiếp theo theo kế hoạch vẫn là E1 (task
agile, chỉ trụ 3, vẫn MLP, vẫn 61D), cộng thêm một việc mà E0 vừa làm rõ là bắt buộc:
**train có EMA khớp runtime**, và một baseline MLP-no-symmetry để về sau so với GRU cho
công bằng.

## 5. Cách chạy lại

```bash
# arm mặc định runtime
python scripts/eval_battery.py \
  --policy /path/to/alpha_walking.onnx \
  --legs-lowpass 0.7 --head-lowpass 0.5 \
  --out filtered.json

# arm tắt filter (alpha >= 1.0 ⇒ None)
python scripts/eval_battery.py \
  --policy /path/to/alpha_walking.onnx \
  --legs-lowpass 1.0 --head-lowpass 1.0 \
  --out unfiltered.json

# tuỳ chọn: tái hiện scheduler stand↔walk của runtime
python scripts/eval_battery.py --policy alpha_walking.onnx \
  --stand-policy alpha_stand.onnx --only m6_idle_stability
```

`--only <metric_id>` chạy một phần battery. Toàn bộ 6 metric mất ~30–40 phút/arm trên
CPU.
