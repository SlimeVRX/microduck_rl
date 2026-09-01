# Đột phá lần hai: kiến trúc chuyển động cho microduck

> Tài liệu này để **review trước khi implement**. Nó trả lời hai câu: (1) *nguyên lý* nào — không phải tính năng nào — khiến microduck vượt Open Duck Mini v2; (2) áp dụng lại đúng nguyên lý đó thì kiến trúc tiếp theo phải là gì.
> Đi kèm: [`agility-servo-locomotion-design.md`](agility-servo-locomotion-design.md) (chẩn đoán giới hạn + số đo cơ khí/actuator).

Nhãn nguồn: `[MD_RL]` repo này · `[MD]` runtime `microduck` · `[ODM]` `Open_Duck_Mini` · `[LIB]` mã nguồn thư viện (rsl_rl 5.0.1 / mjlab 1.3.0, đã đọc trực tiếp) · `[SUY LUẬN]` suy luận của tôi · `[NGOÀI REPO]` ngoài code.

---

## Phần 1 — Nguyên lý: vì sao microduck sống, còn robot servo khác thì máy móc

Cảm nhận của bạn ("không còn dấu hiệu máy móc") không phải cảm tính — nó có nguyên nhân kỹ thuật đo được, và nguyên nhân số một không phải "microduck dùng RL". **Cả hai đều dùng RL.** Khác biệt nằm ở chỗ *ai phát minh ra dáng đi*.

### P1. microduck không bắt chước con người viết ra dáng đi — nó tự phát minh

- `[ODM]` `docs/sim2real.md:39-41`: Open Duck Mini v2 dùng **imitation reward kiểu BDX của Disney**, và imitation reward đó **cần reference motion**, sinh ra bởi một **parametric walk engine** (`Open_Duck_reference_motion_generator` → `polynomial_coefficients.pkl`). Trước đó là cả một walk engine ZMP/PlaCo (`mini_bdx/placo_walk_engine/`, `get_current_support_phase()`).
- `[ODM]` `experiments/RL/new/simple_env.py:35`: observation có **`clock signal`** — nhịp bước được *đưa vào* cho policy.
- `[MD_RL]` Trong toàn bộ `microduck_velocity_env_cfg.py`: **không có một chữ `clock` hay `phase` nào.** `grep` trả về 0. Docstring của roulade nói thẳng triết lý: *"no phase clock, no reference motion"*.

Đây là điểm cốt lõi. Một policy bắt chước reference motion **bị chặn trên bởi chất lượng của reference đó**. Reference motion sinh từ walk engine ZMP là chuyển động chuẩn tắc: quasi-static, CoM trên polygon đỡ, nhịp cố định, đối xứng hoàn hảo — nghĩa là **đúng cái mà mắt người đọc ra là "máy móc"**. RL học nó rất tốt, và học luôn cả tính máy móc của nó. Còn microduck chỉ được cho *mục tiêu* (bám vận tốc, đứng thẳng, chân không trượt) và bị buộc tự tìm cách — nên cái nó tìm ra là chuyển động do vật lý của **chính cơ thể nó** quyết định, không phải do một mô hình toán của con người quyết định. Chuyển động sinh ra từ vật lý của một cơ thể thì trông như sinh vật, vì sinh vật cũng vậy.

`[SUY LUẬN]` Đó là lý do "hầu hết robot servo hiện nay chỉ ngang cấp ODM v2": gần như tất cả đều đang phát lại quỹ đạo do người thiết kế — scripted keyframe (Robi, PLEN2 và hobby humanoid nói chung `[NGOÀI REPO]`), hoặc walk engine ZMP, hoặc RL bắt chước walk engine. Ba mức khác nhau về công nghệ, **nhưng cùng một trần**: dáng đi vẫn là ý tưởng của con người.

### P2. Không có khoảng cách để bù — servo được *định danh*, không được *lý tưởng hoá*

`[MD_RL]` `microduck_constants.py`: actuator là BAM điều khiển **theo điện áp** cho XL330 (`kp_fw=200`), có **sụt áp theo tải** (`vin_range=(6.5,8.2)`, `vin_drop_gain_range=(0,0.2)`, `vin_min=6.0`), **trễ 3–6 bước sim** (dt=0.005 ⇒ 15–30 ms), **ma sát Coulomb/Stribeck/theo tải** do actuator tự tính, và **backlash 2° với encoder nằm sau khe hở** (`BacklashEncoderBamActuator` — firmware đọc `qpos_main + qpos_backlash`, đúng vật lý servo thật).

Ý nghĩa: policy không học trong một thế giới có servo hoàn hảo rồi ra ngoài đời "cố sống sót". Nó đã luyện *với con servo thật*, gồm cả lúc servo yếu đi đúng lúc cần nhất. `[SUY LUẬN]` Cái mà mắt ta đọc là "tự tin", "có chủ đích" chính là **sự vắng mặt của hành vi bù sai** — không có giật, không có dao động sửa lỗi, không có do dự.

`[ODM]` Open Duck Mini v2 cũng dùng định danh BAM — nên đây là điểm *chung*, không phải điểm hơn. Điểm hơn là P2 được đặt **dưới** một policy reference-free (P1): định danh actuator tốt chỉ có ý nghĩa khi policy được tự do khai thác nó.

### P3. Không có điểm gián đoạn nào trong toàn hệ

`[MD]` `last_action` nằm trong obs (14 slot) ⇒ mỗi lệnh là phần tiếp nối của lệnh trước, không phải một quyết định độc lập. Scheduler chuyển kỹ năng theo ưu tiên nhưng mọi ranh giới là **hằng số thời gian**, kể cả lúc ngã (limp-fall thả moment theo thời gian, không snap). `[SUY LUẬN]` Sinh vật không có frame nào bị "cắt". Đây là lý do cảm giác sống *không* biến mất ở các đường chuyển trạng thái — chỗ mà mọi robot scripted đều lộ ra ngay.

### P4. Toàn thân là một chuyển động

`[MD_RL]` `head_pose` (4 chiều) nằm **trong** observation của policy đi, weight 2.0. Đầu không phải một cơ cấu riêng chạy song song — nó là một phần của bài toán đi. `[SUY LUẬN]` Với cái đầu chiếm **38% khối lượng**, đây là bắt buộc về vật lý, và cũng là lý do gaze và bước chân trông như *một* ý định thay vì hai hệ thống.

### P5. Kỷ luật đo lường — nguyên lý meta khiến bốn cái trên hoạt động

`[MD_RL]` AGENTS.md và các docstring cfg là một bộ sưu tập bài học rất đắt: *"RL optimizes the letter of the reward"*, *"Measure before theorizing"*, *"every `Episode_Reward/<penalty>` must be ≤ 0"*, *"a 25 cm robot tumbles at 3.5–5.5 rad/s NATURALLY — don't impose human-scale intuitions"*. Không có kỷ luật này thì P1 (bỏ reference) chỉ tạo ra reward hacking, chứ không tạo ra dáng đi.

### Bất biến của đột phá — công thức để lặp lại

> **Mỗi lần bỏ được một prior của con người, và đồng thời đóng được một khoảng cách mô hình, robot sống thêm một bậc.**

Bỏ prior mà không đóng khoảng cách ⇒ policy hack reward hoặc chết ở robot thật. Đóng khoảng cách mà không bỏ prior ⇒ ta có một walk engine chạy rất ổn (ODM v2). Phải làm **cả hai cùng lúc**. Đó chính xác là điều đội microduck đã làm — và là điều tôi đề xuất làm lại lần nữa dưới đây.

---

## Phần 2 — Prior nào còn sót lại, và khoảng cách nào còn mở

**Prior của con người vẫn còn hard-code trong công thức đi** `[MD_RL]` (`microduck_velocity_env_cfg.py`):

| Prior còn sót | Giá trị | Nó đang quyết định thay policy điều gì |
|---|---|---|
| `air_time` window | 0.125–0.300 s, w=3.0 | **nhịp bước** — chính là cái clock signal của ODM v2, chỉ ở dạng reward |
| `foot_clearance` / `foot_swing_height` | target 0.02 m | **độ nhấc chân** |
| `upright` | w=2.0, std²=0.05 | **thân phải thẳng** — cấm lean khi tăng tốc/quay |
| `action_rate_l2` | ramp tới **−1.0** @iter1500 | **trần băng thông** của policy |
| `twist` ranges | `vx ±0.4`, `vy ±0.3`, `ωz ±1.0`, cố định | **trần tốc độ** do config, không do vật lý |
| `body_ang_vel`, `angular_momentum` | −0.05, −0.02 | cấm đúng thứ humanoid dùng để nhanh |
| `body_pose` (6 slot lệnh) | **weight 0** | 6 kênh điều khiển đang bỏ trống |

**Khoảng cách mô hình còn mở:**

1. **Policy không có ký ức.** Actor là MLP `(512,256,128)` thuần feed-forward `[MD_RL]`: mỗi tick nó nhìn 61 số và quyết định, **không biết gì về chính cơ thể mình hôm nay** — ma sát khớp đang cao hay thấp, pin đang tụt bao nhiêu, backlash đang ở phía nào, chân vừa chạm đất hay chưa. Domain randomization dạy nó *chịu đựng* mọi trường hợp; nó không cho phép *nhận ra* trường hợp nào đang xảy ra. Đây là khoảng cách lớn nhất còn lại. `[ODM]` ODM v2 từng thử RMA/adaptation — nhưng dưới một policy imitation, nên trần vẫn là reference.
2. **Actor mù tiếp xúc.** contact / air_time / `base_lin_vel` chỉ có ở critic `[MD_RL]`. Trong khi `present_current` của mọi servo **đã được đọc mỗi tick** trên bus `[MD]` (`duck-control/src/bus.rs:282-283`) — tín hiệu tải đang có sẵn và bị bỏ.
3. **Train ≠ deploy ở dải tần.** AGENTS.md: *"Policies are UNFILTERED"*; runtime mặc định `legs_lowpass=0.7`, `head_lowpass=0.5` `[MD]` (`robotd-params/src/lib.rs:655-656`) ⇒ `[SUY LUẬN]` fc ≈ 9,6 Hz / 5,5 Hz ở 50 Hz. Policy đang bị lọc đúng dải tần mà nó chưa từng thấy khi train.

---

## Phần 3 — Kiến trúc đề xuất: **MDP-Agile** (Memory · Delay-aware · Priorless)

Ba trụ, đúng công thức bất biến: trụ 1–2 đóng khoảng cách mô hình, trụ 3 bỏ prior. **Không được làm riêng lẻ.**

### Trụ 1 — MEMORY: policy có ký ức, mà **không phá contract 61D**

Đây là phát hiện quan trọng nhất của lần điều tra này. Tôi đã đọc mã thư viện `[LIB]`:

- `rsl_rl 5.0.1` có `RNNModel` (GRU/LSTM) đầy đủ: `rsl_rl/models/rnn_model.py`, PPO có `recurrent_mini_batch_generator` (`algorithms/ppo.py:222-223`).
- `mjlab 1.3.0` đã expose sẵn: `RslRlModelCfg.class_name="RNNModel"`, `rnn_type`, `rnn_hidden_dim`, `rnn_num_layers` (`mjlab/rl/config.py:31-37`), và runner strip đúng field khi không dùng (`mjlab/rl/runner.py:28-31`).
- **ONNX export của GRU giữ nguyên obs 61D**: `_OnnxRNNModel.forward(obs, h_in) -> (actions, h_out)`, `input_names=["obs","h_in"]`, `output_names=["actions","h_out"]` (`rsl_rl/models/rnn_model.py:180-250`). Hidden state là **I/O của model**, không phải một phần của observation.

Nghĩa là: **contract 61D bất biến vẫn được giữ nguyên.** Runtime chỉ cần mang vector `h` từ tick này sang tick sau và zero nó khi hot-swap policy — một thay đổi nhỏ, khoanh vùng được, ở `[MD]` `duck-control/src/policy.rs` (chỗ `ort::inputs!["obs" => &input]` và `check_width`).

Ký ức mua được ba thứ mà 61 số của một tick không thể có:

1. **Cảm giác tiếp xúc, miễn phí.** Chân vừa chạm đất là một *mẫu thời gian* của joint_vel + previous_action + gyro. Một GRU đọc ra được; một MLP một-frame thì không. `[SUY LUẬN]` Đây là cách có "force sense" mà **không thêm sensor và không phá contract** — tốt hơn cả phương án current-based obs mà tôi đề xuất ở tài liệu trước (phương án đó phải bump obs v2).
2. **Tự định danh cơ thể (RMA ẩn).** Từ vài chục tick lịch sử, policy suy ra ma sát / sụt áp / backlash / tải hiện tại và **thích nghi thay vì chịu đựng**. Chính DR mạnh sẵn có (CoM, head CoM, mass/inertia, friction, armature, encoder bias, IMU misalignment) biến từ "nguồn nhiễu phải bền với" thành "tín hiệu phải nhận ra".
3. **Nền cho anticipation** (trụ 2) — không có ký ức thì không thể dự đoán.

**Ràng buộc đã kiểm chứng** `[LIB]` `ppo.py:104-105`: *"Symmetry augmentation is not supported for recurrent policies"*. Velocity env có symmetry mirror-loss ⇒ nhánh recurrent phải tắt symmetry. `[SUY LUẬN]` Điều này làm A/B bị nhiễu bởi hai biến ⇒ phải chạy thêm nhánh MLP-không-symmetry làm baseline sạch.

`[SUY LUẬN]` Chi phí CPU trên RK3566: GRU 61→256 ≈ 3·(61+256)·256 ≈ 0,24 MFLOP/tick, so với MLP (512,256,128) ≈ 0,2 MFLOP/tick — cùng cấp độ, không đe doạ vòng 50 Hz. Cần đo thật bằng `scripts/infer_policy.py`, không tin con số này.

### Trụ 2 — DELAY-AWARE: đóng khoảng cách băng thông bằng **dự đoán**, không bằng phản hồi

Đây là luận điểm trung tâm về "tiến gần BLDC hơn". BLDC humanoid nhanh vì vòng moment băng thông cao: nó *phản ứng*. microduck có 50 Hz + 15–30 ms trễ actuator + EMA ~10 Hz ở chân `[MD_RL]`/`[MD]`. **Phản ứng nhanh là bất khả thi về mặt vật lý** — và không có reward nào sửa được điều đó.

Con đường duy nhất còn lại: **hành động sớm**. Một policy có ký ức + mô hình nội tại về chính actuator của nó có thể phát lệnh *trước* khi sai số xuất hiện — feedforward thay cho feedback. `[SUY LUẬN]` Đây là cách một robot chậm về băng thông vẫn có thể *trông* nhanh và dứt khoát: nó không đợi để sửa, nó đã biết trước.

Cụ thể:
- Đưa **đúng transfer function của runtime vào training loop**: nếu deploy có EMA α=0.7 ở chân thì train có EMA α=0.7, cùng với trễ đã có. Mismatch ở §Phần 2 (3) chuyển từ bug thành **tài sản**: policy học pre-compensate cái lọc đó (phát lệnh vượt trước để sau lọc ra đúng ý định). Nếu chọn hướng ngược lại (tắt lọc ở runtime) thì phải kèm test transfer — đây là điểm quyết định cần đo ở E0, **không đoán**.
- Thêm tín hiệu chi phí *thật* của actuator vào reward (torque rate, bão hoà PWM/current) để policy học biên năng lực thay vì học một trần tốc độ do người đặt.

### Trụ 3 — PRIORLESS: bỏ nốt các prior dáng đi, thay bằng chi phí vật lý

Đúng đòn đã thắng một lần (P1), nay áp cho những gì còn lại ở Phần 2:

| Bỏ / nới | Thay bằng |
|---|---|
| `air_time` window cố định, `clearance` 2 cm | window rộng, clearance theo curriculum; nhịp bước là thứ policy **phải tự chọn** |
| `upright` cứng w=2.0 | upright phụ thuộc lệnh: nghiêm khi lệnh ≈ 0, nới khi có `vx`/`ωz` lớn (lean là **cần thiết** để tăng tốc) |
| `body_ang_vel`, `angular_momentum` | ≈ 0 trong pha khám phá; thay bằng **|a_z| impact, foot slip, torque rate, PWM saturation** — đúng bài học roulade/standup của repo |
| `action_rate` ramp tới −1.0 sớm | ramp muộn hơn và nhẹ hơn ở chế độ agile (smoothness chỉ vào **sau** khi kỹ năng đã hình thành) |
| `twist` range cố định | **curriculum theo năng lực**: mở range chỉ khi tracking error và tỉ lệ ngã dưới ngưỡng ⇒ hệ tự tìm ra biên vật lý của XL330 |
| `head_pose` tracking tức thời | tracking **hướng nhìn theo EMA** ⇒ đầu 280 g trở thành **bánh đà phản lực có chủ đích** thay vì ràng buộc (kỹ thuật EMA repo đã dùng cho `head_pose_bias`) |
| `body_pose` 6 slot weight 0 | 1 slot làm **agility knob** liên tục (calm ↔ agile) — một policy, hai tính cách, không cần scheduler mới, và chuyển chế độ *liên tục* đúng triết lý P3 |

### Trụ 4 (dự phòng) — teacher→student nếu recurrent học chậm

Teacher privileged (contact thật, `base_lin_vel`, tham số DR — critic đã có sẵn hết) → distill vào student GRU chỉ có 61D proprio. Chỉ mở nếu E2 cho thấy PPO recurrent hội tụ quá chậm. Không làm trước.

### Vì sao gói ba trụ này *là* "gần BLDC hơn" — và chỗ nó không phải

Nói thẳng để không đặt mục tiêu sai: servo **không** có mật độ moment, băng thông kHz, backdrivability hay truyền động không khe hở của BLDC. Nhảy cao, chạy pha bay dài, tiếp đất hấp thụ mạnh — không nên là mục tiêu.

Nhưng "cảm giác BLDC" mà mắt người đọc được phần lớn **không** đến từ moment đỉnh; nó đến từ: không do dự (⇒ anticipation, trụ 2), dùng cả cơ thể để cân bằng thay vì cứng người (⇒ head-as-momentum + agility knob, trụ 3), và thích nghi tức thì với mặt sàn/tải (⇒ memory, trụ 1). `[SUY LUẬN]` Cả ba đều **không** cần thêm watt nào. Và servo còn có ba lợi thế bất đối xứng mà kiến trúc này khai thác thẳng: giữ tư thế gần như miễn phí (tỉ số truyền cao), đặt chân rất chính xác, và một khối lượng đầu 38% — thứ mà robot BLDC coi là tải, ở đây là **bánh đà**.

---

## Phần 4 — Kế hoạch thực hiện (đề xuất, chờ bạn duyệt)

Mỗi giai đoạn có điều kiện go/no-go. Ước lượng theo *session làm việc của tôi*, chưa tính thời gian train GPU.

**E0 — Đo trước, không train (≈1 session).** Battery eval 6 metric (§2 của tài liệu chẩn đoán) trên policy walking hiện tại → baseline bằng số. A/B `legs_lowpass` 0.7 vs 1.0 trên `scripts/infer_policy.py` (và trên robot nếu bạn chạy được) → **chốt** hướng trụ 2: train-có-lọc hay deploy-không-lọc.
*Go/no-go: có baseline số và một quyết định lọc dứt khoát. Không có ⇒ không đi tiếp.*

**E1 — Task `Mjlab-Agile-Flat-MicroDuck`, chỉ trụ 3 (≈1 session + 1 run).** Fork từ `make_microduck_velocity_env_cfg` (giữ toàn bộ DR / obs-noise / NaN-guard / `expand_bam_friction_fields`). Agility knob ở `body_pose`, gaze EMA, đổi cấu trúc regularizer, curriculum lệnh theo năng lực. Vẫn MLP, vẫn 61D. Cfg test (joint index trên model thật, dấu mọi penalty, gate). Smoke test 64 env/5 iter → run dài. Train cả biến thể `-Backlash-`.
*Go/no-go: battery E0 phải cải thiện ở ≥3/6 metric mà không tăng |a_z| đỉnh và bão hoà PWM.*

**E2 — Trụ 1: GRU (≈1 session + 1–2 run).** `class_name="RNNModel"`, `rnn_type="gru"`, symmetry tắt (ràng buộc `[LIB]`), kèm nhánh MLP-không-symmetry làm baseline sạch. Kiểm tra ONNX có `h_in`/`h_out`; đo latency CPU bằng `scripts/infer_policy.py`.
*Go/no-go: GRU thắng baseline trên battery **ở env backlash + DR mạnh** (nếu chỉ thắng ở env sạch = overfit sim, dừng lại).*

**E3 — Trụ 2 đầy đủ + runtime (≈1 session).** Lọc/trễ của runtime nằm trong training loop; runtime mang `h` qua các tick và zero khi switch policy (`policy.rs`), rehearse `scripts/infer_policy.py` trước khi chạm robot.
*Go/no-go: rehearsal CPU khớp sim; hidden state không drift sau vài phút chạy liên tục.*

**E4 — Phần cứng (theo tài liệu chẩn đoán).** Gót chân / backlash / CoM đầu; rồi đo ngân sách bus Dynamixel trước khi cân nhắc +2 ankle roll (16 servo) → +tay (18 servo).

**Footgun đã biết, đừng đạp lại** (từ AGENTS.md và các cfg): dấu penalty tự-phủ-định; jackpot reward; so *reward mass* chứ không so weight khi copy regularizer; siết std cho một dao động không thể tránh (đầu 38%); DR tích luỹ qua reset; đưa smoothness tax vào trước khi kỹ năng hình thành; và tin metric sim mà không xem video + kiểm geom/trục nào chạm đất.

---

## Phần 5 — Rủi ro và cách bác bỏ nhanh

| Rủi ro | Bác bỏ bằng |
|---|---|
| GRU học "đọc vị" mô phỏng thay vì đọc vị cơ thể | bắt buộc thắng trên env `-Backlash-` + DR ở mức cuối curriculum, không chỉ env sạch |
| Hidden state drift / bẩn sau hot-swap trên robot thật | zero `h` mỗi lần switch policy; chạy liên tục nhiều phút trong rehearsal CPU và đối chiếu quỹ đạo với sim |
| Mất symmetry loss ⇒ dáng đi lệch trái/phải | baseline MLP-không-symmetry để tách biến; nếu lệch, thêm penalty bất đối xứng ở qpos thay cho mirror loss |
| Bỏ prior ⇒ reward hacking (bài học đã có 2 lần ở standup/roulade) | gate theo trạng thái (contact, trục orientation, latch), không phải nudge bằng penalty; audit mọi `Episode_Reward/<penalty>` ≤ 0 |
| Nới `upright`/nhịp bước ⇒ gait chỉ đẹp trong sim, ngã ngoài đời | mọi metric của E1/E2 đều báo cáo trên env backlash; và luôn xem video, kiểm geom/trục chạm đất |
| Agility knob làm policy học hai chế độ nửa vời | sample zero-command tường minh + bucket riêng cho hai đầu knob (bài học `rel_turn_in_place_envs`: 2% kinh nghiệm ⇒ không bao giờ học được) |

---

## Phần 6 — Một câu

Đội microduck đột phá bằng cách **bỏ dáng đi do con người viết ra, đồng thời định danh đúng con servo thật**. Prior lớn nhất còn lại là *nhịp bước, độ nhấc chân, thân thẳng và trần tốc độ* vẫn do con người đặt; khoảng cách mô hình lớn nhất còn lại là *policy không có ký ức nên không biết cơ thể mình đang ở trạng thái nào*. Đóng cả hai cùng lúc — và mừng là ký ức **không phá contract 61D** (hidden state là I/O của ONNX, đã có sẵn trong rsl_rl 5.0.1 và mjlab 1.3.0) — là đề xuất của tôi cho đột phá lần hai.
