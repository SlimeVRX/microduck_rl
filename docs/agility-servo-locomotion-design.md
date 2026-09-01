# Thiết kế tối ưu cho chuyển động linh hoạt của robot servo

**Mục tiêu**: nâng mức độ linh hoạt khi di chuyển của microduck lên một cấp nữa — tiến gần cảm giác chuyển động của humanoid BLDC — bằng cách xác định đúng *giới hạn tiếp theo* và tấn công nó theo tầng chi phí, thay vì tinh chỉnh vài trọng số reward.

**Quy ước dẫn nguồn** (đọc kỹ trước khi dùng bất kỳ con số nào):

| Nhãn | Nghĩa |
|---|---|
| `[MD_RL]` | Kiểm chứng trực tiếp trong repo này (`microduck_rl`), có đường dẫn file |
| `[MD]` | Kiểm chứng trực tiếp trong repo runtime `microduck` |
| `[ODM]` | Kiểm chứng trong repo `Open_Duck_Mini` |
| `[SUY LUẬN]` | Suy luận vật lý/kỹ thuật của tôi từ số liệu đã kiểm chứng — chưa chạy thực nghiệm |
| `[NGOÀI REPO]` | Kiến thức thị trường/cơ khí bên ngoài (ROBOTIS MINI, DARWIN-MINI, ROBO-ONE). **Không** kiểm chứng được từ code; cần tra nguồn nhà sản xuất trước khi đưa vào quyết định kỹ thuật |

---

## 1. Chẩn đoán: giới hạn linh hoạt hiện tại nằm ở đâu

Không thể "phá giới hạn" nếu không biết giới hạn nào đang chặn trước. Dưới đây là bốn tầng, sắp theo mức độ chặn mà tôi đánh giá — mỗi tầng đều có số đo.

### 1.1 Cơ khí: cấu hình chân thiếu 2 bậc tự do quan trọng nhất cho agility

Đo trực tiếp từ MJCF `[MD_RL]` (`src/mjlab_microduck/robot/microduck/robot_walk.xml`):

| Đại lượng | Giá trị đo | Ý nghĩa |
|---|---|---|
| Tổng khối lượng model walk | **0.737 kg** | khớp với "~800 g" khi tính thêm dây/pin không mô hình hoá |
| Khối đầu (neck + neck_pitch + yaw_roll_motion + jaw_soft) | **0.280 kg = 38%** | trùng ghi chú trong cfg: "a 280 g head (38% of robot mass) MUST oscillate" |
| Thân (trunk_base) | 0.199 kg = 27% | |
| Mỗi chân (5 link) | 0.129 kg = 17,5% | hai chân 35% |
| Bàn chân (mesh `sole_left`) | ~**41 × 54 mm** | polygon đỡ cực nhỏ so với chiều cao đứng 0.117 m |
| `hip_roll` | **±22°** (±0.384 rad) | biên lateral rất hẹp |
| `hip_yaw` | −25° … +30° | |
| `hip_pitch`, `knee`, `ankle` (pitch) | ±90° | biên sagittal thoải mái |
| **`ankle_roll`** | **KHÔNG TỒN TẠI** | 5 DOF/chân: yaw, roll, pitch, knee, ankle-pitch |
| Tay/vai | **KHÔNG TỒN TẠI** | 14 servo = 2×5 chân + 4 cổ/đầu |

Hai kết luận cơ khí quan trọng:

1. **Không có ankle roll ⇒ không có thẩm quyền CoP theo trục ngang.** Toàn bộ cân bằng lateral phải đến từ `hip_roll` ±22° cộng với việc dịch chân. Ở một robot 25 cm, bàn chân rộng 41 mm, đây là điểm nghẽn cứng: mọi chuyển động cần *đứng một chân có kiểm soát* (đá, bước ngang thật, nghiêng người tránh, xoay nhanh) đều bị chặn ở tầng cơ khí, không phải ở tầng policy.
2. **Không có tay ⇒ không có bộ điều tiết động lượng góc.** Trong một humanoid, tay là "vô lăng" momentum: nó cho phép thân xoay/gia tốc mà không mất tiếp xúc chân. microduck hiện chỉ có một khối phản lực duy nhất: **cái đầu 280 g** — mà cái đầu đang bị RL đối xử như *ràng buộc phải bám lệnh*, không phải như *cơ cấu điều khiển* (xem §1.4).

`[NGOÀI REPO]` Đây chính là chỗ ROBOTIS MINI / DARWIN-MINI và các robot ROBO-ONE khác biệt về triết lý: chúng có **ankle roll** (bàn chân 2 DOF ⇒ điều khiển CoP hai trục) và **tay nhiều DOF** dùng cho cân bằng, chống đỡ khi ngã và làm "cánh" quán tính; bàn chân của robot ROBO-ONE thường được làm lớn có chủ đích để mở rộng polygon đỡ. *Các đặc điểm này cần xác nhận từ tài liệu ROBOTIS/luật ROBO-ONE trước khi dùng làm căn cứ thiết kế; tôi không xác minh được từ code.*

### 1.2 Actuator: XL330 + BAM — giới hạn đã được mô hình hoá khá trung thực

`[MD_RL]` `src/mjlab_microduck/robot/microduck_constants.py`:

```python
motor_name="xl330", model="m6", kp_fw=200.0,
vin_range=(6.5, 8.2), vin_drop_gain_range=(0.0, 0.2), vin_min=6.0,
delay_min_lag=3, delay_max_lag=6,
```

- Sụt áp theo tải đã có (`vin_drop_gain`) — tức mô hình đã biết rằng servo *mất moment đúng lúc cần nhất*.
- Trễ lệnh 3–6 bước sim; `[MD_RL]` ghi chú trong `robot_walk_backlash.xml` xác nhận `dt=0.005` cho velocity task ⇒ **15–30 ms**.
- Ma sát Coulomb/Stribeck/theo tải do BAM tự tính, có DR nhân hệ số (`friction_dr_bam.py`).
- Backlash: **2° tổng (±1°) mỗi khớp**, encoder nằm *sau* khe hở (`BacklashEncoderBamActuator`) — mô hình đúng vật lý servo thật.

`[SUY LUẬN]` Ước lượng ảnh hưởng backlash: ±1° tại một khớp với cánh tay đòn ~0.1 m ⇒ ~1,7 mm ở bàn chân; chuỗi 5 khớp cộng dồn (trường hợp xấu, không tương quan) ⇒ **~4–8 mm sai vị trí bàn chân — tương đương 10–20% chiều rộng bàn chân 41 mm**. Đây là lý do một gait "đẹp trong viewer" mất biên độ trên robot thật, và là lý do phiên bản `-Backlash-` không nên là biến thể phụ mà nên là môi trường train chính (robot thật *luôn* có backlash).

### 1.3 Runtime: có một mâu thuẫn train/deploy cần xác minh — và nó chặn đúng dải tần của agility

- `[MD_RL]` AGENTS.md, invariant: *"**Policies are UNFILTERED** (no action low-pass in training). Don't add EMA filtering without a matched runtime flag and a transfer test."*
- `[MD]` `robotd-params/src/lib.rs:655-656`: mặc định runtime là `head_lowpass = 0.5`, `legs_lowpass = 0.7` (EMA alpha), và test khoá giá trị `Some(0.7)`.

`[SUY LUẬN]` Với EMA `y ← αx + (1−α)y` ở 50 Hz: α=0.7 ⇒ tần số cắt ≈ **9,6 Hz**; α=0.5 ⇒ ≈ **5,5 Hz** (Nyquist là 25 Hz). Nghĩa là mọi thành phần lệnh nhanh — đúng thành phần tạo nên bước chân nhanh, phản ứng chống ngã, động tác nổ — bị suy giảm ở chân, và suy giảm mạnh hơn ở đầu. Nếu policy được train **không** filter mà deploy **có** filter, ta đang mất chính xác dải tần cần cho agility, và mất nó một cách vô hình (robot vẫn đi được, chỉ là "mềm" hơn bản trong sim).

**Đây là hạng mục kiểm tra số 1, chi phí gần bằng 0**: đo lại một policy hiện có với `legs_lowpass=1.0` (tắt filter, khớp training) vs mặc định 0.7, so sánh tracking error và biên độ khớp. Tôi *chưa* kết luận đây là bug — có thể alpha đã được chọn cố ý để bù backlash/nhiễu; nhưng hai repo đang phát biểu ngược nhau, và đó là điều phải giải quyết trước khi tối ưu bất cứ thứ gì khác.

### 1.4 Reward/observation: công thức hiện tại là "đi bộ ổn định", được thiết kế để *hạn chế* dynamic

Tất cả từ `[MD_RL]` `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`:

| Thành phần | Giá trị | Tác dụng lên agility |
|---|---|---|
| `upright` | weight 2.0, std²=0.05 | giữ thân *thẳng đứng* — chặn lean khi tăng tốc/quay nhanh |
| `body_ang_vel` | −0.05 | motion-blocker |
| `angular_momentum` | −0.02 | motion-blocker (đúng thứ humanoid dùng để nhanh) |
| `action_rate_l2` | curriculum −0.1 → **−1.0 ở iter 1500** | tax mạnh lên thay đổi lệnh nhanh |
| `air_time` | window **0.125–0.300 s**, w=3.0 | quy định nhịp bước, không cho pha bay ngắn/dài |
| `foot_clearance` / `foot_swing_height` | target **0.02 m** | nhấc chân 2 cm — dáng "lết" |
| lệnh `twist` | `vx ±0.4`, `vy ±0.3`, `ωz ±1.0`, **range cố định, không curriculum** | trần tốc độ do config, không do vật lý |
| push | ±0.3 m/s mỗi 3–6 s | ghi chú: đã *giảm* từ mức cao hơn vì gây "nervous fall-recovery gait" |
| terrain | bậc ≤1,5 cm, slope nhẹ | hợp lý vì clearance chỉ 2 cm |
| `body_pose` (6 slot lệnh) | **weight 0.0** — infra sống, reward tắt | **6 kênh lệnh đang bỏ trống** |
| `head_pose` | weight 2.0, std 0.5, range ramp tới ±1.10/±1.40/±0.31 rad | đầu bị *ràng buộc bám lệnh* |
| contact/air-time/`base_lin_vel` | **chỉ trong critic** (privileged) | **actor mù tiếp xúc chân** |
| `joint_vel` | noise ±0.25, **lag 1 tick cứng** | tín hiệu vận tốc kém — nền tảng của mọi phản ứng nhanh |

Đọc bảng này theo hướng ngược lại thì rõ: **microduck hiện tại không bị giới hạn agility bởi RL — nó bị giới hạn bởi một reward stack được chọn để đổi agility lấy sim2real an toàn.** Đó là lựa chọn đúng cho v1 (và là lý do nó đi được ngoài đời), nhưng nó chính là "giới hạn tiếp theo".

Repo cũng đã tự học được nguyên tắc để mở giới hạn đó — trong docstring của `microduck_roulade_env_cfg.py` `[MD_RL]`:

> *"the motion-blockers (`body_ang_vel`, `|a_z|`, arrival damping) kept near zero during discovery and introduced late by curriculum — the roll IS a large angular-velocity, large-impact event; taxing attempts prevents discovery (proven twice on standup)."*

và trong AGENTS.md: *"a 25 cm robot tumbles at 3.5–5.5 rad/s NATURALLY — don't impose human-scale speed intuitions via caps; put anti-violence pressure on impacts and thrash, not on rotation speed."*

Nguyên tắc đã có. Nó chỉ **chưa được áp dụng cho việc đi bộ** — velocity env vẫn giữ nguyên triết lý "quasi-static, thân thẳng, bước đều".

---

## 2. Định nghĩa lại "linh hoạt" thành đại lượng đo được

Trước khi tối ưu, phải có thước đo. "Giống BLDC hơn" không phải metric. Đề xuất bộ 6 chỉ số, đo trên checkpoint thật bằng eval headless (theo đúng luật *"Measure before theorizing"* của AGENTS.md):

1. **Trần tốc độ thực dụng**: `vx` lớn nhất còn giữ tracking error < 0.05 m/s và không ngã trong 30 s.
2. **Băng thông gia tốc**: thời gian từ lệnh 0 → 0.4 m/s (và 0.4 → −0.4, tức đảo chiều).
3. **Băng thông xoay**: `ωz` đỉnh đạt được khi xoay tại chỗ, và thời gian ổn định hướng.
4. **Biên độ phục hồi**: xung đẩy lớn nhất (m/s) chịu được ở mỗi tốc độ, theo 8 hướng.
5. **Chỉ số dynamic**: tỉ lệ thời gian *single-support*, chiều cao nhấc chân đỉnh, và độ dài pha bay (nếu có). Gait quasi-static ≈ double-support cao; gait dynamic đảo lại.
6. **Chi phí trung thực**: |a_z| đỉnh của thân + moment đỉnh servo + PWM saturation %. Chỉ số này ngăn "agility giả" — tức thứ nhanh trong sim nhưng bão hoà actuator trên robot thật.

`[SUY LUẬN]` Không có bộ số này, mọi thay đổi reward chỉ là cảm tính, và đúng như AGENTS.md cảnh báo, total reward có thể tăng thuần nhờ regularizer trong khi hành vi không hề khá hơn.

---

## 3. Thiết kế đề xuất — ba tầng theo chi phí

### Tầng 0 — Không đổi phần cứng (đòn tốt nhất trên mỗi đồng chi)

**0.1 Giải quyết mâu thuẫn low-pass (§1.3).** Hoặc tắt filter ở runtime, hoặc train *có* filter khớp alpha runtime. Không được để lệch. Đây là điều kiện tiên quyết — mọi số đo agility trước khi giải quyết việc này đều không tin được.

**0.2 Biến cái đầu 280 g từ ràng buộc thành cơ cấu — "cái đầu là đôi tay".**
Hiện `head_pose_tracking` (w=2.0, std 0.5) buộc 4 khớp cổ/đầu bám lệnh tức thời; cfg đã tự ghi lại rằng siết std xuống 0.1 làm policy **ngừng đi hẳn**, vì đầu 38% khối lượng *buộc* phải dao động khi đi. Đề xuất: chuyển gaze thành ràng buộc **hướng nhìn theo EMA** (charge DC bias, cho phép dao động triệt tiêu — đúng kỹ thuật mà repo đã dùng cho `head_pose_bias`), và mở một dải tự do quanh lệnh để policy dùng đầu như khối phản lực có chủ đích. `[SUY LUẬN]` 280 g ở cánh tay đòn ~0.1 m là một actuator momentum thật sự — đây là cách thay thế gần nhất cho "tay" mà không thêm một servo nào. Đây là hạng mục tôi cho rằng có khả năng tạo ra bước nhảy chất lượng chuyển động lớn nhất ở tầng 0.

**0.3 Cho actor "cảm nhận" được mặt đất — contact suy ra từ present_current.**
`[MD_RL]` contact và air-time hiện chỉ ở critic; actor mù. `[ODM]` Open Duck Mini v2 có foot contact trong observation. microduck không có sensor lực bàn chân — nhưng `[MD]` `duck-control/src/bus.rs:282-283` cho thấy **`present_current` của mọi servo đã được đọc mỗi tick** trong cùng block 12 byte với velocity và position. Tức tín hiệu tải đang có sẵn, miễn phí, đồng bộ.
Đề xuất: ước lượng contact/tải từ moment khớp chân (sim: `joint_torques`, thật: `currents_ma`) và đưa vào actor obs. `[SUY LUẬN]` Đây là con đường duy nhất tôi thấy để bỏ giả định "blind policy" mà không thêm phần cứng — và contact là thông tin thiếu quan trọng nhất cho gait dynamic (biết khi nào chân *thực sự* chạm đất là điều kiện để dám có pha bay).
⚠ Việc này **phá contract 61D** ⇒ phải là một policy family v2 với version bump ở runtime, chứ không sửa tại chỗ. Chi phí thật, phải quyết có chủ đích.

**0.4 Dùng 6 slot `body_pose` đang bỏ trống làm "núm agility".**
`[MD_RL]` `body_pose` giữ nguyên hạ tầng nhưng weight 0. Repo đã có tiền lệ dùng slot lệnh làm cờ chế độ (*"a posture flag lives in the twist vx slot"*). Đề xuất: một slot làm **agility/stiffness knob** — một policy, hai chế độ hành vi (calm: thân thẳng, bước ngắn, an toàn cho demo và pin; agile: cho phép lean, cho phép ω lớn, bước dài). `[SUY LUẬN]` Ưu điểm so với train hai policy riêng: giữ nguyên contract, không cần scheduler mới, và cho phép chuyển chế độ liên tục thay vì snap — đúng triết lý "mọi đường ranh là hằng số thời gian" mà runtime microduck đang theo.

**0.5 Curriculum lệnh theo *năng lực*, không theo iteration.**
Range đang cố định (`vx ±0.4`); cfg cũng ghi rằng curriculum mở range theo iteration từng "outpace capability". Đề xuất: mở range chỉ khi tracking error trung bình < ngưỡng và tỉ lệ ngã < ngưỡng — tự động dừng ở đúng biên vật lý của XL330, thay vì để con người đoán trần tốc độ. `[SUY LUẬN]` Đây là cách để *tìm ra* giới hạn hardware thật thay vì áp đặt nó bằng config.

**0.6 Đổi cấu trúc regularizer: tax bạo lực, đừng tax vận tốc.**
Theo đúng bài học roulade/standup, cho chế độ agile: giảm `body_ang_vel`/`angular_momentum` về gần 0 trong pha discovery; giữ `action_rate` thấp lúc đầu rồi ramp muộn hơn (hiện đạt −1.0 ở iter 1500); thay bằng áp lực lên **|a_z| impact, torque rate, PWM saturation và foot slip**. Nới `upright` khi có lệnh tốc độ lớn (lean là *cần thiết* để tăng tốc), giữ nghiêm khi lệnh ≈ 0.

**0.7 Mở nhịp bước và độ nhấc chân.**
`air_time` 0.125–0.3 s và clearance 2 cm là ràng buộc gait, không phải quy luật vật lý. Cho phép window rộng hơn theo chế độ agile và nâng clearance mục tiêu — kèm điều kiện tiên quyết là terrain/robustness tương ứng. `[SUY LUẬN]` Bước ngắn + chân lết là nguyên nhân trực tiếp của cảm giác "robot servo" ở bất kỳ humanoid nhỏ nào.

### Tầng 1 — Đổi cơ khí nhỏ, giữ nguyên 14 servo

| Thay đổi | Lý do (đo được) | Trade-off |
|---|---|---|
| Nới gót bàn chân (hiện chỉ ~20 mm sau trục ankle, tổng ~41×54 mm) | mở rộng polygon đỡ theo trục dọc ⇒ chống pitch tốt hơn khi hãm | thêm khối lượng ở đầu chi (tệ cho quán tính swing); cần cân bằng bằng số |
| Giảm backlash cơ khí (preload/hard stop) | ~4–8 mm sai bàn chân (§1.2) | chi phí gia công |
| Giảm khối lượng đầu, hoặc *hạ CoM đầu* | 280 g/38% là đòn bẩy quán tính lớn nhất trên robot | mất chức năng/biểu cảm — cân nhắc: có thể *giữ* khối lượng nếu áp dụng 0.2 và dùng nó có chủ đích |
| Tăng độ cứng link chân (leg 21,6 g, ankle 30 g) | biến dạng đàn hồi ăn vào thẩm quyền ankle | khối lượng |
| Đặt `-Backlash-` làm môi trường train mặc định | robot thật luôn có backlash | ~chi phí train tương đương |

### Tầng 2 — Thêm DOF (bước nhảy lớn nhất, và là chỗ ROBOTIS MINI đúng)

`[SUY LUẬN]` Xếp hạng ROI trên mỗi servo thêm vào:

1. **+2 ankle roll ⇒ 16 servo.** Đây là DOF có ROI cao nhất cho agility. Nó mở ra: điều khiển CoP hai trục, đứng một chân có kiểm soát, bước ngang thật, nghiêng người khi xoay nhanh. Hiện toàn bộ cân bằng lateral dựa vào `hip_roll` ±22° trên bàn chân phẳng 41 mm — đây là nghẽn cứng, và không có reward nào vượt qua được nghẽn cơ khí.
2. **+2…4 servo tay ⇒ 18 servo.** Tay mua ba thứ: điều tiết động lượng góc (đi nhanh/xoay nhanh), chống đỡ khi ngã (giảm hư hại — thứ đang phải xử lý bằng limp-fall), và đòn bẩy cho standup/roulade. **Đúng bằng nền tảng 18 servo bạn đang có**: 2×5 chân + 4 cổ/đầu + 2×2 tay = 18, hoặc 2×6 chân (có ankle roll) + 2×2 tay + 2 đầu = 18.
3. Thêm DOF thân (waist yaw) — ROI thấp hơn ở kích cỡ 25 cm `[SUY LUẬN]`.

**Ràng buộc phải đo trước khi cam kết**: `[MD]` mỗi tick 50 Hz đọc một block 12 byte/servo trên bus Dynamixel dùng `sync_read`, và IMU chia sẻ *cùng* bus qua `imu_to_dxl`. Thêm 2–4 thiết bị ⇒ tăng thời gian bus mỗi tick. **Phải đo ngân sách bus thực tế** trước khi chốt số servo; tôi không suy đoán con số này. Cộng thêm: khối lượng, giá, và điện (sụt áp đã là yếu tố có trong mô hình BAM).

---

## 4. Cái gì RL mua được — và cái gì không

Trung thực về khoảng cách servo↔BLDC, vì chiến lược sản phẩm phụ thuộc vào nó:

**RL mua được** (và microduck đã chứng minh): khai thác gần biên vật lý của actuator; phản ứng đúng lúc dựa trên trạng thái thay vì quỹ đạo dựng trước; phối hợp toàn thân (đầu + chân + thân là một chuyển động); dung sai với backlash, ma sát, sụt áp, trễ, lệch IMU nhờ DR; và tính liên tục thời gian — thứ tạo ra cảm giác "sống".

**RL không mua được**: công suất đỉnh và mật độ moment; băng thông vòng điều khiển (50 Hz control + 15–30 ms trễ + EMA ở chân, so với hàng kHz và moment trực tiếp của BLDC); độ backdrivable và compliance thật; độ cứng truyền động (backlash 2°). Nghĩa là: **nhảy cao, chạy có pha bay dài, tiếp đất hấp thụ mạnh — không nên là mục tiêu**. Mục tiêu đúng là *dải agility trung gian*: tăng tốc/hãm nhanh, xoay nhanh, bước dài và nhấc chân cao, single-support dài, lean có kiểm soát, phục hồi mạnh — cộng với ankle roll và tay ở tầng 2. `[SUY LUẬN]` Toàn bộ dải này khả thi với XL330 và **chưa ai làm trên nền servo giá rẻ** — đó chính là đột phá có thể lặp lại được, không phải việc đua với BLDC ở chỗ nó thắng chắc.

---

## 5. Lộ trình thực nghiệm

Tuân thủ workflow của AGENTS.md: xây trên velocity recipe, giữ DR/obs-noise/NaN-guard, viết cfg test, và **luôn** smoke test `uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5` trước mọi run dài.

**Giai đoạn A — thước đo và kiểm tra miễn phí (không train mới)**
1. Xây battery eval theo §2 trên policy walking hiện tại → có baseline bằng số.
2. Đo A/B `legs_lowpass` 0.7 vs 1.0 (§0.1) trên `scripts/infer_policy.py` và trên robot.
3. Chốt kết luận về mâu thuẫn filter trước khi đi tiếp.

**Giai đoạn B — task mới `Mjlab-Agile-Flat-MicroDuck`** (fork từ `make_microduck_velocity_env_cfg`)
4. Thêm agility knob vào một slot `body_pose` (§0.4); gaze theo EMA (§0.2); regularizer đổi cấu trúc (§0.6); nhịp bước mở (§0.7); curriculum lệnh theo năng lực (§0.5). Giữ 61D.
5. Cfg test: joint index đúng trên model thật, dấu mọi penalty (mọi `Episode_Reward/<penalty>` ≤ 0), gate đóng/mở đúng chỗ.
6. Smoke test → train → eval bằng battery §2, so với baseline A. Train luôn cả biến thể `-Backlash-`.

**Giai đoạn C — bỏ giả định blind (phá contract có chủ đích)**
7. Obs v2 = 61D + contact/tải suy từ moment khớp (§0.3), bump version model API ở runtime, rehearse bằng `scripts/infer_policy.py`.

**Giai đoạn D — phần cứng**
8. Tầng 1 (gót chân, backlash, CoM đầu) — mỗi thay đổi một lần, đo lại battery.
9. Đo ngân sách bus, rồi prototype +2 ankle roll (16 servo); sau đó +tay (18 servo). Mỗi lần thêm DOF là một model MJCF mới + retrain, không phải một reward tweak.

**Footgun đã biết, đừng đạp lại** (từ AGENTS.md và các cfg): dấu penalty tự-phủ-định; jackpot reward khi "đạt X"; so weight thay vì so *reward mass* khi copy regularizer giữa các env; siết std tracking cho một dao động *không thể tránh* (đầu 38%); DR tích luỹ qua reset; đưa smoothness tax vào *trước* khi kỹ năng hình thành; và tin metric sim mà không xem video + kiểm geom/trục nào chạm đất.

---

## 6. Kết luận một câu

Giới hạn tiếp theo của microduck **không phải RL** — mà là (a) một mâu thuẫn low-pass train/deploy đang lặng lẽ cắt dải tần nhanh, (b) một reward stack chọn quasi-static để đổi lấy sim2real, (c) một actor mù tiếp xúc trong khi tín hiệu current đã có sẵn trên bus, và (d) thiếu đúng hai DOF (ankle roll) cùng bộ điều tiết momentum (tay/đầu) mà ROBOTIS MINI và robot ROBO-ONE đều có `[NGOÀI REPO]`. Tấn công theo thứ tự đó — và đo bằng §2 sau mỗi bước.
