# Trần vật lý của microduck & lộ trình phá trần (bản tổng kết)

Tài liệu này trả lời ba câu hỏi bằng ngôn ngữ thường, không cần đọc số liệu thô:

1. Kế hoạch đang làm gì, và **kết quả cuối cùng trông như thế nào**.
2. Các con số đã đo (E0) **kết luận điều gì** — mỗi con số một câu.
3. **Trần vật lý** của thiết kế hiện tại: tính từ thông số servo + hình học
   robot, không cần train, để biết còn bao nhiêu dư địa và dư địa đó nằm ở đâu.

Mọi hằng số đều lấy từ chính repo (BAM params, MJCF), có ghi nguồn. Không có
số nào ở đây là ước lượng theo cảm giác.

---

## 1. Kế hoạch: đang ở đâu, kết quả cuối cùng là gì

**Mục tiêu**: giữ đúng tinh thần đã tạo ra microduck — *bỏ prior của con người
và đóng khoảng cách mô hình* — để đẩy độ linh hoạt lên một tầng nữa.

**Kết quả cuối cùng** (cái bạn sẽ thấy, không phải cái tôi sẽ viết):

| Chỉ số | Hôm nay (đo được) | Mục tiêu | Trần vật lý (tính được) |
|---|---|---|---|
| Tốc độ tiến | 0.229 m/s | 0.40–0.45 m/s | ~0.83 m/s (ranh giới đi/chạy) |
| Tốc độ ngang | 0.022 m/s | 0.10 m/s | ~0.15 m/s (bước ngang 40 mm ở 4 Hz, mục 3.4) |
| Quay tại chỗ | 0.41 rad/s | 1.0 rad/s | không bị chặn bởi servo |
| Chịu đẩy ngang | ngã ở ≥ 0.6 m/s | sống ở 0.6 m/s | ~0.33 m/s nếu giữ nhịp bước hiện tại |
| Tần số bước | 1.70 Hz | 3–4 Hz | không bị chặn bởi servo |
| Pha bay (flight) | 3% | 10–15% | có thật, không phải ảo tưởng |

Nghĩa là: **robot hiện đang chạy ở khoảng 27% trần vật lý của chính nó về tốc
độ**. Trần đó không do servo yếu, mà do ba thứ khác — xem mục 3.

**Lộ trình**, trạng thái thật:

| Giai đoạn | Nội dung | Cần GPU? | Trạng thái |
|---|---|---|---|
| E0 | Đo baseline trước khi train | Không | **Xong** (`docs/e0-baseline-report.md`) |
| **E-theory** | Trần vật lý, mục 3 tài liệu này | Không | **Xong** |
| E1 | Task `Agile`: EMA khớp runtime, bỏ prior gait, lệnh vừa năng lực | Có (train) | Code xong (`docs/e1-agile-task.md`), **chờ GPU** |
| E1.5 | Đầu làm cơ quan giữ thăng bằng (mục 3.5) | Không (thiết kế + đo) | Đề xuất trong tài liệu này |
| E2 | Bộ nhớ (GRU) — policy biết cơ thể mình đang ở đâu | Có | Chưa bắt đầu |
| E3 | Delay-aware: policy biết lệnh của mình đến muộn | Có | Chưa bắt đầu |
| E4 | Phần cứng: ankle roll / giảm khối lượng đầu | — | Chỉ phân tích |

**Bạn không cần mua GPU.** Repo đã có sẵn đường chạy trên GPU thuê theo giờ của
Hugging Face — chỉ cần thêm `--hf-jobs` vào đúng lệnh train (xem
`scripts/hf/README.md`, cần `hf auth login` + tài khoản HF có bật Jobs):

```bash
uv run train Mjlab-Agile-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max_iterations 4000 --hf-jobs
```

Đây là thứ tôi cần bạn quyết: mở HF Jobs (tôi chạy được ngay, tính phí theo
giờ GPU) hoặc dừng ở phần không cần GPU. Chi phí thực tế mỗi giờ tôi chưa tra
nên không ghi số ở đây.

---

## 2. E0 nói gì — mỗi con số một kết luận

E0 chạy policy thật (`alpha_walking.onnx`) trong MuJoCo trên CPU, tái hiện đúng
đường điều khiển của runtime. Đọc theo cột "kết luận", bỏ qua cột số nếu muốn.

| Đo được | Kết luận một câu |
|---|---|
| Lệnh 0.8 m/s → 0.229 m/s; lệnh 0.4 m/s → cũng ~0.23 m/s | Policy **đã bão hoà**: nửa dải lệnh là vùng chết, mọi gradient học tốc độ ở đó vô nghĩa. |
| Đi ngang: lệnh 0.4 → 0.022 m/s | Trục ngang **coi như không có**; đây không phải tinh chỉnh, mà là một kỹ năng chưa từng hình thành. |
| Mọi cú đẩy ≥ 0.6 m/s làm ngã, và đều theo trục ngang | Điểm yếu chịu nhiễu **trùng đúng** trục yếu nhất — cùng một nguyên nhân, không phải hai lỗi. |
| Tắt filter: bão hoà moment 0% → 9.3%, trượt chân ×2, tốc độ chỉ +17% | Nới filter **không mở được trần**; nó chỉ đổi trượt chân lấy tốc độ. Filter không phải nút thắt. |
| 94.3% single support, 3% pha bay, 1.70 Hz cố định | Dáng đi bị **đóng khuôn**: cửa sổ air-time do người đặt đã trở thành nhịp duy nhất policy biết. |
| Đứng yên vẫn trôi 0.111 m/s | Policy **không biết mình đang đứng yên** — dấu hiệu thiếu trạng thái nội tại (lý do E2 tồn tại). |
| `robotd/src/control.rs` ghi rõ alpha policies được train *có* filter | Mâu thuẫn low-pass trong tài liệu đã được giải quyết: **train phải có filter**, không phải tắt filter ở robot. |

Một câu tổng: **giới hạn hiện tại không nằm ở servo, mà ở việc policy chỉ biết
một dáng đi duy nhất, trong nửa dải lệnh, không có trục ngang, và không có ký ức.**

---

## 3. Trần vật lý: tính từ thông số, không cần train

### 3.0 Các hằng số (nguồn trong repo)

Servo XL330, mô hình BAM `m6` (`bam/params/xl330/m6.json`, cấu hình ở
`robot/microduck_constants.py`): `kt = 0.3660 N·m/A`, `R = 2.811 Ω`,
`kp_fw = 200`, `error_gain = 2.877e-3`, giới hạn dòng firmware `1.75 A`,
điện áp `6.5–8.2 V` (sàn 6.0 V), trễ actuator `3–6` bước sim (15–30 ms).

Hình học (MJCF `robot_walk.xml`, keyframe `STAND`): tổng khối lượng
**0.737 kg**, cụm đầu **0.280 kg = 38.0%**, CoM cao **0.1417 m**, đùi 42.2 mm,
cẳng 49.4 mm, bàn chân dài **54.7 mm** (34.2 mm trước trục ankle, 20.5 mm sau),
rộng 44.3 mm, hai chân cách nhau 81 mm.

### 3.1 Servo mạnh hay yếu? — Mạnh hơn bạn tưởng, nhưng theo cách khó dùng

Ba con số suy ra từ mô hình BAM ở 7.4 V:

- **Moment tối đa 0.64 N·m** (bị giới hạn bởi dòng 1.75 A, không phải bởi PWM).
- **Tốc độ không tải 20.2 rad/s** (1158 °/s).
- **Độ cứng vị trí 0.554 N·m/rad** = 9.7 mN·m cho mỗi độ sai số.

Con số thứ ba là chìa khoá và ít ai để ý: servo là **cái lò xo**. Nó chỉ sinh
moment khi có sai số vị trí. Muốn dùng hết thẩm quyền của bàn chân (0.247 N·m,
mục 3.3) thì phải có **25.6° sai số** giữa vị trí lệnh và vị trí thật.

Hệ quả trực tiếp: *vị trí lệnh không phải vị trí robot*. Một policy không có ký
ức phải suy ra tải trọng hiện tại chỉ từ một khung quan sát — mà cùng một góc
khớp có thể ứng với moment rất khác nhau. Đây là lý do định lượng để làm E2
(bộ nhớ), không phải lý do thẩm mỹ.

Điều servo **không** phải là nút thắt: nhấc chân nhanh. Một chu kỳ 1.7 Hz cần
hông quay ~3.4 rad/s — chỉ 17% tốc độ không tải. Muốn 4 Hz vẫn còn thừa.

### 3.2 Đây là robot rơi nhanh gấp 2.5 lần người

Con lắc ngược: `ω₀ = √(g/z_com) = √(9.81/0.1417) = 8.32 rad/s`, tức hằng số
thời gian rơi **τ = 0.120 s** (người: 0.30 s).

Ngân sách trễ của vòng điều khiển: 20 ms (50 Hz) + 15–30 ms (trễ actuator) =
**35–50 ms = 0.29–0.42 τ**. Quy đổi sang tỷ lệ người, tương đương một người
điều khiển thăng bằng với **88–125 ms** trễ. Con người ở mức đó đã đi lảo đảo.

Kết luận: microduck đang thăng bằng ở chế độ trễ nặng. Đó chính là lý do
policy phải *đoán trước* chứ không thể *phản ứng* — và lý do E3 (delay-aware)
là một trụ riêng, không phải chi tiết kỹ thuật.

### 3.3 Trần tốc độ: còn dư 3.6 lần, và không phải lỗi của kích thước

Số Froude `Fr = v²/(g·z_com)` là thước đo không chiều để so sánh robot 25 cm
với người. Hiện tại `Fr = 0.229²/(9.81×0.1417) = 0.038`. Ranh giới đi → chạy
trong sinh học ở `Fr ≈ 0.5`, tức **0.834 m/s** với chiều cao CoM này.

Nói cách khác: microduck hiện đi *tương đương một người đi bộ 0.58 m/s* — dạo
chậm. Vật lý cho phép 3.6 lần nữa trước khi buộc phải có pha bay. **Trần tốc độ
hiện tại là do reward và curriculum, không phải do robot nhỏ.**

Cái *bị* chặn bởi hình học là thẩm quyền gia tốc, do bàn chân quyết định:

| Hướng | Tay đòn | Gia tốc tối đa |
|---|---|---|
| Tiến | 34.2 mm | **2.37 m/s²** |
| Lùi (phanh) | 20.5 mm | **1.42 m/s²** |
| Ngang (một chân) | 22.2 mm | **1.53 m/s²** |

Phanh yếu hơn tăng tốc 1.7 lần — bàn chân ngắn phía sau. Đây là lý do vật lý
để không kỳ vọng "dừng gấp" trước khi có ankle roll hoặc bàn chân dài hơn.

### 3.4 Trục ngang: lý thuyết và số đo trùng nhau

Muốn cứu một cú đẩy ngang `Δv`, chân phải đặt ra ngoài một khoảng
`Δv/ω₀`. Đo bằng động học thuận trên chính MJCF: biên hip_roll `±0.384 rad`
cho bàn chân dịch ngang **+40 mm ra ngoài / −28 mm vào trong** so với tư thế
đứng — và khi xoay hết ra ngoài, bàn chân *nâng lên 21 mm*, tức mất tiếp xúc.

Vậy `Δv` tối đa cứu được bằng một bước ≈ `0.040 × 8.32 =` **0.33 m/s**, cộng
thêm phần bàn chân hấp thụ được (1.53 m/s² trong khoảng thời gian một bước).

E0 đo: ngã ở mọi cú đẩy ≥ 0.6 m/s, sống ở mức thấp hơn. **Lý thuyết và số đo
khớp nhau** — nên đây không phải lỗi huấn luyện, mà là trần động học.

Hai đường phá trần, và chỉ hai:

1. **Bước nhanh hơn** (không cần phần cứng): thời gian rơi là 0.120 s, nên muốn
   cứu bằng bước chân thì nhịp bước phải tiến tới ~4 Hz, gấp **2.4 lần** mức
   1.70 Hz hiện tại. Đây chính là điều E1 nhắm tới khi bỏ cửa sổ air-time cố
   định — và giờ nó có một con số mục tiêu vật lý, không phải "cho nó linh hoạt hơn".
2. **Phần cứng** (E4): ankle roll, bàn chân rộng hơn, hoặc hạ CoM. Mỗi 10 mm
   nới thêm về ngang đổi được +0.083 m/s khả năng chịu đẩy.

### 3.5 Phát hiện lớn nhất: cái đầu là một cơ quan giữ thăng bằng đang bị cấm dùng

Cụm đầu chiếm 38% khối lượng. Tôi đo dịch chuyển CoM toàn thân khi *chỉ* đổi
góc cổ/đầu (động học thuận trên MJCF, so với tư thế đứng):

| Khớp | Dịch CoM theo trục tiến (x) |
|---|---|
| `neck_pitch` về sau hết biên | **+30.0 mm** |
| `neck_pitch` về trước | −14.8 mm |
| `head_pitch` | −14.8 mm … +6.6 mm |
| `head_yaw` (±1.4 rad) | ±3.1 mm (và chỉ ±3.7 mm theo trục ngang) |

Toàn bộ bàn chân chỉ cho 54.7 mm cửa sổ CoP. **Chỉ riêng khớp cổ cho 45 mm dịch
CoM** — bằng 82% cả bàn chân. Và cái giá về moment gần như bằng không: đưa đầu
đạt 6 rad/s trong 0.1 s chỉ cần 0.022 N·m, tức **2.3° sai số vị trí**, so với
trần 0.64 N·m.

Nói cách khác: microduck có sẵn một "cánh tay ROBO-ONE" — chỉ là nó đang được
dùng làm camera. Reward stack hiện tại còn *chủ động ngăn* việc dùng đầu để
thăng bằng (bám tư thế đầu theo lệnh gaze + `head_pose_bias_penalty`).

Điểm tinh tế, và là lý do việc này khả thi: `AGENTS.md` đã ghi bài học "đầu
38% khối lượng *bắt buộc* phải dao động khi đi, chỉ nên tính phí phần lệch DC
qua L1 trên EMA 1 s". Tức là **kênh dùng đầu để thăng bằng đã mở về mặt reward**:
gaze giữ giá trị trung bình, còn thành phần dao động được miễn phí. Chưa có gì
*thưởng* cho việc dùng nó.

Hai điều quan trọng phải nói thẳng, vì rất dễ oversell:

- Theo **trục ngang**, cái đầu gần như vô dụng (±3.7 mm). Cứu ngã ngang vẫn
  phải bằng bước chân. Cái đầu là cơ quan **sagittal**.
- Khi hai chân đang trên mặt đất, một cú xoay đầu **không** làm robot quay:
  0.022 N·m nhỏ hơn ngân sách ma sát xoay của bàn chân (~0.14 N·m). Cái đầu
  đóng góp ở pha bay, ở lúc chân sắp trượt, và ở việc **đặt CoM** — không phải
  làm bánh đà quay thân trên mặt đất.

Đề xuất **E1.5** (không cần GPU để thiết kế và đo, chỉ cần GPU để train):
thêm một reward trả cho *tương quan* giữa vận tốc đầu và sai số CoM–CoP —
tức trả tiền khi cái đầu di chuyển đúng hướng cần để giữ thăng bằng, giữ
nguyên phần gaze DC. Đây là "bỏ một prior của con người" tiếp theo: prior đang
bị bỏ là *"đầu để nhìn, chân để đi"*.

---

## 4. Việc làm được ngay, không cần GPU

Theo thứ tự giá trị / chi phí:

1. **E0.5 — battery đo trục ngang và đầu** (mở rộng `scripts/eval_battery.py`,
   CPU): quét cú đẩy ngang từng bước 0.05 m/s để tìm chính xác ngưỡng sống/ngã
   (lý thuyết dự đoán ~0.33 m/s); và bơm một chuỗi lệnh gaze hình sin vào chính
   policy alpha đang có để **đo thật** xem đầu tạo ra bao nhiêu tác động lên
   thân. Kết quả: hoặc xác nhận mục 3.5, hoặc bác bỏ nó trước khi tốn một giờ GPU.
2. **Kiểm tra giả định vật lý theo đúng checklist của `AGENTS.md`** (CPU): tư
   thế đích có phải điểm cân bằng ổn định, đo chiều cao thật dưới policy đứng.
3. **Đo compliance và backlash trên testbench BAM một khớp** (CPU): xác định
   vùng chết thực tế của ±1° backlash so với 25.6° sai số cần cho moment đầy —
   để biết backlash có đáng lo hay là nhiễu bậc hai.
4. **Viết trước giao diện GRU** (không train): trạng thái ẩn là I/O của ONNX,
   nên phần Rust runtime mang `h` qua các tick có thể viết và test trước khi có
   policy. Giữ nguyên contract 61D.
5. **Phân tích phương án phần cứng E4** trên giấy: mỗi mm bàn chân, mỗi gram đầu
   đổi được bao nhiêu m/s khả năng chịu đẩy (công thức ở mục 3.3–3.4).

Việc **bắt buộc phải có GPU**: mọi thứ có chữ "train". mjlab chạy trên MuJoCo
Warp nên cần CUDA — không có đường CPU. Nhưng như mục 1: `--hf-jobs` là GPU
thuê theo giờ, không cần mua máy.

---

## 5. Nếu chỉ đọc một đoạn

Servo không phải giới hạn. Robot đang đi ở 27% trần vật lý của nó, phanh yếu
gấp 1.7 lần so với tăng tốc vì bàn chân ngắn, và ngã theo trục ngang ở đúng
ngưỡng mà động học hông dự đoán (0.33 m/s). Ba nút thắt thật, theo thứ tự:
**(1)** policy chỉ biết một nhịp bước 1.70 Hz trong khi cứu ngã cần ~4 Hz;
**(2)** servo chỉ sinh lực qua sai số vị trí (25.6° cho moment đầy) nên policy
không có ký ức thì không thể biết cơ thể mình đang chịu tải gì; **(3)** cái đầu
nặng 38% khối lượng, có thẩm quyền dịch CoM bằng 82% cả bàn chân, moment gần
như miễn phí — và đang chỉ được dùng làm camera.
