# E5 — Thiết kế cơ khí thật cho mắt cá: kết quả

Tiếp theo kế hoạch đã duyệt ở `docs/e5-ankle-mechanical-design-plan.md`. Khác với
E4b (hai khối đỏ, khối lượng tự đặt 20 g, không va chạm), lần này mọi chi tiết đều
là **STL in được sinh bằng CAD tham số**, khối lượng lấy từ **thể tích in thật ×
1.24 g/cm³**, và va chạm được kiểm bằng **hình học đã biên dịch trong MuJoCo**.

Tất cả số dưới đây là **mô phỏng và hình học**, chưa có gì chạy trên robot thật.

Sinh lại được bằng:

```bash
uv run python scripts/e5_ankle_cad.py        # STL chi tiết in
uv run python scripts/e5_variant.py          # 6 MJCF (walk + allcollisions)
uv run python scripts/e5_interference.py --analysis --fine-scan
uv run python scripts/ankle_roll_sim.py ...  # T0-T3, không sửa protocol
```

## 1. Thiết kế: cái gì đi đâu

| Chi tiết in | Thể tích | Khối lượng PLA |
|---|---:|---:|
| `e5_yoke` (yoke chữ thập, mang trục roll) | 2.23 cm³ | **2.76 g** |
| `e5_footplate` (tấm mang đế + tay đòn 5 mm) | 0.96 cm³ | **1.19 g** |
| `e5_cradle` (giá servo trên cẳng chân) | 1.71 cm³ | **2.12 g** |
| `e5_pushrod` (thanh đẩy Ø3 + rotuyn) | 0.34 cm³ | **0.42 g** |
| `e5_sole` (đế mới 51 × 59, có rocker) | 2.94 cm³ | **3.65 g** |

Để so sánh: đế gốc `sole_left.stl` là 6.25 cm³ ⇒ 7.75 g, nên **đế mới rộng hơn
nhưng nhẹ hơn một nửa** vì nó là vỏ có gân, không phải khối đặc.

| Biến thể | Thêm mỗi chân | Tổng khối lượng robot |
|---|---:|---:|
| baseline (14 servo) | — | 737.2 g |
| `direct` (trục roll tại trục pitch) | +25.9 g | 789.1 g |
| `remote` (trục roll lệch vào 12 mm, servo trên cẳng chân, đòn 2:1) | +28.5 g | 794.2 g |
| `remote_sole` (`remote` + đế mới) | +32.1 g | 801.5 g |

Tức chi phí thật là **+7.7% khối lượng**, không phải con số 40 g/chân tôi ước ở E4.

Vị trí giá servo roll **không phải tôi chọn bằng mắt**: nó là kết quả tìm kiếm trên
100 phương án, lấy phương án đầu tiên **không sinh thêm bất kỳ va chạm nào** trong
toàn dải khớp gốc (225 tư thế) — kết quả: lùi 12 mm, dọc theo cẳng chân 12 mm, vào
trong 18 mm. Thanh đẩy dài **27–31 mm**, tính từ toạ độ thế giới ở tư thế `STAND`.

## 2. Hai lỗi mô hình đã bắt được (và đó là lý do phải làm bước này)

1. **Lần đầu các chi tiết là khối đặc**: đế mới nặng 41.6 g, tổng robot 0.93 kg
   (+26%). Với khối lượng đó `remote` ngã 4/5 lần và bão hoà moment 37.9%. Sau khi
   làm đúng kiểu vỏ có gân: 3.65 g.
2. **Thanh đẩy xuyên vào giá servo**. Ở `STAND`, model có 4 tiếp xúc âm
   (−2.35 đến −5.89 mm) giữa `pushrod` và `cradle`/`servo`. Đầu rotuyn *phải* tựa
   lên horn — đó là khớp, không phải va chạm — nên MuJoCo mỗi bước lại giải một
   xuyên thấu vĩnh viễn: robot nghiêng 46°, bão hoà moment 100%, đi **giật lùi**.
   Đã khai báo `contact/exclude` cho các cặp lắp ghép và **thêm assert `ncon == 0`
   ở `STAND` và tư thế zero** vào generator để lỗi loại này không quay lại.

Cả hai lỗi đều cho ra những bảng số "trông như kết quả thiết kế". Đây chính là thứ
mà mô hình khối đỏ ở E4b không thể phát hiện.

## 3. Giới hạn thật: biên roll dùng được, đo từng độ

Quét từng 1°, mỗi lần một mắt cá, các khớp khác giữ `STAND`:

| Model | Roll vào trong | Chi tiết chạm | Roll ra ngoài |
|---|---:|---|---:|
| `direct` | **12°** | đế chạm **chính cẳng chân mình** | > 30° |
| `remote` | **22°** | đế chạm cẳng chân | > 30° |
| `remote` (còn bị giới hạn sớm hơn) | **18°** | `footplate` chạm **servo roll của tôi** | — |

Biên này **không đổi theo ankle pitch** (đo ở −0.6 … +0.6 rad).

Nguyên nhân là hình học đã đo từ đầu: khe dọc giữa mặt trên đế và cẳng chân chỉ
**5.39 mm**, và nó đóng lại rất nhanh — 0° → +5.39 mm, 5° → +2.26 mm, 10° → −0.92 mm.

Hai hệ quả **đảo lại một khuyến nghị của tôi trong kế hoạch**:

- **Dịch trục roll vào trong 12 mm là đúng, nhưng vì một lý do khác lý do tôi nói.**
  Tôi đề xuất nó để giảm moment tĩnh; thực tế cái nó mua được là **biên roll dùng
  được gấp gần đôi (12° → 22°)**, vì nó đưa tâm quay của đế ra xa cạnh cẳng chân.
- **Giả thuyết "vát góc cẳng chân là mua biên roll miễn phí" của tôi SAI với
  `direct`.** Điểm chạm nằm ở *giữa* cạnh dưới tấm cẳng chân, không phải góc trong:
  vát 2 mm và 4 mm không mua được độ nào, vát 6 mm mua **1°** và chỉ còn 1.95 mm
  vật liệu (quá mỏng để in chịu lực). Với `remote` thì vát 6 mm có tác dụng
  (22° → 27°), nhưng khi đó **giới hạn lại là bracket của tôi (18°)** — nghĩa là
  vòng thiết kế tiếp theo phải thu gọn tay đòn `footplate`/dịch servo, chứ không
  phải cắt vào chân robot.

Cách còn lại để mua biên roll là hạ bàn chân xuống, đo được **1.8° mỗi mm**
(3 mm → +6°, 6 mm → +11°, 9 mm → +16°). Nhưng E4b đã đo hạ 18 mm làm **xấu** khả
năng chịu đẩy ngang (−7.7%), nên đây là đánh đổi phải trả bằng chiều dài chân, và
tôi **không** đưa nó vào thiết kế đề xuất.

Va chạm còn lại của các chi tiết mới đều ở góc dải quét cực đoan (ví dụ
`ankle_pitch −0.6`, `roll −45°`, `hip_roll −0.6`, `knee −0.9`, `hip_yaw −0.4` cùng
lúc), không phải tư thế đi bộ; trong toàn dải khớp gốc là **0 va chạm mới**.

## 4. Mô phỏng: 4 robot, cùng một bộ test (T0–T3 của E4b, không sửa protocol)

| Chỉ số | baseline | `direct` | `remote` | `remote_sole` |
|---|---:|---:|---:|---:|
| T0 nghiêng khi đứng yên | 1.13° | 1.05° | 1.13° | 1.16° |
| T0 hai chân tiếp đất | 5/5 | 5/5 | 5/5 | 5/5 |
| T1 chịu đẩy **ngang** | 0.598 m/s | 0.629 | 0.614 | **0.766 (+28%)** |
| T1 chịu đẩy **trước-sau** | 0.263 m/s | 0.279 | 0.294 | **0.355 (+35%)** |
| T2 dịch CoM khi đế còn phẳng | 2.6 mm | **40.7** | 10.2 | 18.4 |
| T3 tốc độ đi với policy 14-action hôm nay | 0.163 m/s | 0.122 | 0.111 | 0.060 |
| T3 số lần ngã | 0/5 | 0/5 | 0/5 | 0/5 |

Đọc bảng này:

- **Đế mới là món lãi lớn nhất và rẻ nhất.** Nó không cần servo nào, nhẹ hơn đế cũ,
  và một mình nó (trong `remote_sole`) mang lại +28% chịu đẩy ngang và +35% trước-sau
  — lớn hơn mọi thứ mà hai servo roll mang lại trong bài test đứng.
- **Hai servo roll vẫn *không* tự trả lời được câu hỏi lớn.** Giống E4b: khả năng
  hình học tăng (T2), nhưng ở bài test **đứng** thì chỉ +2…+5%. Ankle roll chỉ có
  giá trị khi policy học **bước ngang** với nó — và đó là bài test cần GPU.
- **Policy 14-action hôm nay luôn chạy chậm hơn trên phần cứng mới** (−25% đến
  −63%) nhưng **không ngã lần nào**. Đúng như dự đoán: thêm thẩm quyền cho một
  controller chưa biết dùng nó thì phải trả giá, cho tới khi train lại.

## 5. Thiết kế đề xuất chốt lại

1. **Yoke chữ thập, trục roll giao trục pitch nhưng dịch vào trong 12 mm** — biên
   roll dùng được 22° thay vì 12°, không tăng chiều cao, không tăng chiều dài chân.
2. **Servo roll trên cẳng chân** ở (lùi 12, dọc 12, vào trong 18) mm, truyền bằng
   **thanh đẩy 27–31 mm, tỷ số 2:1** (moment tại khớp ±1.92 N·m, độ cứng ×4).
3. **Đế mới 51 × 59 mm có rocker** — làm được ngay, độc lập với mọi thứ khác, và là
   phần thắng đậm nhất trong đo lường.
4. **Vòng thiết kế tiếp theo (chưa làm):** thu gọn tay đòn `footplate` để bỏ giới
   hạn 18° do chính bracket của tôi tạo ra; mục tiêu là biên roll đối xứng ±22° trở
   lên mà không hạ bàn chân.

## 6. Nói rõ giới hạn

- Tôi **không có quyền truy cập Onshape** của microduck, nên các STL này là chi
  tiết mới thiết kế khớp theo kích thước đo được, **không phải bản sửa trên CAD
  gốc**. Trước khi in, phải đối chiếu lỗ vít và dung sai bằng tay.
- Kiểm va chạm bằng MuJoCo bắt **giao thoa hình học**; nó không nói gì về độ bền
  chi tiết in, backlash rotuyn, cọ dây hay sai số lắp ráp.
- Thanh đẩy được mô hình bằng `gear=2` (đòn lý tưởng), không phải cơ cấu bốn khâu
  đầy đủ; ở góc roll lớn tỷ số truyền thật sẽ thay đổi chút ít.
- Với `gear=2`, `ctrlrange` mặc định của class ứng với dải mục tiêu ±5 rad, rộng
  hơn dải khớp ±0.785 rad rất nhiều. Không phải lỗi ở đây, nhưng **khi train thì
  phải đặt lại `ctrlrange`**, nếu không action scale sẽ lệch.
- **Câu hỏi quyết định vẫn chưa được trả lời**: policy 16-action *học* dùng ankle
  roll để bước ngang thì đi được bao nhanh và chịu đẩy bao nhiêu. Nó cần GPU.

## Ảnh

Các chi tiết mới được tô **màu cam** để phân biệt với chi tiết gốc.

- `docs/img/scene_walk_e5_remote_left_mechanism_{side,front}_{stand,roll20}.png` —
  cận cảnh chân trái, ở `STAND` và ở `left_ankle_roll = +20°`. Yoke và tay đòn thấy
  rõ; giá servo và thanh đẩy trên cẳng chân bị vỏ đùi che phần lớn ở các góc này.
- `docs/img/scene_walk_e5_{direct,remote,remote_sole}_{stand,ankle,lean,lean_ankle}.png`
  — toàn thân, so sánh với bộ ảnh baseline của E4b.
