# E5 — Thiết kế cơ khí thật cho mắt cá + gói tối ưu linh hoạt (KẾ HOẠCH, chờ duyệt)

Bạn nói đúng hai điều. Thứ nhất: hai khối đỏ trong E4b là **hộp bao**, không phải
chi tiết cơ khí — chúng mô phỏng đúng *động lực học* (khối lượng, quán tính,
khớp, tiếp xúc của bàn chân) nhưng **không chứng minh được là lắp được**. Thứ
hai: chỉ thêm ankle roll thì chưa phải "thiết kế tối ưu cho linh hoạt".

Tài liệu này là **kế hoạch chờ bạn duyệt**, chưa implement gì.

## 1. Số liệu cơ khí đo từ chính CAD của microduck

Đo từ STL trong `robot/microduck/assets/` (xuất từ Onshape qua onshape-to-robot)
và từ khung động học `robot_walk.xml` ở keyframe `STAND`:

| Chi tiết | Kích thước thật (mm) |
|---|---|
| servo `xl330` (cả horn) | **29 × 20 × 34** |
| `sole_left` (đế tiếp đất) | **41 × 54 × 12.9** |
| `ankle_left` (bracket mắt cá) | 39.5 × 36.5 × 25.5 |
| `leg` (cẳng chân — một tấm mỏng!) | **8 × 20 × 58** |
| `upper_leg_left` | 28 × 47.7 × 61 |

| Vị trí (STAND, mm) | |
|---|---|
| trục ankle pitch | z = **25.4**, y = **57.9** |
| tâm đế | z = 8.6, y = **40.7** → trục pitch **lệch ra ngoài 17.2 mm** so với tâm đế |
| mặt trên đế | z ≈ 15 → **chỉ còn ~10 mm** khoảng trống dọc dưới trục pitch |
| hai tâm đế cách nhau | 81.3 (khe trong giữa hai đế ≈ 40 mm) |
| trục hip_pitch | z = 102.5 |

Ba kết luận cơ khí rút ra ngay, và chúng loại bỏ luôn một số phương án:

1. **Xếp tầng (serial) là bất khả thi về cơ khí, không chỉ "kém hơn".** Dưới trục
   pitch chỉ có ~10 mm; một servo XL330 dày 20–29 mm không nhét vào đó mà không
   nâng cả robot lên — đúng như bản `serial` phải nâng 18 mm, và E4b đã đo được
   nó làm **xấu** trục hay ngã (−7.7% chịu đẩy ngang).
2. **Trục mắt cá đang lệch ra ngoài 17.2 mm so với tâm đế.** Hệ quả tính được:
   để giữ đế phẳng khi trọng tâm nằm ở *tâm đế*, khớp roll phải giữ
   **0.125 N·m**; ra tới cạnh đế là **0.20 N·m**. Servo có 0.64 N·m nên đủ lực,
   **nhưng** độ cứng vị trí chỉ 0.554 N·m/rad ⇒ 0.20 N·m tương ứng **~21° sai số
   góc**. Direct-drive roll sẽ *oằn xuống* dưới tải. Đây là con số quyết định
   thiết kế, và nó không xuất hiện trong bất kỳ phân tích trước đó của tôi.
3. **Khối lượng đặt ở đâu quan trọng hơn khối lượng bao nhiêu.** 20 g ở mắt cá
   cách trục hip_pitch 76.6 mm ⇒ quán tính swing thêm 1.18e-4 kg·m²; cũng 20 g
   đặt ở đầu trên cẳng chân (r = 42.2 mm) chỉ còn 3.6e-5 — **ít hơn 3.3 lần**.
   Đây gần như chắc chắn là nguyên nhân của −22% tốc độ đi mà E4b đo được ở bản
   `coincident`.

Tham chiếu: Open Duck Mini v2 cũng **không** có ankle roll (`robot.xml`: mỗi
chân đúng 5 khớp yaw/roll/pitch/knee/ankle) — cả lớp robot này đều dừng ở đây.

## 2. Bốn phương án cơ khí, và cái tôi chọn

| Phương án | Cơ khí | Phán quyết |
|---|---|---|
| A. Xếp tầng dưới pitch | servo roll nằm dưới mắt cá | **Loại.** Không có 10 mm để nhét, và đã đo là xấu hơn baseline. |
| B. Roll đồng trục, servo tại mắt cá | trục roll giao trục pitch (khớp chữ thập), servo bắt cạnh bàn chân | Dùng được, nhưng trả bằng 20 g distal ⇒ đúng cái làm chậm 22%. |
| C. **Roll đồng trục + truyền động từ xa** | khớp mắt cá là **khớp chữ thập (Hooke)**: roll giao pitch, không thêm chiều cao. Servo roll dời **lên đầu cẳng chân**, kéo bàn chân qua **thanh đẩy** (push-rod) + rotuyn hai đầu | **Chọn.** Giữ nguyên chiều dài chân và CoM, nhưng quán tính swing thêm ít hơn 3.3 lần; và tỷ số cánh tay đòn của thanh đẩy là **núm xoay để đổi độ cứng**: đòn 2:1 biến 21° oằn thành ~5°. |
| D. Mắt cá vi sai 2 servo song song | hai servo song song cùng tạo pitch+roll (kiểu ROBO-ONE / ankle vi sai) | Mạnh nhất về moment (cả hai servo góp cho cả hai trục) nhưng cần đổi cả cẳng chân + coupling trong policy. **Để dành làm bước 2**, sau khi C chạy được. |

Trục chọn của C không phải "cho đẹp": nó là **cách duy nhất trong bốn cách vừa
không nâng robot, vừa không đặt thêm khối lượng ở mắt cá, vừa cho phép tăng độ
cứng roll bằng tỷ số truyền** — ba đúng cái mà E4b đo được là chỗ đau.

## 3. Gói "linh hoạt nhất", không chỉ ankle roll

Sáu hạng mục, xếp theo tỷ lệ lợi ích/chi phí đã có số:

1. **Đế mới** (in lại, ~0 g, không phá contract): 41 × 54 → **51 × 59 mm**, gót
   dài thêm 5 mm, mũi và gót vê tròn thành *rocker* để có toe-off. E4 đã tính:
   +45% thẩm quyền ngang, +25% lực phanh.
2. **Dịch trục mắt cá vào tâm đế** (giảm lệch 17.2 mm → ~5 mm). Chỉ là in lại
   bracket mắt cá; nó xoá gần hết 0.125 N·m tải tĩnh trên khớp roll, tức xoá
   phần lớn cái oằn 21°. Miễn phí về khối lượng, và **có lợi cả khi không thêm
   servo nào**.
3. **Khớp chữ thập + roll truyền động từ xa** (phương án C): 2 servo, +2 action.
4. **Nới hip_roll 0.384 → 0.60 rad** — chỉ có nghĩa *sau khi* có roll (E4 đo:
   nới một mình làm xấu đi). Phải kiểm tra va chạm thật đùi–vỏ thân trước.
5. **Giữ nguyên cái đầu nặng.** Nó là cơ quan dịch CoM 45 mm với giá 2.3° sai số
   — làm nhẹ đầu là bẫy (E4 §4).
6. **Đường dây trong chuỗi**: XL330 là bus daisy-chain, dây roll phải đi qua tâm
   khớp chữ thập, nếu không nó sẽ là cái hãm mềm ở đúng biên roll lớn.

## 4. Tôi sẽ làm gì (E5), và bằng chứng nào để bạn tin

1. **CAD tham số bằng Python** (`trimesh` + `manifold3d`, CSG thật — đã thử chạy
   được trên máy này): sinh **STL in được thật** cho 4 chi tiết — yoke chữ thập,
   giá servo roll trên cẳng chân, thanh đẩy + rotuyn, đế mới. Kích thước lấy từ
   bảng §1, giao diện bắt vít theo envelope XL330 29×20×34.
2. **Khối lượng lấy từ vật lý, không ước lượng**: thể tích STL × mật độ PLA
   1.24 g/cm³ + XL330 **18 g** (số nhà sản xuất) + bulông/bearing. Thay cho con
   số "20 g" tôi tự đặt ở E4b.
3. **MJCF với collision geom thật của chi tiết mới** (không phải hộp
   `contype=0`), rồi **quét toàn dải khớp** (roll × pitch × hip_roll × knee) đếm
   tiếp xúc MuJoCo để tìm **giao thoa cơ khí thật**: chi tiết mới với đế, với
   chân kia, với vỏ thân. Đây chính là điều bản E4b không làm được.
4. **Mô phỏng thanh đẩy đúng cách**: `equality/joint` (hoặc tendon) để moment và
   tỷ số truyền phản ánh cánh tay đòn thật, không phải actuator lý tưởng.
5. **Chạy lại đúng bộ test E4b (T0–T3) trên 4 model**: baseline / B / C / C +
   đế mới + trục dịch vào, cùng render ảnh — để bạn thấy chênh lệch trên cùng
   một thước đo, kể cả nếu kết quả lại đảo chiều nữa.
6. **Báo cáo `docs/e5-*.md`** với bảng vật liệu (in gì, mua gì), khối lượng thật,
   kết quả quét giao thoa, và kết quả mô phỏng.

Ước lượng công sức của tôi: khoảng **1–2 session**, không cần GPU.

## 5. Giới hạn phải nói trước

- Tôi **không có quyền truy cập Onshape** (`cad.onshape.com/documents/8049...`),
  nên STL của tôi là **chi tiết mới thiết kế để khớp giao diện đo được**, không
  phải bản sửa trên CAD gốc. Trước khi in thật, một người phải đối chiếu lỗ vít
  và dung sai với CAD.
- Quét giao thoa bằng MuJoCo bắt được **va chạm hình học**, không bắt được: độ
  bền chi tiết in, backlash rotuyn, cọ dây, và độ lệch lắp ráp.
- Bài test quyết định — policy 16 action **học** dùng ankle roll để bước ngang —
  vẫn **cần GPU**. E5 trả lời "thiết kế này lắp được, nặng đúng bao nhiêu, và có
  làm hỏng dáng đi hiện tại không", **không** trả lời "nó đi giỏi hơn bao nhiêu".
- Vẫn chưa có gì được kiểm chứng trên robot thật.

Duyệt là tôi bắt đầu từ hạng mục 1–3 của §3 (đế mới + dịch trục + khớp chữ thập
truyền động từ xa) theo đúng §4.
