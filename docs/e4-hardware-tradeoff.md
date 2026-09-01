# E4 — Phương án phần cứng: mỗi thay đổi đổi được bao nhiêu

> **Đính chính (đọc trước):** dòng "+ ankle roll" dưới đây là **hình học tĩnh**
> với 20 g đại diện và giả định đế giữ phẳng — không phải mô phỏng. Mô phỏng
> thật với robot 16 servo (khớp + actuator + tiếp xúc) cho kết quả khác hẳn:
> khả năng dịch ngang tăng ~16×, nhưng khả năng chịu đẩy khi đứng chỉ +5%. Xem
> `docs/e4b-ankle-roll-simulation.md` — nó thay thế con số +81% ở §"Sáu kết luận".

Không cần GPU, không cần train. Mọi con số dưới đây tính từ chính MJCF đang
dùng (`robot/microduck/scene_walk.xml`, keyframe `STAND`) cộng thông số servo
BAM `xl330/m6`, bằng script tái lập được:

```bash
uv run python scripts/hw_tradeoff.py
```

Cách đọc: mỗi dòng là một phương án phần cứng, được áp vào model *trong bộ nhớ*
rồi tính lại cùng tám đại lượng. Cột quan trọng nhất là **`dv_lat`** — cú đẩy
ngang mạnh nhất mà robot còn cứu được — vì E0 đã chỉ ra: mọi cú ngã đều xảy ra
trên trục ngang.

| Phương án | z_com | a_tiến | a_phanh | a_ngang | với ngang | nâng chân | f_bước | **dv_ngang** |
|---|---|---|---|---|---|---|---|---|
| baseline (hiện tại) | 0.1411 | 2.37 | 1.42 | 1.54 | 19.4 | 20.8 | 11.97 | **0.226** |
| + ankle roll | 0.1418 | 2.36 | 1.42 | 1.53 | 40.1 | 20.8 | 10.11 | **0.409** |
| bàn chân rộng +10 mm | 0.1416 | 2.37 | 1.42 | 2.23 | 19.4 | 20.8 | 11.97 | **0.254** |
| bàn chân dài +5 mm | 0.1411 | 2.72 | 1.77 | 1.54 | 19.4 | 20.8 | 11.97 | 0.226 |
| nới hip_roll → 0.60 rad | 0.1411 | 2.37 | 1.42 | 1.54 | 18.3 | 36.0 | 11.97 | **0.217** ↓ |
| đầu nhẹ đi 30% | 0.1151 | 2.91 | 1.75 | 1.89 | 19.4 | 20.8 | 11.97 | 0.258 |
| ngồi thấp 0.6 rad (không đổi phần cứng) | 0.0938 | 3.39 | 1.87 | 2.51 | −0.0 | 14.3 | 20.37 | 0.062 ↓ |
| rộng +10 mm & nới hip_roll | 0.1416 | 2.37 | 1.42 | 2.23 | 18.3 | 36.0 | 11.97 | 0.245 |
| **ankle roll + rộng +10 mm + hip_roll 0.60** | 0.1423 | 2.36 | 1.41 | 2.22 | 54.3 | 36.0 | 10.11 | **0.561** |

Đơn vị: m, m/s², mm, Hz, m/s. "với ngang" = bàn chân đưa được ra ngoài bao
nhiêu mm ở biên hip_roll; "nâng chân" = bàn chân bị nhấc lên bao nhiêu khi làm
việc đó (nhấc lên nghĩa là **không đặt xuống được**, nên phần với đó vô dụng);
`f_bước` = tần số bước tối đa mà moment servo cho phép (chặn trên, không phải
mức đạt được thực tế).

## Sáu kết luận

**1. Ankle roll là thay đổi duy nhất mang tính quyết định: +81% khả năng chịu
đẩy ngang** (0.226 → 0.409 m/s). Lý do không phải "thêm moment" mà là hình học:
hôm nay khi hông xoay hết ra ngoài, bàn chân *nhấc lên 21 mm* nên chỉ 19 trong
40 mm với ngang là dùng được. Có ankle roll, đế chân giữ phẳng và cả 40 mm trở
thành dùng được.

**2. Nới biên hip_roll một mình làm robot TỆ HƠN** (0.226 → 0.217). Đây là kết
quả phản trực giác quan trọng nhất của E4: xoay hông thêm mà không có ankle roll
thì chỉ nhấc chân cao hơn (20.8 → 36.0 mm), không với xa hơn. Nếu không tính,
đây đúng là kiểu thay đổi cơ khí tốn tiền mà kết quả âm. Nới hip_roll **chỉ có
nghĩa khi đi kèm ankle roll** — lúc đó nó cộng thêm 14 mm (40.1 → 54.3).

**3. Bàn chân là món rẻ nhất trên bàn** (một chi tiết in lại, ~0 g, không dây
điện): rộng thêm 10 mm mỗi bên cho **+45% thẩm quyền ngang** (1.54 → 2.23 m/s²);
dài thêm 5 mm cho **+25% lực phanh** (1.42 → 1.77 m/s²) — đúng chỗ yếu nhất
theo phân tích lý thuyết (phanh yếu hơn tăng tốc 1.7 lần vì đế sau ngắn). Làm
trước mọi thứ khác.

**4. Làm đầu nhẹ đi là một cái bẫy.** Bảng nói giảm 30% khối lượng đầu cải thiện
mọi gia tốc — nhưng chỉ vì nó hạ CoM. Đi kèm là hai mất mát không nằm trong
bảng: thời gian phản ứng ngắn lại (0.120 → 0.108 s, robot rơi *nhanh hơn*), và
mất phần lớn "cơ quan giữ thăng bằng" đã tính ở
`docs/physics-limits-and-roadmap.md` §3.5 (khớp cổ dịch được CoM 45 mm, bằng 82%
cả bàn chân, gần như miễn phí về moment). Đổi lại chỉ +14% chịu đẩy. **Không
nên làm.** Cái đầu nặng là tài sản, không phải nợ.

**5. Servo không phải giới hạn ở bất kỳ dòng nào.** `f_bước` thấp nhất trong
bảng là 10.1 Hz, trong khi cứu ngã ngang chỉ cần ~4 Hz. Thêm 40 g ở mắt cá (hai
servo ankle roll) chỉ hạ chặn trên từ 12.0 → 10.1 Hz — vẫn thừa 2.5 lần. Nghĩa
là: **thêm ankle roll không phải đánh đổi**, đây là thay đổi thuần lợi về động
học.

**6. Gói đầy đủ cho 0.561 m/s** — đúng ngưỡng mà E0 đo được là robot ngã (mọi
cú đẩy ≥ 0.6 m/s). Nói cách khác: gói ankle roll + đế rộng + nới hip_roll
chuyển "chắc chắn ngã" thành "biên giới". Đây là con số hợp lý duy nhất cho câu
"phá giới hạn vật lý" theo nghĩa thật của từ đó.

Ghi chú về "ngồi thấp": hạ CoM cho gia tốc cao hơn (a_ngang 1.54 → 2.51) nhưng
đồng thời gập chân lại, làm cú với ngang gần như bằng 0 — nên trong tính toán
động học nó **không phải** món miễn phí. Con số 0.062 chỉ là chặn dưới của phép
tính hình học tĩnh; một dáng đi thật có thể lấy lại phần nào. Cần rollout để
kết luận, chưa nên tin.

## Thứ tự khuyến nghị

| Ưu tiên | Việc | Chi phí thật | Được gì |
|---|---|---|---|
| **P0** | In lại đế chân: rộng +10 mm mỗi bên, dài +5 mm | một chi tiết in, ~0 g | +45% ngang, +25% phanh, +12% chịu đẩy |
| **P1** | Thêm 2 servo ankle roll | xem cảnh báo dưới | +81% chịu đẩy ngang |
| **P1b** | Cùng lúc với P1: nới hip_roll lên 0.60 rad | thay bracket hông | +37% nữa (chỉ có tác dụng sau P1) |
| — | Làm đầu nhẹ | — | **không nên** (kết luận 4) |
| — | Nới hip_roll một mình | — | **không nên** (kết luận 2) |

**Cảnh báo về chi phí thật của P1, và nó lớn hơn tiền servo:** thêm hai khớp là
16 servo thay vì 14. Điều đó phá **contract 61D** — thứ mà `AGENTS.md` liệt vào
bất biến, vì nó cho phép runtime hot-swap 9 policy dùng chung một layout
observation. Cụ thể: block vị trí khớp 14 → 16, block vận tốc 14 → 16, block
action trước 14 → 16 ⇒ observation 61 → 67, action 14 → 16. Hệ quả: **mọi
policy hiện có phải train lại**, `duck-control` phải lên obs v2, và không thể
chạy song song policy cũ/mới trên cùng robot.

Vì vậy thứ tự đúng của cả dự án là: **làm xong phần mềm (E1 → E2 → E3) trên
14 servo trước, rồi đổi phần cứng một lần duy nhất** và train lại trên nền
kiến trúc đã tốt. Làm ngược lại là trả giá train lại hai lần.

## Giới hạn của tài liệu này

- `dv_ngang` là chặn trên bậc một (đặt chân ở capture point + phần đế hấp thụ
  trong nửa bước). Nó dùng để **xếp hạng phương án**, không phải để hứa một
  con số cho robot thật.
- Khối lượng servo ankle roll lấy tròn 20 g/khớp (XL330 + bracket in). Nếu
  thiết kế thật nặng hơn nhiều thì phải chạy lại script với số đúng.
- Chưa mô phỏng va chạm chân-với-chân khi nới hip_roll và mở rộng đế: hai đế
  cách nhau 81 mm tâm–tâm, mép trong hiện ở 18.5 mm. Rộng thêm 10 mm mỗi bên là
  còn khoảng, nhưng khi bước ngang thì phải kiểm tra bằng va chạm thật.
- Toàn bộ tính trên model `walk` (14 servo, không backlash). Backlash làm giảm
  thẩm quyền thật, theo hướng bất lợi cho mọi dòng như nhau nên không đổi thứ hạng.
