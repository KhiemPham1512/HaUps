# Chọn kích thước và so sánh ảnh trực tiếp

## Hành vi hiện hành — thay thế mô tả lịch sử bên dưới

Trên màn hình `> 800px`, giao diện dùng bố cục app-shell vừa khít một màn hình (`100vh`, không cuộn trang ở `1440×900` và `1280×720`): header mảnh chỉ có tên ứng dụng, cột trái `~360px` gồm vùng chọn/kéo thả ảnh + danh sách file đã chọn, tuỳ chọn (preset, giữ chi tiết nguồn, hệ số `2×`/`4×`/`8×`) và bảng hàng đợi (thanh tải hàng loạt + danh sách `#jobs` tự cuộn nội bộ `overflow: auto`); cột phải là khung xem ảnh `#preview-panel` chiếm toàn bộ phần rộng và cao còn lại. Trên màn hình `<= 800px`, các khối xếp dọc (khung xem ảnh trước, rồi cấu hình, rồi hàng đợi) và trang cuộn bình thường.

Khung `#image-viewport` có kích thước cố định theo CSS (bằng diện tích cột phải trừ thanh tiêu đề và chân viewer, không co giãn theo tỷ lệ từng ảnh). Ở mức zoom `100%` (`Vừa khung`), toàn bộ ảnh (kể cả ảnh dọc cao lẫn ảnh ngang rộng) đều hiển thị trọn vẹn và căn giữa. Dải phóng to `100..800%` (bước `25%`), giữ nguyên điểm neo (anchor). Thanh zoom nổi, thanh trượt `#zoom` và hai nút `+/−` đã được bỏ; thanh tiêu đề `.preview-heading` nằm ngoài ảnh (không bao giờ đè lên ảnh hay `#preview-placeholder`) chứa khối tên file + dòng meta hai dòng bên trái, cụm chuyển chế độ `#preview-mode` (`Upscale` / `Hình nền AI`, chỉ hiện khi ảnh đang xem có `wallpaper_available`, mặc định `Upscale`) ở giữa, và nhóm zoom (`#zoom-value`, `Vừa khung`, `1:1`) bên phải. Khi xem ở chế độ `Hình nền AI`, khung xem hiển thị ảnh `/api/jobs/<id>/wallpaper?inline=1` đơn lẻ (ẩn vạch chia và thanh trượt so sánh), dòng meta hiển thị `Hình nền AI · 3840 × 2160`, giữ nguyên toàn bộ thao tác zoom/pan (`Vừa khung`, `1:1` theo kích thước `3840 × 2160`, lăn chuột, phím tắt, nhấp đúp, kéo pan); khi tác vụ tạo hình nền AI của ảnh đang xem vừa hoàn tất, khung xem tự chuyển sang `Hình nền AI` một lần duy nhất. Thanh chân `.preview-footer` chứa nhóm `Kết quả` và `Hình nền AI · 3840 × 2160`.

Con lăn chuột trên `#image-viewport`: ở bố cục desktop (`> 800px`) lăn chuột trực tiếp sẽ phóng to/thu nhỏ tại vị trí con trỏ; ở bố cục xếp chồng (`<= 800px`) lăn chuột thường sẽ cuộn trang và hiện gợi ý tạm `Ctrl + lăn chuột để phóng to` trong `~1.5 s` (`pointer-events: none`), còn `Ctrl + wheel` sẽ phóng to/thu nhỏ. Vẫn giữ phím tắt (`+`/`=`/`-`/`0`, phím mũi tên pan `10%`), nhấp đúp chuyển `100%` ↔ `200%`, kéo chuột trái để pan, kéo vạch chia hoặc chuột phải để đổi vị trí so sánh.

Văn bản rút gọn: bỏ tiêu đề lớn ở đầu trang, các dòng hướng dẫn rút gọn còn một dòng ngắn hoặc `title`, dòng giới hạn hiển thị `PNG/JPG/WebP · ≤ 50 MB · ≤ 16 MP · ≤ 20 ảnh/lần`.

Hàng đợi có thanh tải hàng loạt (`Chọn tất cả`, đếm số ảnh chọn, `Kèm hình nền AI` mặc định bật, `Tải đã chọn`, `Tải tất cả`) và checkbox `Chọn <filename>` trên từng dòng đã hoàn tất, tải rời từng file cách nhau `~300 ms`. Nhãn hình nền AI chuẩn 4K (`3840 × 2160`).

Radio `2×`/`4×`/`8×` khi xem kết quả chuyển biến thể đã có, hoặc gửi `POST /api/jobs/{id}/scale` để nối FIFO tạo bản thiếu.

## Lịch sử

Yêu cầu bổ sung ngày 2026-11-11: giải thích loại ảnh; làm lựa chọn 2×/4× dễ nhìn và có kích thước pixel dự kiến, ưu tiên vùng ảnh; so sánh ảnh gốc bên trái và upscale bên phải bằng vạch chia kéo ngang.

## Phạm vi kỹ thuật

Thay đổi `web/index.html`, `web/app.js`, `web/style.css`, `web/preview.js` và `tests/test_web.py`. Không có build frontend hay dependency mới.
