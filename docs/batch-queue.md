# Batch tuần tự — bổ sung phạm vi HaUps

## Cập nhật 2026-11-16 — bỏ tùy chọn giữ chi tiết nguồn anime

- **Bỏ `preserve_details`:** Tùy chọn `preserve_details` (“Giữ chi tiết nguồn”) và nhánh hòa trộn 50% Lanczos (`detail-native.png`) đã được gỡ bỏ vào ngày 2026-11-16 vì model anime `haups-anime-x4` tự giữ được chi tiết nguồn.
- **Tương thích ngược:** `POST /api/batches` bỏ qua trường form thừa nếu client cũ còn gửi; các bản ghi `queue.json` cũ còn khóa `preserve_details` vẫn nạp và chạy bình thường từ `native.png`.
- **Quản lý cache và chịu lỗi:** Kết quả xuất trực tiếp từ `native.png` (gồm cả hợp nhất kênh alpha bằng Lanczos 4× khi ảnh nguồn có trong suốt); lỗi ghi downstream sau khi xác minh `native.png` hợp lệ vẫn giữ nguyên source, native cache và output đã có.

## Cập nhật 2026-11-12 — tile anime đã kiểm chứng

Anime dùng x4plus-anime/native4/TTA off, tile128 và fallback128→64→32 hữu hạn chỉ khi có lỗi thiếu VRAM; photo giữ64→32. Mọi lần thử chia sẻ deadline600s, vẫn một worker. Đã so tile64/128 trên ba mẫu anime và gate năm nhóm/alpha/crop qua đường nối; chưa thấy giảm chất lượng rõ trong vùng đã xem. Thời gian trung bình ba mẫu giảm3–17%, không phải cải thiện model hoặc cam kết chung. Ca tải7MP tổng hợp native4→output2 chạy thành công; chưa7MP→8× hoặc OOM thật. Cache/job cũ không đổi. Xem `.local-artifacts/anime-focus/README.md` và README để phân biệt benchmark tài nguyên với chất lượng thật. Các ghi chú tile64 mặc định anime bên dưới là lịch sử.

## Cập nhật 2026-11-12 — lưu trữ và anime

Giữ nguyên API/scale/cache và mọi job cũ. Output mới dùng hard link cho các PNG trùng nội dung (native4/result4 và result mặc định), fallback copy nếu filesystem không hỗ trợ; thay link bằng replace để không ghi đè cache. Quota vẫn dự trù bảo thủ. Anime tắt TTA sau A/B ba mẫu thật, giữ model x4plus-anime/native4; các ghi chú TTA bật bên dưới là lịch sử. Không đưa thử nghiệm PyTorch native2 vào production vì chất lượng năm ảnh chụp chưa cải thiện nhất quán. Hướng dẫn UI khuyên bắt đầu ảnh chụp2×, nêu rõ8× có16 lần số pixel và không thêm bước phục hồi.

## Cập nhật hiện hành — viewer và scale, 2026-11-11

Giới hạn input 16MP (50 MiB/file), subprocess 600s cả retry, quota 16GiB; các mức 2MP/7MP/60s/1GiB/4GiB bên dưới là lịch sử. Native4× được giữ cache; tạo các output2/4/8 không lớn hơn scale yêu cầu. POST /api/jobs/{id}/scale với form scale=2|4|8: bản đã có trả200 ngay, bản chưa có nối cùng FIFO, vẫn một worker. Running/queued chưa có scale đó trả409, thiếu dung lượng409, ID lạ404, storage503. GET download thêm query scale=2|4|8, chỉ trả biến thể đã có; link download cũ vẫn hoạt động. State lưu available_scales qua restart; cache/output dọn cùng TTL24h. Job cũ thiếu cache có thể cần inference lại khi yêu cầu thêm scale.

Người dùng yêu cầu ngày 2026-11-11: chọn nhiều ảnh cùng lúc, AI tự xử lý từng ảnh để giảm thao tác đưa ảnh rồi chờ; không chạy nhiều inference GPU đồng thời. Yêu cầu mới đưa phần batch tuần tự của FR-09 vào phạm vi được giao. BRD Word gốc và các quyết định kiến trúc hiện có giữ nguyên; tài liệu này ghi phần bổ sung và hành vi thực tế đã triển khai, không tuyên bố nghiệm thu toàn MVP.

## Quy tắc và tiêu chí nghiệm thu

- Một lượt tải chứa nhiều ảnh dùng chung loại ảnh và scale; mỗi ảnh thành một job riêng, ID độc lập dù trùng tên. Nhận tối đa 20 ảnh/lượt và tối đa 100 ảnh chưa hoàn tất trên toàn server, còn phụ thuộc quota dung lượng.
- Một hàng đợi FIFO trên backend, một worker, một subprocess GPU tại một thời điểm. Thêm lượt mới lúc đang chạy được nối cuối hàng. Nhiều tab dùng chung hàng đợi. Khóa OS chặn server thứ hai trong cùng workspace, không chỉ khóa nút UI.
- Job: `queued` → `running` → `done` hoặc `failed`. Ảnh hỏng/không hợp lệ bị từ chối riêng lúc upload; ảnh hợp lệ vẫn nhận. Engine lỗi/timeout/thiếu VRAM chỉ làm ảnh đó lỗi, worker tiếp tục ảnh sau.
- Refresh/đóng tab không hủy job khi server còn chạy. Lưu trạng thái filesystem và output để tải lại sau restart. Shutdown bình thường chờ tác vụ hiện tại kết thúc và giữ pending; restart sau sự cố đánh dấu job đang chạy là lỗi, tiếp tục pending theo thứ tự lưu.
- Kết quả PNG chỉ tải được khi engine trả thành công, output decode được, kích thước và alpha cần thiết hợp lệ. X2 là inference x4 rồi downsample, không chạy nhiều engine song song để tạo x2.
- Bản sao upload và kết quả thuộc thư mục dữ liệu app. Không ghi đè/xóa ảnh gốc người dùng. Kết quả đã kết thúc quá 24 giờ được dọn tại startup hoặc submit tiếp theo, bỏ qua path lạ/symlink/junction; không dọn job active.
- Không thêm DB, Redis, worker service, account, cloud, hủy/reorder hoặc ZIP download. Backend FastAPI, adapter subprocess NCNN, filesystem và frontend thuần vẫn theo hướng BRD.

## Interface local mới, đã triển khai

Các endpoint phục vụ cùng origin `127.0.0.1:8765`, chưa có hợp đồng API trước đó tại `docs/api.md` để sửa.

| Interface | Hành vi |
| --- | --- |
| `GET /` và `/assets/*` | Giao diện local, không asset/CDN ngoài |
| `GET /api/capabilities` | Preset thực có executable và cặp model files, GPU id được chọn |
| `POST /api/batches` | Multipart nhiều field `files`, `scale` là `2` hoặc `4`, `preset` là `photo` hoặc `anime`; 202 gồm `jobs` nhận và `rejected` theo file |
| `GET /api/jobs` | Snapshot chung theo thứ tự thêm, trạng thái/error/thời gian/source/download URL |
| `GET /api/jobs/{id}/source` | Bản sao chuẩn hóa để preview/so sánh |
| `GET /api/jobs/{id}/download` | PNG attachment khi `done`; query `inline=1` dùng xem/so sánh |

Lỗi quan trọng: 400 host không local, 403 origin ngoài, 411 upload không có Content-Length, 413 body quá 500 MiB, 422 scale/preset/file không hợp lệ hoặc không ảnh nào được nhận, 429 vượt số lượng, 503 không ghi được trạng thái. Không expose path ảnh tùy ý qua ID. Chi tiết giới hạn upload, màu, tile, timeout, quota và cleanup trong README.

## Bằng chứng

- TDD qua HTTP/adapter: hàng đợi tuần tự/trùng tên, lỗi ảnh không dừng hàng, file hỏng và traversal, restart/download, ngăn server thứ hai, quota, cleanup chỉ file sở hữu, local origin/body cap, GPU discovery, retry OOM hữu hạn, lỗi ghi state và orientation/metadata. Mock ở subprocess/filesystem bên ngoài; fixture là ảnh tổng hợp và thư mục tạm.
- Browser smoke bằng Playwright Chromium local vì không có Browser/IAB tool: chọn hai ảnh một lần, thấy `running=1`/`queued=1`, reload vẫn thấy hai job, hai output GPU thật và tải PNG 512×512, dialog so sánh dùng được, không JS error hay network ra ngoài.
- Ảnh dùng trong smoke: hai bản sao ảnh phong cảnh mẫu 256×256 của gói upstream. Model photo, tile64, TTA off, GPU native id2; lần lượt 6,74 và 4,58 giây gồm xử lý file. Đây không phải benchmark tốc độ chung.
- Kiểm tra 375, 768, 1024, 1440px và ảnh concept 1487×1058; không viewport overflow. Evidence ở `.local-artifacts/batch-queue/`.

## Còn giới hạn

Chưa benchmark ảnh lớn, peak VRAM, chất lượng đủ năm nhóm, model anime/alpha/profile màu đa dạng; fallback OOM và timeout được kiểm tra qua boundary mock, chưa ép GPU thật vào OOM. Chưa chốt Quality/Balanced/TTA hoặc nghiệm thu AC-01–AC-07 toàn bộ. Batch giảm thời gian người dùng phải canh và chọn lại ảnh; tổng thời gian inference vẫn phụ thuộc từng ảnh và model.

## Bổ sung 8× / ngân sách một phút

Người dùng bổ sung 8× và ưu tiên chất lượng anime trong khoảng một phút/ảnh. POST /api/batches nhận thêm scale=8; AI vẫn native4×, đầu ra8× qua Lanczos. Anime TTA đang thử nghiệm, subprocess chia sẻ deadline60s kể cả retry; encode/resize có thể thêm thời gian. Tài nguyên8× dự trù340bytes/pixel đầu vào thay140 để tính cả ảnh trung gian/kết quả. Không thêm worker hoặc chạy GPU song song. Chưa nghiệm thu chất lượng ≥5 nhóm; xem README và benchmark local.

## Bổ sung bỏ ảnh — 2026-11-11

Người dùng yêu cầu bỏ ảnh cả trước upload và trong hàng đợi, mở rộng giới hạn không hủy job nêu ở phần trên. Trước upload có Bỏ chọn từng ảnh và Bỏ chọn tất cả. Sau upload có Bỏ khỏi hàng đợi từng ảnh đang chờ hoặc đã kết thúc; ảnh đang chạy khóa nút đến khi hoàn tất, chưa hỗ trợ ngắt subprocess GPU.

`POST /api/jobs/{id}/remove` trả 200 khi lưu thành công; 404 nếu không còn job, 409 nếu đang chạy, 503 nếu không lưu được. Kiểm tra origin local như POST upload. Backend giữ cùng khóa với worker để tránh race; job queued bị bỏ không chạy nữa. Snapshot/source/download không còn hiển thị job đã bỏ, kể cả sau restart. Bản sao/kết quả được giữ để dọn theo TTL 24 giờ hiện có; không xóa ảnh nguồn người dùng.
