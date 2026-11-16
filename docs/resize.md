# TASK: Integrate 3840×2160 Wallpaper Export into Existing AI Image Upscaler

> Làm rõ ngày 2026-11-12 (cập nhật kích thước 2026-11-16): đặc tả bên dưới chỉ mô tả **Padding**, nên ảnh dọc vẫn có dải nền hai bên. Yêu cầu hiện hành là dùng ảnh dọc làm hình nền phủ 3840×2160 (thay cho 1920×1080 trước đây để giữ chi tiết nhân vật) với ít biến dạng; nền mờ, crop và AI outpainting được phân tích trong [nghiên cứu chi tiết anime và hình nền](anime-detail-wallpaper-research.md). Chưa chọn cách mở rộng đặc tả hoặc triển khai tính năng; nền mờ không phải cảnh mới liền mạch, outpainting cần thêm AI và khác FR-02/NFR hiện tại.

## 1. Context

Dự án hiện tại là ứng dụng AI Image Upscaler chạy Web Local trên Windows, sử dụng GPU NVIDIA RTX 4050, backend Python và AI engine hiện có.

**Yêu cầu:** Tích hợp chức năng xuất ảnh 3840×2160 bằng Padding sau khi AI upscale hoàn tất.

Đây là nhiệm vụ mở rộng dự án hiện có, KHÔNG phải xây dựng lại ứng dụng.

## 2. Mandatory Instructions

1. Đọc toàn bộ `AGENTS.md` và những hướng dẫn liên quan trước khi thực hiện.
2. Tuân thủ các coding conventions, skills, hooks và quy trình kiểm thử hiện có.
3. Kiểm tra cấu trúc source code, frontend, backend, pipeline upscale và các bài test.
4. Không thay thế hoặc viết lại những thành phần đang hoạt động tốt nếu không cần thiết.
5. Giữ nguyên khả năng xuất ảnh ở độ phân giải gốc sau upscale.
6. Không thêm API trả phí, dịch vụ cloud hoặc dependency không cần thiết.
7. Không tự ý thay đổi AI model hoặc cấu hình GPU.
8. Ưu tiên giải pháp đơn giản, dễ bảo trì, có thể triển khai nhanh.

## 3. Functional Requirements

### FR-01: Output Resolution

Bổ sung tùy chọn xuất ảnh:

- Original Upscaled Resolution (mặc định).
- Full HD 1920×1080 — Padding.

Đảm bảo chức năng upscale hiện tại vẫn hoạt động bình thường khi không chọn Full HD.

### FR-02: Aspect Ratio Preservation

Khi chọn Full HD Padding:

- Không crop hình ảnh.
- Không kéo giãn hoặc làm méo ảnh.
- Không thay đổi tỷ lệ khung hình của nội dung.
- Không áp dụng thêm AI upscale ở bước Padding.
- Đầu ra phải có kích thước chính xác 1920×1080 pixel.

Tính toán tỷ lệ:

`scale = min(1920 / source_width, 1080 / source_height)`

`new_width = round(source_width * scale)`

`new_height = round(source_height * scale)`

Đảm bảo kích thước sau làm tròn không vượt quá kích thước đích.

Căn giữa nội dung trên canvas 1920×1080.

### FR-03: Image Quality

Sử dụng Pillow với `Image.Resampling.LANCZOS` để resize chất lượng cao.

Yêu cầu:

- Chỉ resize một lần sau khi AI upscale.
- Không tạo thêm JPEG compression không cần thiết.
- Ưu tiên PNG lossless.
- Bảo toàn màu sắc và alpha channel nếu định dạng xuất hỗ trợ.
- Xử lý đúng EXIF orientation nếu pipeline hiện tại chưa xử lý.
- Tránh conversion RGB/RGBA không cần thiết.

Lưu ý: Không thể bảo đảm khôi phục chính xác chi tiết vốn không tồn tại trong ảnh gốc.

### FR-04: Padding Background

Hỗ trợ tối thiểu:

- Black background (#000000).
- White background (#FFFFFF).
- Transparent background (PNG).

Đề xuất mặc định: Black.

Đảm bảo transparent padding thực sự giữ được alpha channel.

### FR-05: Frontend Integration

Thêm các điều khiển vào giao diện hiện tại:

**Output Resolution**
- Original (Default).
- Full HD 1920×1080 (Padding).

**Padding Background**
- Black.
- White.
- Transparent.

Chỉ hiển thị hoặc kích hoạt Padding Background khi Full HD được chọn.

Giữ nguyên thiết kế UI/UX hiện tại. Không xây dựng lại frontend.

Hiển thị kích thước ảnh đầu ra sau xử lý.

### FR-06: Backend Integration

Tích hợp bước post-processing sau khi Real-ESRGAN upscale thành công.

Pipeline:

`Upload → AI Upscale → Optional Full HD Padding → Preview → Download`

Yêu cầu:

- Tách logic Padding thành module/function độc lập.
- Validate kích thước ảnh và tham số người dùng.
- Không ghi đè ảnh gốc.
- Không sửa hoặc ghi đè kết quả upscale gốc.
- Mỗi tác vụ sử dụng đường dẫn tạm an toàn, tránh xung đột tên file.
- Không cho phép path traversal từ tham số người dùng.
- Đóng file handle và xử lý file tạm đúng quy trình hiện tại.
- Xử lý lỗi bằng thông báo rõ ràng.
- Giữ nguyên API contract hiện tại nếu có thể; chỉ mở rộng theo hướng tương thích ngược.

## 4. Non-Functional Requirements

- 100% offline.
- Không cần API key.
- Không gửi hình ảnh lên server bên ngoài.
- Không yêu cầu GPU chạy thêm tác vụ AI cho bước Padding.
- Tối ưu thời gian xử lý và mức sử dụng RAM.
- Tránh thêm framework hoặc thư viện nặng.
- Không tăng độ phức tạp kiến trúc khi không cần thiết.

## 5. Test Requirements

Viết unit tests cho module Padding với các trường hợp:

1. 1600×1200 → 1920×1080, nội dung 1440×1080, padding trái/phải 240 px.
2. 1080×1920 → 1920×1080, giữ đúng tỷ lệ ảnh dọc.
3. 1920×1080 → 1920×1080, không thay đổi bố cục.
4. 3840×2160 → 1920×1080, resize chính xác.
5. 1000×1000 → 1920×1080, padding cân đối.
6. PNG có alpha → Transparent Padding, alpha được bảo toàn.
7. Ảnh RGB → Black/White Padding, không bị biến đổi màu không cần thiết.
8. File đầu vào không hợp lệ → xử lý lỗi an toàn.
9. Tham số resolution/background không hợp lệ → validation error rõ ràng.
10. Original Upscaled Resolution → không có thay đổi hành vi so với trước.

Thực hiện integration tests cho backend API và kiểm tra rằng frontend gửi đúng tham số.

Nếu môi trường không có GPU để kiểm thử, phải tách unit tests của Padding khỏi AI inference và báo rõ phần chưa xác minh.

## 6. Acceptance Criteria

Nhiệm vụ chỉ hoàn thành khi:

- Xuất được ảnh chính xác 1920×1080.
- Không crop hoặc kéo giãn nội dung ảnh.
- Padding đúng màu được lựa chọn.
- Transparent Padding giữ nguyên alpha.
- AI upscale vẫn hoạt động theo pipeline cũ.
- Original export vẫn giữ nguyên hành vi.
- Preview và Download hiển thị/sử dụng đúng file kết quả.
- Các bài kiểm thử mới chạy thành công.
- Các bài kiểm thử hiện có không bị regression.
- Không phát sinh dependency trả phí.

## 7. Implementation Workflow

Thực hiện theo trình tự:

1. Inspect repository, AGENTS.md, skills và existing tests.
2. Identify current upscale pipeline and API boundaries.
3. Viết tests trước cho logic Padding theo TDD nếu skill hiện có yêu cầu.
4. Implement image post-processing module.
5. Integrate backend API theo hướng backward-compatible.
6. Integrate frontend controls.
7. Run unit tests, integration tests và regression tests.
8. Review security, performance và code quality.
9. Update README/documentation.
10. Báo cáo kết quả và những phần chưa thể kiểm chứng.

Không thực hiện refactor lớn ngoài phạm vi nhiệm vụ.

## 8. Final Report

Sau khi hoàn tất, báo cáo bằng tiếng Việt:

- Những file đã tạo hoặc chỉnh sửa.
- Các tính năng đã tích hợp.
- Những bài kiểm thử đã chạy và kết quả.
- Vấn đề phát hiện và cách khắc phục.
- Cách chạy và sử dụng tính năng.
- Những giới hạn còn tồn tại.

**Priority: Image Quality → Backward Compatibility → Reliability → Simplicity → Performance.**

Mục tiêu là hoàn thiện tính năng nhanh, ổn định và không làm hỏng những chức năng hiện có.
