**TÀI LIỆU YÊU CẦU NGHIỆP VỤ (BRD) / PHẠM VI MVP**

LOCAL AI IMAGE UPSCALER

Ứng dụng web chạy trên máy cá nhân • Tối ưu chất lượng ảnh • Không sử dụng API trả phí

|  |  |
| --- | --- |
| **Thông tin** | **Giá trị** |
| Phiên bản tài liệu | 1.0 — định hướng triển khai cho Codex CLI |
| Đối tượng sử dụng | Cá nhân, một người dùng, Windows |
| Phần cứng mục tiêu | NVIDIA GeForce RTX 4050 (xác nhận VRAM thực tế trước benchmark) |
| Nền tảng | Web local tại 127.0.0.1; không cần triển khai Internet |
| Mức ưu tiên | Chất lượng ảnh > độ ổn định > thời gian xử lý > hiệu ứng giao diện |
| Chi phí bắt buộc | 0 đồng cho API, cloud, subscription và license thương mại ở phạm vi cá nhân |

# 1. Tầm nhìn sản phẩm

Xây dựng một công cụ giúp người dùng nâng độ phân giải và cải thiện độ rõ nét của ảnh ở mức tốt nhất có thể trên RTX 4050, sử dụng hoàn toàn các công nghệ/model miễn phí, xử lý offline và không chuyển ảnh lên máy chủ bên thứ ba. Ứng dụng cần hoàn thiện phiên bản hữu dụng trong thời gian ngắn, tránh thiết kế quá mức hoặc phát triển những chức năng không trực tiếp tăng chất lượng đầu ra.

Nguyên tắc cốt lõi: “Rõ nét nhất có thể” nghĩa là chọn cấu hình/model phù hợp theo từng loại ảnh và hạn chế artefact; không phải cam kết phục hồi chính xác những chi tiết đã mất. Không được làm sắc nét giả quá mức, tạo khuôn mặt sai hoặc quảng cáo “khôi phục 100%”.

# 2. Mục tiêu và tiêu chí thành công

* Sản phẩm chạy được tại localhost trên Windows, khởi động bằng 1–2 lệnh hoặc một script .bat; không yêu cầu tài khoản hay internet sau khi đã cài phụ thuộc/model.
* Người dùng tải ảnh, chọn cấu hình chất lượng, nhấn Upscale, xem so sánh và tải ảnh đầu ra; không cần thao tác dòng lệnh trong lúc sử dụng.
* Mặc định ưu tiên chất lượng, có chế độ cân bằng tốc độ; kết quả không bị nhòe, quá sắc, sai màu hoặc xuất hiện artefact nghiêm trọng đối với tập ảnh kiểm thử.
* Không gọi API bên ngoài, không gửi ảnh qua Internet, không phát sinh chi phí theo số ảnh; mọi tệp tạm được quản lý/xóa đúng cách.
* MVP đạt nghiệm thu trong mục tiêu 5–8 ngày làm việc tập trung; đây là khung kế hoạch ước lượng, phụ thuộc môi trường, driver và chất lượng ảnh đầu vào, không phải cam kết cố định.

# 3. Phạm vi chức năng MVP

|  |  |  |
| --- | --- | --- |
| **Mã** | **Yêu cầu** | **Ưu tiên / nghiệm thu** |
| FR-01 | Chọn/kéo thả một ảnh PNG, JPG hoặc WebP; hiển thị thông tin kích thước, định dạng, dung lượng. | P0 — thông báo lỗi dễ hiểu nếu tệp hỏng/không hợp lệ. |
| FR-02 | Upscale x2 và x4; x3 nếu engine hỗ trợ hoặc thông qua resize sau xử lý, giải thích rõ. | P0 — kích thước đầu ra đúng theo lựa chọn, xử lý giữ tỷ lệ. |
| FR-03 | Chọn preset: Photo/General, Anime/Illustration; đặt mặc định Auto/Recommended khi có mapping phù hợp. | P0 — chỉ hiển thị các model đã cài/được engine hỗ trợ. |
| FR-04 | Preset Quality (ưu tiên chất lượng) và Balanced (ưu tiên tốc độ); tile-size có giá trị an toàn. | P0 — tránh treo với ảnh lớn và báo rõ lỗi VRAM. |
| FR-05 | Thanh trạng thái “đang xử lý”, thời gian hoàn thành và thông báo thất bại; khóa yêu cầu chạy trùng. | P0 — không tạo nhiều tiến trình GPU ngoài ý muốn. |
| FR-06 | So sánh Before/After bằng thanh trượt, zoom cơ bản, xem kích thước ảnh. | P0 — xem được sau khi upscale thành công. |
| FR-07 | Tải xuống file PNG hoặc định dạng xuất hỗ trợ, đặt tên theo ảnh gốc và hệ số. | P0 — file mở được và không ghi đè ảnh nguồn. |
| FR-08 | Lưu ảnh tạm trong thư mục riêng và dọn dẹp có kiểm soát. | P0 — không để tăng dung lượng ổ đĩa vô hạn. |
| FR-09 | Batch processing, hủy tác vụ, nhiều model cao cấp, phục hồi khuôn mặt. | P1/P2 — không chặn phát hành MVP. |

# 4. Yêu cầu chất lượng ảnh và khả năng kiểm chứng

Mỗi lần nâng cấp thuật toán phải so sánh trên bộ ảnh mẫu: ảnh chân dung, phong cảnh, chữ/đồ họa, anime và ảnh nén JPEG hoặc ảnh có nhiễu. Giữ bản gốc, không thay thế bằng ảnh đã xử lý.

* Đánh giá trực quan ở mức zoom 100% và cùng vùng crop: độ sắc nét, viền/răng cưa, texture, lỗi da/mặt, noise, halo, sai màu và chi tiết giả.
* Chọn model dựa trên nội dung. Không giả định một model cho kết quả tốt nhất trên mọi hình ảnh.
* Bảo toàn tỷ lệ, không làm méo ảnh; dữ liệu EXIF/định hướng cần được xử lý nhất quán; ảnh RGBA cần bảo toàn alpha nếu engine thực tế hỗ trợ.
* Không ép tất cả ảnh qua các bộ lọc sharpen/denoise mạnh. Các công đoạn hậu xử lý phải có thể tắt và chỉ thêm khi chứng minh cải thiện.
* Không đặt mục tiêu PSNR/SSIM cố định cho ảnh thực tế thiếu ground truth; ghi nhận đánh giá bằng bộ ảnh và log thông số chạy.

# 5. Kiến trúc kỹ thuật khuyến nghị

|  |  |  |
| --- | --- | --- |
| **Lớp** | **Giải pháp** | **Lý do** |
| Frontend | HTML + CSS + JavaScript thuần (hoặc React nếu repo đã có) | Giảm thời gian dựng MVP; dễ nâng cấp UI về sau. |
| Backend | Python 3.11 + FastAPI + Uvicorn | API upload, điều phối và trả file; nhẹ, dễ bảo trì. |
| AI engine chính | Real-ESRGAN NCNN/Vulkan bản phát hành Windows | Sử dụng RTX 4050, không yêu cầu thiết lập PyTorch/CUDA ngay từ đầu. |
| Xử lý ảnh | Pillow + xử lý file tạm | Kiểm tra định dạng/kích thước, preview và hỗ trợ output. |
| Lưu trữ | Filesystem local: uploads/, outputs/, temp/ | Không cần database, không cần máy chủ. |
| Triển khai | 127.0.0.1 + start.bat | Chỉ máy cá nhân truy cập mặc định. |

Tách EngineAdapter khỏi API và giao diện để sau này thử PyTorch/CUDA, ONNX hoặc model khác mà không viết lại ứng dụng. Tất cả công cụ/model sử dụng phải được kiểm tra giấy phép cụ thể trước khi đóng gói/phân phối. Không cần huấn luyện model từ đầu.

# 6. Yêu cầu phi chức năng

|  |  |
| --- | --- |
| **Mã** | **Ràng buộc** |
| NFR-01 | Không phụ thuộc cloud, API key, subscription hay dịch vụ xử lý ảnh online; mọi inference chạy local. |
| NFR-02 | Chỉ bind 127.0.0.1, giới hạn định dạng và kích thước upload, tránh path traversal, tên file phải được chuẩn hóa. |
| NFR-03 | Có timeout/xử lý lỗi tiến trình con; không để tiến trình mồ côi; không log nội dung ảnh hoặc đường dẫn nhạy cảm. |
| NFR-04 | Có phương án giảm tile khi VRAM không đủ; không đưa ra cam kết tốc độ cố định trước khi benchmark RTX 4050 thực tế. |
| NFR-05 | README có hướng dẫn cài đặt, chạy, lỗi phổ biến; file requirements có phiên bản phụ thuộc được kiểm soát. |
| NFR-06 | UI responsive cho desktop, tiếng Việt, dễ hiểu; trạng thái loading, lỗi, hoàn thành rõ ràng. |

# 7. Kịch bản nghiệp vụ chính (User flow)

Mở ứng dụng → Chọn/kéo thả ảnh → Kiểm tra file → Xem preview/kích thước gốc → Chọn loại ảnh + tỷ lệ x2/x4 + chế độ Quality/Balanced → Nhấn Upscale → Backend tạo job và chạy AI engine → Theo dõi trạng thái → Xem so sánh gốc/kết quả → Tải file → Dọn file tạm.

Ngoại lệ phải xử lý: tệp không hợp lệ; thiếu binary/model; Vulkan/driver chưa sẵn sàng; GPU hết VRAM; ảnh quá lớn; tiến trình AI trả mã lỗi; thiếu dung lượng ổ cứng; làm mới trang trong lúc xử lý. Không được báo “thành công” khi file kết quả chưa được tạo/kiểm tra.

# 8. Kế hoạch triển khai nhanh (timebox)

|  |  |  |
| --- | --- | --- |
| **Giai đoạn** | **Ước lượng** | **Kết quả bàn giao** |
| Ngày 1 | 0,5–1 ngày | Khởi tạo repo, môi trường, engine; upscale thành công 1 ảnh bằng CLI. |
| Ngày 2–3 | 1–2 ngày | FastAPI upload/process/download, kiểm tra file, lỗi rõ ràng. |
| Ngày 4–5 | 1–2 ngày | Frontend lựa chọn ảnh/model/tỷ lệ; preview, Before/After, tiến trình. |
| Ngày 6 | 0,5–1 ngày | Test với tập ảnh đa dạng, điều chỉnh preset/tile theo GPU. |
| Ngày 7–8 (dự phòng) | 0–2 ngày | Sửa lỗi, README, script start.bat, nghiệm thu MVP. |

Quản trị phạm vi: nếu một tính năng không giúp đạt luồng “chọn ảnh → upscale → so sánh → tải ảnh”, chuyển sang backlog. Không triển khai đăng nhập, cơ sở dữ liệu, cloud deployment, payment, huấn luyện model hoặc batch trong MVP.

# 9. Tiêu chí nghiệm thu (Definition of Done)

* AC-01: Trên Windows có RTX 4050, chạy hướng dẫn trong README và truy cập được localhost mà không cần tài khoản hoặc API key.
* AC-02: Ít nhất 5 ảnh thử thuộc các nhóm khác nhau được xử lý thành công, có kết quả xem/tải được; ghi log thông số từng lần.
* AC-03: Các mức x2 và x4 cho kích thước phù hợp; model không hỗ trợ mức nào thì UI không đưa ra lựa chọn sai.
* AC-04: So sánh Before/After thao tác được; không tự đổi ảnh gốc và không ghi đè file gốc.
* AC-05: Lỗi model thiếu, upload sai, VRAM/engine lỗi được thông báo bằng tiếng Việt; không crash server.
* AC-06: Test bảo mật cơ bản cho upload, đường dẫn và local-only; không phát sinh kết nối tới AI API bên ngoài.
* AC-07: Có tài liệu setup, lệnh chạy, cấu trúc dự án, phương pháp thêm model và danh sách hạn chế chưa giải quyết.

# 10. Chỉ dẫn giao việc cho Codex CLI

Dùng tài liệu này làm nguồn yêu cầu nghiệp vụ; Codex phải khảo sát repo hiện có, lập kế hoạch ngắn, triển khai từng milestone có thể chạy, test và báo cáo bằng chứng. Không tự mở rộng phạm vi hoặc cài API cloud. Ưu tiên dùng lại cấu trúc dự án đang có và tuân theo AGENTS.md của repo khi không mâu thuẫn với yêu cầu nghiệp vụ.

Prompt khởi đầu đề xuất cho Codex CLI:

“Đọc BRD\_AI\_Image\_Upscaler\_Local\_Codex\_CLI.docx và AGENTS.md của repo. Hãy xây dựng MVP Local AI Image Upscaler theo BRD, ưu tiên chất lượng ảnh, miễn phí, offline trên RTX 4050. Trước tiên kiểm tra repository, đề xuất kế hoạch 5–8 ngày công (không cam kết thời gian), bắt đầu bằng proof-of-concept Real-ESRGAN NCNN/Vulkan. Triển khai từng milestone nhỏ, chạy test ở mỗi bước, chỉ nhận hoàn thành khi thỏa các tiêu chí AC-01 đến AC-07. Không thêm tính năng ngoài MVP nếu chưa cần.”

# 11. Quyết định và rủi ro cần kiểm tra sớm

* RTX 4050: kiểm tra chính xác VRAM, driver NVIDIA và Vulkan; không giả định hiệu năng trước khi đo.
* Real-ESRGAN NCNN hỗ trợ một tập model/preset cụ thể; chỉ đưa ra lựa chọn thực sự cài và chạy được.
* Một số ảnh rất mờ, nhiễu nặng hoặc chứa chữ nhỏ có thể không phục hồi đáng tin; cần thử nghiệm và nêu giới hạn rõ ràng.
* Chất lượng cao nhất không đồng nghĩa tỷ lệ upscale lớn nhất; chất lượng phụ thuộc model, dữ liệu đầu vào, resize và artefact.
* Nếu bộ engine không thể đạt mục tiêu chất lượng trên nhóm ảnh chính sau benchmark, mới cân nhắc nâng sang pipeline PyTorch/CUDA ở giai đoạn 2.

Trạng thái: BRD khởi tạo — có thể giao Codex CLI triển khai. Mọi thay đổi phạm vi phải cập nhật tiêu chí nghiệm thu tương ứng.