# Ngữ cảnh nghiên cứu xử lý ảnh Anime (Ghi nhận theo yêu cầu người dùng)

## 1. Mục đích dự án và phạm vi mẫu thử (User-stated intent)

- **Mục đích do người dùng xác định**: Đây là dự án cá nhân phục vụ nghiên cứu khoa học về xử lý ảnh và nâng cao chất lượng outpainting cho ảnh anime; người dùng yêu cầu lưu ghi nhớ mục đích này xuyên suốt các phiên làm việc.
- **Bối cảnh mẫu thử do người dùng khai báo**: Các mẫu mới được thêm vào là nhân vật hư cấu anime do người dùng chọn và duyệt.
- **Nguyên tắc ghi nhận**: Nội dung trên được lưu như mục đích và mô tả do người dùng cung cấp (user-stated intent), không phải tuyên bố đã xác minh độ tuổi/nội dung của mọi tệp trên đĩa và không phải phê duyệt tự động vô điều kiện cho mọi nội dung trong tương lai.
- **Quy trình chọn tác vụ**: Phải xem xét mẫu thực tế ở môi trường cục bộ trước khi chọn thao tác hoặc công cụ xử lý phù hợp. Không đưa dữ liệu ảnh, tên tệp mẫu hay nội dung nhạy cảm vào tài liệu.

## 2. Ràng buộc vận hành, bảo vệ dữ liệu và quyền riêng tư

- **Cục bộ / Offline**: Toàn bộ quy trình xử lý và đánh giá ảnh thực hiện cục bộ (local/offline). Tuyệt đối không gửi ảnh mẫu lên web, Antigravity hay bất kỳ dịch vụ bên thứ ba nào.
- **Không dùng bộ lọc nội dung**: Giữ nguyên thiết lập hiện có không bật hoặc thêm bộ lọc nội dung (`safety_checker=None`, `feature_extractor=None`, `requires_safety_checker=False`). Làm rõ: đây là các tham số cấu hình lúc nạp base model (`from_pretrained`); riêng `CLIPImageProcessor` do `IPAdapterMixin` sử dụng cho nhánh `ip_adapter_image` là bước tiền xử lý conditioning hợp lệ, **không phải** bộ lọc nội dung hay bộ lọc chặn nội dung.
- **Bảo vệ nguồn và kiểm tra tài nguyên**: Duy trì đầy đủ việc kiểm tra hợp lệ tệp/tài nguyên, giới hạn bộ nhớ/đĩa và bảo vệ nguyên vẹn vùng ảnh gốc (source protection).

## 3. Quyền tải chủ động model checkpoint phục vụ nghiên cứu Anime

- **Phê duyệt từ người dùng**: Người dùng đã cho phép chủ động tải các checkpoint model công khai phù hợp cho nghiên cứu anime từ các nguồn chính thức mà không cần hỏi lại đối với các lượt tải thuộc phạm vi đã được ủy quyền này.
- **Điều kiện bắt buộc khi tải**:
  - Chỉ tải từ nguồn chính thức công khai, có pin revision rõ ràng.
  - Kiểm tra giấy phép (license), kích thước và mã băm (SHA-256) đầy đủ.
  - Tuân thủ ngân sách dung lượng đĩa (disk budget) của dự án.
- **Giới hạn ủy quyền**: Quyền này không bao gồm tải dữ liệu lên đám mây/dịch vụ riêng tư, không cho phép làm hỏng môi trường thực thi hiện có và không tự động thay thế model/cấu hình của hệ thống production.

## 4. Cập nhật bộ mẫu `images/` và ủy quyền đánh giá cải thiện upscale (2026-11-13)

Người dùng thông báo đã xóa 5 ảnh từng bị loại trừ trước đây và cho phép sử dụng các ảnh còn lại để cải thiện chất lượng upscale trong tương lai. Tại thời điểm ghi nhận ban đầu (`edit-bd832e68c01f4caf`), kiểm tra cục bộ tại `images/` xác nhận có `20` tệp thông thường; ở lần kiểm đếm chỉ đọc (read-only) cuối cùng sau khi chạy đánh giá, thư mục `images/` có `15` tệp thông thường (số lượng tệp trong thư mục làm việc có thể thay đổi theo thao tác bên ngoài; không suy đoán ai đã thay đổi thư mục và agent không xóa hay khôi phục tệp nào). Cả 5 tệp từng bị loại trừ đều không còn tồn tại, đồng thời 2 ảnh gốc anime mặc trang phục được chọn cùng các bản sao fixture RGB vẫn khớp mã băm (hash-identical) trước và sau khi chạy GPU (không liệt kê tên tệp gốc, đường dẫn riêng tư hay mô tả chi tiết nội dung ảnh tại đây).

Các ảnh còn lại trong `images/` được ghi nhận là mẫu đánh giá và đối chuẩn cục bộ (local evaluation/benchmark examples) hợp lệ cho công việc cải thiện chất lượng upscale về sau (bao gồm ưu tiên cho ảnh anime), với điều kiện vẫn phải qua kiểm tra thực tế về nội dung, định dạng, giới hạn tài nguyên và quyền sử dụng hợp pháp của người dùng. Toàn bộ tệp ảnh gốc phải được giữ nguyên bất biến (immutable) và tuyệt đối không tải lên bất kỳ dịch vụ bên ngoài nào.

Phạm vi ủy quyền này được lưu giữ cho việc đo lường, so sánh và đánh giá chất lượng upscale trong tương lai, không đồng nghĩa với việc tự động huấn luyện (automatic training) trên mọi tệp hay khôi phục các ảnh đã xóa. Công việc hiện tại chỉ tập trung vào đánh giá outpainting anime (`DEIS` và bộ mẫu `evaluation`); lượt công việc này không thay đổi thuật toán upscale và không thực hiện bất kỳ đợt huấn luyện nào.


