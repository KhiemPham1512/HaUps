# Thử nghiệm model anime — 2026-11-12

## Phạm vi và quyết định

Người dùng giao tiếp tục thử nghiệm anime, ưu tiên chất lượng trước thời gian. App vẫn dùng NCNN/x4plus-anime, native4, TTA off, tile128 và fallback64/32 khi OOM. Nghiên cứu/thử CUDA dưới đây chưa là quyết định chuyển kiến trúc hoặc tích hợp model mới; không thay photo/API/cache/job cũ.

## Nguồn và checkpoint đã xác minh

Antigravity được gọi trước mọi lần tra/tải web, chỉ gửi câu hỏi công khai về model/version/license; không gửi ảnh, mã riêng hoặc log. Đã đối chiếu lần hai trước tải, sau đó đọc trực tiếp metadata/model card/file của repository tác giả. Không chỉ dựa vào bản tóm tắt tham khảo để tải model.

Repository [Kim2091/AnimeSharp](https://huggingface.co/Kim2091/AnimeSharp) cung cấp checkpoint safetensors. [Model card tại revision đã dùng](https://huggingface.co/Kim2091/AnimeSharp/blob/7696d95ced82b0c1f2a41f6ac73336133f0a90e1/README.md) ghi tên4x-AnimeSharp (tên cũ4x-TextSharpV1), native4, kiến trúcESRGAN, mục đích Anime/Text, license weights **CC BY-NC-SA4.0**; đây là license weights cần giữ riêng với phần mềm loader/engine. Không đóng gói hoặc phân phối lại trong lượt thử này.

Tải [4x-AnimeSharp.safetensors tại revision cố định](https://huggingface.co/Kim2091/AnimeSharp/resolve/7696d95ced82b0c1f2a41f6ac73336133f0a90e1/4x-AnimeSharp.safetensors),33,467,822bytes; SHA256 tự tính `7fc60054d291579acac4fb537efd91c80611cbd281ef8b90f434048cc1313b39`. Hash dùng nhận dạng file đã thử, không thay chữ ký/hash độc lập của nhà phát hành. Metadata/card/manifest được lưu local ở `.local-artifacts/anime-reference/candidate/`. Script chỉ tải nếu repository/license/tên file đúng và giới hạn100MB; không nhận checkpoint pickle không rõ nguồn.

Nhận xét về AnimeJaNai trong câu trả lời nghiên cứu đầu chưa được đối chiếu riêng theo version/checkpoint, nên không dùng để chọn/tải hoặc suy ra kiến trúc/license của toàn dòng model đó.

## Khả năng chạy thật

Môi trường CUDA thử riêng đã có: Python3.12.10, torch2.8.0+cu128, Spandrel0.4.1; không cài thêm dependency hoặc đổi môi trường app. Loader nhận ESRGAN/native4/input-outputRGB3kênh, hỗ trợhalf; đã inference thực FP16 trên RTX4050 Laptop. CUDA0 và Vulkan1 trong phiên này là hai hệ index riêng, không suy ra từ nhau. Inference offline, khóa `single_instance` dùng chung app.

Chín input256×256 có ảnh tham chiếu512 được chạy full-frame, kiểm output1024×1024/finite/decode/nonuniform rồi Lanczos vềoutput2=512×512, đối chiếu cùng mục tiêu với model cũ. Mean PSNR RGB(dB) chỉ trong suy giảm nhân tạo:

| Suy giảm | Current NCNN4→2 | AnimeSharp CUDA4→2 |
|---|---:|---:|
| Chỉ downsample2 |25.744|28.028|
| Downsample + blurGaussian0.7 |25.826|28.273|
| Downsample + JPEG40 |24.702|25.789|

AnimeSharp có sai số pixel thấp hơn current ở9/9ca, nhưng PSNR không đồng nghĩa ảnh đẹp hơn; tham chiếu là cropJPG có artifact, không bản vẽ gốc, và ảnh mềm/giữ nhiễu có thể được điểm cao. Model video vẫn có lợi thế điểm ở vài ca. Không tối ưu app bằng điểm số hoặc một ngưỡng tự đặt.

Đã thử thêm ba ảnh anime kích thước đầy đủ qua tile256/pad32, nguồn giữ hash, decode/dimensions/finite đạt. Thời gian tiled inference/transfer lần lượt5.867/6.836/6.585s; không gồm startup/load model/encode như thời gianCLI NCNN nên không so trực tiếp như tốc độ production. Peak bộ cấp phát torch832.7MiB allocated/2994MiB reserved trong vòng này; đây không phải peakVRAM toàn GPU hoặc phần tài nguyên thực app cần khi tích hợp.

Một crop512×512 được so full-frame với tile256/pad32: khác biệt tối đa1/255 mỗi kênh, mean absoluteRGB≈0.014–0.016, crop qua giao tile chưa thấy đường nối rõ. Điều này chỉ kiểm chứng mẫu đã chạy, chưa bảo đảm mọi ảnh lớn hoặc mọi vùng biên. Bằng chứng `candidate-real/seam.json` và `seam-comparison.png`.

## Đánh giá và bước kế

Trên crop suy giảm và ảnh thật đã xem, AnimeSharp giữ thêm nét tóc/ánh tóc và cảm giác texture so current làm phẳng. Tuy nhiên một số viền bị lặp/nét mảnh thiếu tự nhiên, vùng mắt/tóc vẫn có thể bị vẽ khác; pink có đường viền mảnh/đứt, silver có các nét tóc dày/lặp, blue vẫn đổi mi/mắt. Đây là ứng viên đáng kiểm tiếp, chưa đủ để gọi nâng chất lượng toàn app hoặc thay mặc định.

Giữ trial trong môi trường riêng. Trước đề xuất tích hợp cần gate năm nhóm độc lập/ảnh held-out, alpha và color profile, ảnh lớn, thiếuVRAM/timeout/khởi động/lỗi model/cache và benchmark chất lượng đầy đủ; không dùng alpha từ modelRGB3kênh như alpha đã hỗ trợ. Không có trialAlpha trong lượt này. Nếu muốn tích hợp CUDA thì phải nêu rõ khác kiến trúc NCNN hiện hành trước khi đổi contract/backend, không đổi từ kết quả nghiên cứu này.

Bằng chứng: `.local-artifacts/anime-reference/README.md`, candidate-run/ (9ca), candidate-real/ (3ảnh+full-vs-tile), scripts, JSON và các PNG crop100%. Không gửi ảnh/source ra ngoài. Vòng này không sửa production/tests, không chạy lại suite cấu hình trước hook.41tests của lần đổi tile trước không được gọi là test mới cho model này.
