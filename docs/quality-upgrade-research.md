# Kiểm chứng phương án nâng xử lý — 2026-11-12

Người dùng giao nâng xử lý vì8× ảnh người nặng mà không đạt kỳ vọng; cho phép tra nguồn chính thức sau khi Antigravity timeout90s+150s, rồi cho phép thử PyTorch/CUDA trong môi trường riêng. Ảnh/source riêng chỉ xử lý local.

## Bằng chứng công khai

- [Model Zoo Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md) công bố photo native2 `RealESRGAN_x2plus`; General x4v3 nhỏ nhưng khả năng khử mờ/nhiễu hạn chế; RealESRNet có xu hướng quá mịn. Không suy ra native2 đẹp hơn model4 từ tên.
- [Inference v0.3.0](https://github.com/xinntao/Real-ESRGAN/blob/v0.3.0/inference_realesrgan.py) dùng RRDB scale2 cho x2plus; General nội suy weights strong/weak theo denoise strength. `face_enhance` là nhánh riêng, không bật trong thử nghiệm. `outscale` khác native scale. [Release](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.3.0) mô tả General/DNI, không phải bằng chứng rằng binary NCNN20220424 hỗ trợ các weights PyTorch này.
- [Spandrel](https://github.com/chaiNNer-org/spandrel) cung cấp loader/architecture RRDB và SRVGG, không cung cấp pipeline tiled hoàn chỉnh. Thử nghiệm dùng0.4.1, torch2.8.0+cu128/torchvision0.23.0 theo [cặp bản Windows chính thức](https://pytorch.org/get-started/previous-versions/#v280). Không cài BasicSR/GFPGAN hoặc sửa môi trường app.
- [License Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/v0.3.0/LICENSE) là BSD3-Clause; Spandrel MIT. Weights lấy đúng link official từ model zoo/inference. Manifest local lưu URL, kích thước và hash tự tính; đây không phải chữ ký/hash phát hành độc lập hoặc audit license từng thành phần để phân phối. Chỉ dùng riêng local.
- [NCNN CLI/source](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/blob/master/src/main.cpp) phân biệt model anime video có file theo scale với x4plus. Khả năng của master không tự chứng minh binary đang cài; native2 anime đã chạy thật bằng files đi kèm binary. [RealSR NCNN](https://github.com/nihui/realsr-ncnn-vulkan) cũng có binary Vulkan khác, nhưng chưa tải/chạy và không chọn làm nâng cấp trong lượt này.

## Kết luận dựa trên chạy thật

Bằng chứng chi tiết tại `.local-artifacts/quality-upgrade/README.md`. Native2 CUDA và General DNI0.5 nhanh hơn lượt NCNN cũ trên bộ ảnh, nhưng chưa cải thiện chất lượng khuôn mặt nhất quán. Không chuyển engine mặc định, không thêm preset chưa đạt. Anime so TTA on/off cùng model trên ba phong cách: không thấy lợi ích rõ của TTA, nên tắt mặc định để giảm chi phí. Không đổi sang model video chỉ vì nhanh hơn.

Hướng tiếp theo cho chất lượng ảnh chụp là đánh giá pipeline khử mờ chuyên biệt ở cùng kích thước đích với crop khuôn mặt/tóc, kiểm seam và độ tự nhiên trước tích hợp. Đây chưa là quyết định tải/chọn model khử mờ hoặc bật face restoration. Cần thêm bộ chữ/đồ họa, cảnh quan độc lập, profile màu/alpha và kiểm cold/warm lặp lại trước nghiệm thu toàn bộ pipeline.
