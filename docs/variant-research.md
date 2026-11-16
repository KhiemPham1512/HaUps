# Kiểm chứng chuyển scale — 2026-11-11

- [CLI upstream](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan#full-usages) liệt kê -s2/3/4 và nhiều model; danh sách flag không chứng minh model x4plus đang cài là native2. Giữ pipeline đã kiểm chứng native4, không đổi sang animevideov3 chỉ để có native2.
- [Pillow Image](https://pillow.readthedocs.io/en/stable/reference/Image.html) mô tả resize/Lanczos và giới hạn decompression bomb. Không tắt bảo vệ toàn cục để chấp nhận input lớn. Input giới hạn7MP; native4 của7MP có112MP, output8 có448MP. Không có bằng chứng mức này an toàn tuyệt đối trên6GB VRAM/RAM hiện tại.
- Timeout phải đặt ở subprocess engine; timeout fetch chỉ giới hạn chờ HTTP, không dừng GPU. Adapter chia sẻ600s cho lần đầu và retryOOM. Resize/encode có thể thêm thời gian.
- Cache native4 và resize trực tiếp từng biến thể giữ cùng thuật toán cũ, giảm inference lặp; không suy ra tăng chi tiết hoặc đạt chất lượng tốt nhất. Cần benchmark ảnh thực đủ5nhóm, màu/alpha và peakRAM trước nghiệm thu.
