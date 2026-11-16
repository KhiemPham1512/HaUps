# Báo cáo sàng lọc Candidate Models cho Anime Outpainting (2026-11-12)

## 1. Ngữ cảnh nghiên cứu và phạm vi mẫu thử

- **Tài liệu ghi nhớ xuyên phiên**: Xem [docs/anime-research-context.md](anime-research-context.md) (đã lưu qua `edit-3fb31e7bc13e41ca`; `AGENTS.md` bị chặn bởi allowlist của bridge và không thay đổi).
- **Mục đích nghiên cứu (user-stated intent)**: Dự án cá nhân nghiên cứu khoa học về xử lý ảnh và nâng cao chất lượng outpainting anime, bao gồm các bộ ảnh mẫu anime do người dùng mô tả, cùng quyền chủ động tải các model công khai từ nguồn chính thức (pin revision, kiểm tra license/hash/disk budget, không hỏi lại phạm vi đã duyệt).
- **Ràng buộc riêng tư & phạm vi chạy thực tế**: Chạy hoàn toàn cục bộ/offline, không tải ảnh hay ghi tên tệp cá nhân lên tài liệu/mạng, không thêm bộ lọc nội dung, không tự động thay thế production. Thư mục `images/` có `25` tệp tại thời điểm ghi nhận nhưng các tệp mới **chưa được kiểm duyệt, chọn lọc hay chạy benchmark**. Toàn bộ `68` ảnh output mới trong đợt này chỉ chạy trên **2 fixture anime mặc trang phục sẵn có** (`closeup`, `winter`), **không phải** bộ mẫu mở rộng và **không phải** tập blind holdout độc lập.

## 2. Xác minh và thiết lập hai Candidate Checkpoints

Đã tải và xác minh đủ `13` tệp/model (Diffusers `fp16` safetensors + configs/tokenizer/model cards), kiểm tra khớp kích thước byte và LFS `sha256` (hoặc Git blob metadata) công bố chính thức trước khi đóng dấu `manifest.json` (`complete=True`):

| Model ID (`--model`) | Nguồn & Revision đã pin | Tổng dung lượng | Cơ chế UNet / Inpaint | Thư mục lưu trữ cục bộ |
| :--- | :--- | :--- | :--- | :--- |
| `dreamshaper8` | `Lykon/dreamshaper-8-inpainting` (`a09e9d0bb9849aaf46c8fc59dfa604796da15a39`) | `2,134,226,801` bytes (`13` tệp) | `in_channels=9` (`native9` inpainting) | [../.local-artifacts/outpainting/candidate-models/dreamshaper8/](../.local-artifacts/outpainting/candidate-models/dreamshaper8/) |
| `anylora` | `Lykon/AnyLoRA` (`5dac4354b7f5d32f140e1f0fbc849e92b744f347`) | `2,134,196,745` bytes (`13` tệp) | `in_channels=4` (`latent-mask-blend4`, **không phải** native 9-channel) | [../.local-artifacts/outpainting/candidate-models/anylora/](../.local-artifacts/outpainting/candidate-models/anylora/) |

- **Tương thích & giấy phép**: Cả hai có `cross_attention_dim=768` (tương thích `ip-adapter-plus_sd15.safetensors`), giấy phép `CreativeML-OpenRAIL-M` theo model card chính thức đã pin (ghi nhận metadata, không phải kiểm toán pháp lý phân phối lại). Mỗi thư mục có đủ `README.md`, `metadata.json`, `manifest.json` và các tệp runtime.
- **Loại bỏ kho gated & đính chính nguồn sai**: Bỏ qua `Linaqruf/anything-v3.0` vì kho yêu cầu manual-gate (không bypass). Bác bỏ các nhận định sai trong báo cáo tìm kiếm `web-034f876b25da4ba2` (báo sai trạng thái ungated, nhầm CLIP text encoder với vision encoder, và đòi cài xFormers) dựa trên API chính thức và kết quả chạy GPU thực tế; không lặp lại tuyên bố thiếu kiểm chứng rằng AnyLoRA tự triệt tiêu nhân vật thừa hay không có vết nối.

## 3. Giao thức đối chứng và thực thi GPU (68 outputs mới)

- **Code & kiểm thử phần mềm**: Cập nhật [../.local-artifacts/outpainting/conditioning_gpu.py](../.local-artifacts/outpainting/conditioning_gpu.py) và [../tests/test_outpaint_conditioning.py](../tests/test_outpaint_conditioning.py) (`edit-63253c0606424d5a`; bridge báo `119 passed` và `Ruff E9,F` `exit 0` — kết quả unit test phần mềm, không phải Stop hook cuối lượt hay đánh giá chất lượng ảnh). CLI giữ mặc định `sd15`, thêm `--model {sd15,dreamshaper8,anylora}`, kiểm tra preflight manifest/path/hash/channels (`4` hoặc `9`), cố định `DDIMScheduler` từ [../.local-artifacts/outpainting/model/scheduler/scheduler_config.json](../.local-artifacts/outpainting/model/scheduler/scheduler_config.json) và ghi đầy đủ repo/revision/mode/hashes/scheduler SHA vào `records.json`. Không sửa `app/`, `web/`, API, queue, cache, upscale, base model hay dependencies.
- **Tham số cố định**: Khóa GPU chung `single_instance`, offline `local_files_only=True`, `fp16` safetensors, `safety_checker=None` / `feature_extractor=None` / `requires_safety_checker=False` (`nsfw_content_detected=None`; `CLIPImageProcessor` của IP-Adapter là conditioning hợp lệ), `CFG=7.0`, `30` steps, crop giả lập `20%`, margin hướng cắt (`77` px ngang, `43` px dọc), prompt `SOFT`/`NEGATIVE` từ `region_gpu`, ảnh tham chiếu chỉ lấy phần crop quan sát được letterbox nền xám `(127, 127, 127)` (không lộ vùng ẩn). Mọi output giữ nguyên SHA-256 ảnh nguồn, khớp pixel-exact trên vùng chung và xuất PNG `1920x1080` (sinh ở `768x432`). Môi trường: `diffusers 0.35.2`, `torch 2.8.0+cu128`, RTX 4050 Laptop GPU.
- **Tổng số lượt chạy (`68` outputs kỹ thuật = `4` smoke + `64` full, `exit 0`, KHÔNG phải 68 lượt đạt chất lượng)**:
  - Smoke (`2` outputs/model, timeout `300s`): [../.local-artifacts/outpainting/candidate-smoke-dreamshaper8-20261112/records.json](../.local-artifacts/outpainting/candidate-smoke-dreamshaper8-20261112/records.json) và [../.local-artifacts/outpainting/candidate-smoke-anylora-20261112/records.json](../.local-artifacts/outpainting/candidate-smoke-anylora-20261112/records.json).
  - Full (`32` outputs/model = `2` ảnh `closeup`/`winter` × `4` hướng `left`/`top`/`right`/`bottom` × `2` seeds `42`/`123` × `2` recipes `baseline` không adapter vs `ip-adapter-0.5` `scale=0.5`, timeout `900s`): [../.local-artifacts/outpainting/candidate-full-dreamshaper8-20261112/records.json](../.local-artifacts/outpainting/candidate-full-dreamshaper8-20261112/records.json) và [../.local-artifacts/outpainting/candidate-full-anylora-20261112/records.json](../.local-artifacts/outpainting/candidate-full-anylora-20261112/records.json) (cùng tệp `.log` liền kề).
  - Đối chứng lịch sử SD1.5 (`32` records, không chạy lại): [../.local-artifacts/outpainting/conditioning-anime-20261112/records.json](../.local-artifacts/outpainting/conditioning-anime-20261112/records.json).

### Bảng thời gian và bộ nhớ Torch (Full run, 16 lượt/nhóm)

*Đo quanh lời gọi pipeline + đồng bộ CUDA/offload (gồm warmup, không tính thời gian nạp model từ đĩa); bộ nhớ là Torch peak allocated/reserved, không phải tổng VRAM toàn hệ thống và không phải cam kết tốc độ chung.*

| Model | Recipe | Số lượt | Min (s) | Median (s) | Max (s) | Torch Allocated Peak (MiB) | Torch Reserved Peak (MiB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dreamshaper8` | `baseline` (no adapter) | 16 | 6.327 | 6.463 | 9.558 | 1992.913 | 2188.000 |
| `dreamshaper8` | `ip-adapter-0.5` (`0.5`) | 16 | 7.228 | 7.281 | 10.646 | 2087.618 | 2248.000 |
| `anylora` | `baseline` (no adapter) | 16 | 6.410 | 6.674 | 8.673 | 1992.869 | 2188.000 |
| `anylora` | `ip-adapter-0.5` (`0.5`) | 16 | 7.348 | 7.544 | 10.480 | 2087.574 | 2248.000 |

## 4. Đánh giá trực quan và tài liệu ảnh đối chiếu

Codex đã xem toàn bộ `16` bảng `comparison.png` mới (`8` hướng-ảnh/model), `2` ảnh crop 100% đại diện, và cả `2` bảng tổng hợp liên model (`winter-right-models.png`, `closeup-left-models.png`):

- **Bảng tổng hợp liên model (`SD15` / `DreamShaper 8` / `AnyLoRA` cùng `IP-Adapter 0.5`, `DDIM`, seeds `42` & `123`, mỗi ô thu nhỏ `640x360` toàn khung `16:9` không crop — chỉ dùng xem tổng quan bố cục, không thay thế crop 100%)**:
  - [../.local-artifacts/outpainting/model-screen-20261112/winter-right-models.png](../.local-artifacts/outpainting/model-screen-20261112/winter-right-models.png) (Codex đã xem trực tiếp).
  - [../.local-artifacts/outpainting/model-screen-20261112/closeup-left-models.png](../.local-artifacts/outpainting/model-screen-20261112/closeup-left-models.png) (Codex đã xem trực tiếp khi review cùng đủ các `comparison.png` gốc của `closeup-left`).
  - Manifest ánh xạ thư mục nguồn: [../.local-artifacts/outpainting/model-screen-20261112/manifest.json](../.local-artifacts/outpainting/model-screen-20261112/manifest.json).
- **Nhận xét `DreamShaper 8` (`native9`)**:
  - Bảng so sánh & crop đã xem: [closeup-left comparison](../.local-artifacts/outpainting/candidate-full-dreamshaper8-20261112/closeup-left/comparison.png), [closeup-left IP seed 42 seam100](../.local-artifacts/outpainting/candidate-full-dreamshaper8-20261112/closeup-left/ip-adapter-0.5-42-seam100-left.png), [winter-right comparison](../.local-artifacts/outpainting/candidate-full-dreamshaper8-20261112/winter-right/comparison.png), [winter-bottom comparison](../.local-artifacts/outpainting/candidate-full-dreamshaper8-20261112/winter-bottom/comparison.png) (cùng `closeup-top/right/bottom`, `winter-left/top`).
  - Chất màu và độ chuyển sắc hợp phong cách anime hơn SD1.5 gốc, nhưng `closeup` vẫn sinh thêm mắt/miệng lệch giải phẫu; `winter-left`/`right` ở baseline vẫn thêm người phụ; `winter-right` có IP-Adapter ở seed `42` giảm được người thừa nhưng sang seed `123` vẫn mọc thêm mặt/nhân vật; `winter-bottom` với IP-Adapter xuất hiện mảng bệt phẳng, viền khung giả và đường nối (seam) rõ.
- **Nhận xét `AnyLoRA` (`latent-mask-blend4`)**:
  - Bảng so sánh & crop đã xem: [winter-top comparison](../.local-artifacts/outpainting/candidate-full-anylora-20261112/winter-top/comparison.png), [winter-top IP seed 42 seam100](../.local-artifacts/outpainting/candidate-full-anylora-20261112/winter-top/ip-adapter-0.5-42-seam100-top.png), [winter-right comparison](../.local-artifacts/outpainting/candidate-full-anylora-20261112/winter-right/comparison.png), [closeup-left comparison](../.local-artifacts/outpainting/candidate-full-anylora-20261112/closeup-left/comparison.png) (cùng các hướng còn lại).
  - Trong lượt thử checkpoint 4 kênh với latent-mask blending và DDIM này, khả năng nối biên rất kém (quan sát thực nghiệm trên cấu hình hiện tại, không suy diễn nhân quả rằng riêng việc có 4 kênh hay mọi model 4 kênh đều gây lỗi): thường xuyên vẽ đường tốc độ (speed-lines), khung truyện/cảnh mới tách biệt hoặc nhân vật thừa với ranh giới hình chữ nhật nguồn lộ gắt; IP-Adapter chỉ kéo bảng màu/nét tóc theo mẫu nhưng không tạo được bố cục hay giải phẫu liền mạch.
- **Giới hạn phương pháp luận**: Cố định `DDIMScheduler` giúp cô lập ảnh hưởng của weights so với baseline SD1.5, nhưng **không đại diện cho cấu hình tối ưu nhất theo khuyến nghị riêng của từng tác giả** (`DreamShaper 8` khuyến nghị `DEISMultistepScheduler`; `AnyLoRA` cấu hình mặc định `PNDMScheduler` và ví dụ model card dùng `DEIS`). Không kết luận các checkpoint này kém về mặt tuyệt đối ngoài cấu hình đã đo; không dùng điểm số PSNR/SSIM hay tuyên bố khôi phục đúng phần ảnh gốc bị cắt.

## 5. Kết luận và bước tiếp theo (Chỉ phạm vi ANIME)

1. **Quyết định hiện tại**: Cả hai model mới đã tải, xác minh hash và chạy GPU thực tế hoàn tất (`68` outputs), nhưng chưa model hay recipe nào giải quyết ổn định lỗi nhân vật thừa/lệch giải phẫu qua các seed. **Không chọn model thắng cuộc và giữ nguyên production.**
2. **Bước kế tiếp cho ANIME**:
   - Kiểm tra trực tiếp các mẫu anime người dùng cung cấp tại máy cục bộ trước khi chọn đưa vào bộ thử nghiệm và xây dựng tập holdout anime độc lập.
   - Ưu tiên điều khiển có cấu trúc (mask chủ thể/nhân vật, pose/lineart/depth conditioning như ControlNet) hoặc nhánh thử nghiệm riêng với scheduler khuyến nghị của tác giả (`DEIS`) trên `DreamShaper 8` (`native9`), thay vì quét thêm prompt/seed đơn thuần.
   - Lượt này chưa tải thêm weights ControlNet hay SDXL; quyền chủ động tải model công khai từ nguồn chính thức đã được ghi nhận cho các PoC phù hợp tiếp theo mà không cần hỏi lại. Không tự động huấn luyện, không ghi đè ảnh nguồn và không xóa dữ liệu/artifact.
