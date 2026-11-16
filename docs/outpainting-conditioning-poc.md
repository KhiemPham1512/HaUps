# Kế hoạch PoC Conditioning Tham chiếu cho Outpainting (Scope: ANIME)

## 1. Phạm vi, Nguồn đối chiếu và Nguyên tắc bất biến

- **Phạm vi chỉ ANIME**: Tài liệu này chỉ phục vụ bài toán vẽ tiếp nhân vật **ANIME** bị cắt ở mép ảnh theo chỉ đạo mới nhất của người dùng. Không benchmark Photo, landscape, text, JPEG30 hay mở rộng ra năm nhóm trong công việc này.
- **Trạng thái phê duyệt**: Người dùng mới chỉ cho phép đọc API công khai để tra cứu mã phiên bản/hash, **chưa phê duyệt tải weights** và chưa chạy thử nghiệm model mới.
- **Nguồn đã đối chiếu**: Báo cáo docs bridge `docs-2f98ac484abc40fb` (`COMPLETE`), Codex đã đọc `documents.md` và đối chiếu trực tiếp các mục gốc 2, 3.3, 3.4, 3.6 trong [outpainting-learning-research.md](./outpainting-learning-research.md), kết hợp kiểm tra mã nguồn hiện hành tại [../app/outpaint_runner.py](../app/outpaint_runner.py) và lịch sử bàn giao [../.local-artifacts/handoff.md](../.local-artifacts/handoff.md).
- **Yêu cầu bắt buộc giữ nguyên**:
  - Chạy riêng tư, local/offline (`HF_HUB_OFFLINE="1"`, `TRANSFORMERS_OFFLINE="1"`, `local_files_only=True`), không gửi ảnh ra ngoài, không cần ô nhập prompt.
  - **Không thêm hoặc bật lại bộ phân loại nội dung** (`safety_checker=None`, `requires_safety_checker=False`).
  - Bảo toàn ảnh nguồn bất biến; chỉ vẽ nối tiếp phần nhân vật bị cắt ở mép khung hình.
- **Hiện trạng baseline và lý do chuyển bước**:
  - Production ([../app/outpaint_runner.py](../app/outpaint_runner.py)) giữ nguyên: base model `stable-diffusion-v1-5/stable-diffusion-inpainting` revision `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb` (UNet 9-channel), `guidance_scale=7.0` (CFG 7), `num_inference_steps=30`, suy luận trên canvas `768x432` rồi `restore` ra PNG `1920x1080`.
  - Các thử nghiệm P0 gồm placement geometry (`placement_poc.py`), hướng dẫn mềm (`soft-prompt`) và mask hình học hai lượt (`region_poc.py` / `region_gpu.py`) đã chạy GPU nhưng chưa giải quyết được tính đồng nhất nhân vật (identity) và lỗi sinh thêm người thừa. Không tiếp tục quét seed/prompt vô hạn.
  - Hai bộ ảnh anime đã dùng (`closeup` và `winter`) là dữ liệu phát triển đã qua sử dụng, **không phải blind holdout**.

---

## 2. Kiểm kê Hiện trạng (Inventory) và Tương thích Kỹ thuật

### Sự thật đã xác minh (Facts)
- **Kiểm kê kho lưu trữ local**: Rà soát các thư mục runtime đã biết gồm `.local-artifacts/outpainting/model`, `.local-artifacts/quality-upgrade/models`, `.local-artifacts/anime-models`, `.local-artifacts/anime-reference` và `.local-artifacts/outpainting/cache/hub` **không thấy** weights của IP-Adapter, ControlNet hay CLIPVision image encoder.
- **Giới hạn phạm vi kiểm kê**: Một số thư mục fixture cũ báo `access denied` và cache mức user ngoài phạm vi không truy cập được, do đó không khẳng định tuyệt đối toàn máy không có bản sao nào. Không thực hiện băm (hash) lại toàn bộ base weights trong lượt này.
- **Không tương thích chéo với `safety_checker`**: Các tệp weights `safety_checker` còn lưu trên đĩa **không phải** là CLIP image encoder tương thích cho IP-Adapter.
- **Khả năng hỗ trợ của thư viện đã cài**: Trên `diffusers` `0.35.2` local, lớp `StableDiffusionInpaintPipeline` kế thừa `IPAdapterMixin`, nhận tham số `ip_adapter_image` và hỗ trợ nhánh UNet 9-channel. Khi `feature_extractor=None` (như trong [../app/outpaint_runner.py](../app/outpaint_runner.py)), loader của IP-Adapter tự tạo `CLIPImageProcessor` độc lập với `safety_checker`. Khi tích hợp, bắt buộc phải nạp adapter và image encoder **trước khi** gọi `pipe.enable_model_cpu_offload()`.
- **Giới hạn bằng chứng**: Việc mã nguồn thư viện hỗ trợ đường gọi **chưa chứng minh** chất lượng ảnh đầu ra, mức tiêu thụ VRAM thực tế hay việc tổ hợp weights hoạt động đúng trên GPU mục tiêu.

### Đề xuất kỹ thuật cho PoC (Proposals — chưa code, chưa chạy)
- **Tiền xử lý ảnh tham chiếu (Letterbox thay vì Center Crop)**: Mặc định `CLIPImageProcessor` thực hiện `center crop` về kích thước `224x224`, có nguy cơ âm thầm cắt mất phần đầu/thân nhân vật anime nằm sát mép. Đề xuất trong PoC là tiền xử lý **letterbox** vùng ảnh nguồn quan sát được (observed cropped source) thành hình vuông có ghi nhận rõ màu đệm (pad color) trong manifest trước khi đưa vào CLIP.
- **Chống rò rỉ dữ liệu tham chiếu**: Tuyệt đối **không dùng ảnh full reference** (ảnh gốc trước khi cắt giả lập còn chứa vùng bị giấu) làm đầu vào conditioning.

---

## 3. Pin Phiên bản Chính thức, Kích thước, Mã băm và Giấy phép

Thông tin pin dưới đây được Codex truy vấn qua public API `https://huggingface.co/api/models/h94/IP-Adapter/revision/main?blobs=true` vào ngày `2026-11-12` (chỉ full revision từ API công khai sau khi người dùng cho phép mới xác lập đầy đủ các trường này):

- **Repository**: `h94/IP-Adapter`
- **Pinned Revision (full SHA)**: `018e402774aeeddd60609b4ecdb7e298259dc729`
- **Thư mục ứng viên khi được duyệt cài đặt**: `.local-artifacts/outpainting/ip-adapter-sd15/` (giữ nguyên cấu trúc đường dẫn con `models/...`).

### Bảng kê tệp Runtime cần thiết

| Đường dẫn tệp (`models/...`) | Kích thước (bytes) | Loại mã kiểm tra | Giá trị Hash chính thức |
| :--- | ---: | :--- | :--- |
| `models/ip-adapter-plus_sd15.safetensors` | `98183288` | LFS `sha256` | `a1c250be40455cc61a43da1201ec3f1edaea71214865fb47f57927e06cbe4996` |
| `models/image_encoder/model.safetensors` | `2528373448` | LFS `sha256` | `6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030` |
| `models/image_encoder/config.json` | `560` | Git blob `SHA1` | `9c841dd0314e27141ed01a24df1734e407a95c4a` *(không phải SHA256 nội dung/LFS)* |

- **Tổng dung lượng 3 tệp runtime**: `2626557296` bytes (`2.63 GB` / `2.45 GiB`), chưa tính các tệp notices/metadata.
- **Ngân sách ổ đĩa đề xuất**: Cần tối thiểu **`6 GiB`** dung lượng đĩa trống cho staging và lưu ảnh/manifest đầu ra (đây là ngân sách lưu trữ đĩa, **không phải** số đo VRAM).

### Liên kết Pinned `blob` / `resolve` và Mã nguồn Tham chiếu
- `models/ip-adapter-plus_sd15.safetensors`:
  - Blob: `https://huggingface.co/h94/IP-Adapter/blob/018e402774aeeddd60609b4ecdb7e298259dc729/models/ip-adapter-plus_sd15.safetensors`
  - Resolve: `https://huggingface.co/h94/IP-Adapter/resolve/018e402774aeeddd60609b4ecdb7e298259dc729/models/ip-adapter-plus_sd15.safetensors`
- `models/image_encoder/model.safetensors`:
  - Blob: `https://huggingface.co/h94/IP-Adapter/blob/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder/model.safetensors`
  - Resolve: `https://huggingface.co/h94/IP-Adapter/resolve/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder/model.safetensors`
- `models/image_encoder/config.json`:
  - Blob: `https://huggingface.co/h94/IP-Adapter/blob/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder/config.json`
  - Resolve: `https://huggingface.co/h94/IP-Adapter/resolve/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder/config.json`
- Mã nguồn `diffusers` `v0.35.2` chính thức: `https://github.com/huggingface/diffusers/tree/v0.35.2`

### Đính chính dữ liệu sai và Quy tắc Setup
- **Cảnh báo mã băm sai**: Báo cáo tra cứu trước của Anti (`web-96991fdd889040aa`) ghi hash của `ip-adapter-plus_sd15.safetensors` là `1c9edc21...` — giá trị này **SAI**. Phải thay thế bằng mã LFS `sha256` `a1c250be40455cc61a43da1201ec3f1edaea71214865fb47f57927e06cbe4996` từ API chính thức, tuyệt đối không tái sử dụng hash sai. Kích thước và mã băm của `models/image_encoder/model.safetensors` đã được xác nhận khớp ở lượt kiểm tra tiếp theo.
- **Quy tắc setup khi được duyệt**: Chỉ tải trực tiếp qua đường dẫn `resolve/018e402774aeeddd60609b4ecdb7e298259dc729/<file>`, kiểm tra khớp chính xác số byte và LFS `sha256`; riêng `config.json` phải tính thêm `SHA256` nội dung tệp local và đối chiếu Git blob `SHA1` đúng thuật toán (`blob 560\0...`) nếu kiểm tra `blobId`. Lưu đầy đủ metadata chính thức, model cards, notices và manifest; không ghi đè bằng chứng cũ, không dùng tệp pickle/`.bin`, không tải FaceID/InsightFace, không dùng SDXL và không thay base model.
- **Nguồn giấy phép (License metadata)**:
  - Model card `h94/IP-Adapter` (`https://huggingface.co/h94/IP-Adapter/raw/018e402774aeeddd60609b4ecdb7e298259dc729/README.md`) khai báo giấy phép **`Apache-2.0`** và dẫn chiếu encoder SD1.5 tới `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`.
  - Model card `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (`https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/raw/main/README.md`) khai báo giấy phép **`MIT`**.
  - Base model SD1.5 giữ nguyên **`CreativeML OpenRAIL-M`**. Lưu ý đây là metadata từ model card, chưa phải kiểm toán pháp lý hoặc phân phối lại (exhaustive redistribution/legal audit) toàn diện.

---

## 4. Kế hoạch Thực nghiệm ANIME PoC (Đề xuất — CHƯA code, CHƯA setup)

> **Lưu ý trạng thái**: Toàn bộ nội dung mục này là kế hoạch thiết kế PoC chờ phê duyệt. **Chưa viết code lab mới, chưa tải hay setup weights, và chưa chạy GPU** trong lượt này.

1. **Hạ tầng và kiểm soát an toàn**:
   - Thực hiện trong script lab riêng sau khi người dùng duyệt tải weights; tái sử dụng `placement_poc.prepare` / `restore` và quy ước khóa GPU chung (`single_instance`) cùng thư mục output chống ghi đè của `quality_gate`.
   - Giữ `safety_checker=None`, `requires_safety_checker=False` (bộ xử lý ảnh CLIP dùng cho conditioning tham chiếu, không phải bộ phân loại nội dung).
   - Chạy hoàn toàn offline (`local_files_only=True`, `use_safetensors=True`, `torch.float16`), không tự cài thêm hoặc nâng cấp gói phụ thuộc.
   - **Chưa dùng ControlNet ở giai đoạn đầu** nhằm cô lập riêng hiệu quả của conditioning tham chiếu (IP-Adapter).
2. **Cấu hình đối chứng (A/B Recipes)**:
   - So sánh **Baseline không dùng adapter** với **IP-Adapter (`ip-adapter-plus_sd15.safetensors`) ở mức `scale=0.5`**.
   - Cố định các tham số còn lại: suy luận **single-pass** dùng bộ `soft-prompt` (positive/negative nối nhân vật từ `region_gpu.py`), cùng `canvas`/`mask`, `CFG=7.0`, `30` steps và cùng `seed`. Ảnh tham chiếu chỉ lấy từ phần crop quan sát được đã qua bước letterbox vuông có ghi nhận màu đệm.
3. **Thứ tự thực thi và bảo vệ tiến trình**:
   - **Bước 1 — Smoke test**: Chạy trước ca `winter-bottom` với `seed=42` tạo 2 output (baseline vs IP-Adapter `scale=0.5`), đặt outer timeout `300s` và dùng khóa GPU chung. Nếu ứng dụng cá nhân đang giữ khóa GPU thì dừng lab ngay, không ép dừng ứng dụng.
   - **Bước 2 — Ma trận đầy đủ (nếu smoke ổn định)**: Chạy `2` ảnh anime cũ (`closeup`, `winter`) × `4` hướng cắt (`left`, `top`, `right`, `bottom`) × `2` seeds (`42`, `123`) × `2` recipes = **`32` output**.
4. **Dữ liệu ghi nhận, Self-checks và Đánh giá**:
   - Lưu manifest từng lượt gồm: thông tin source, bounding box, prompt/negative, mask, thông số preprocess letterbox, phiên bản/hash weights, thời gian thực thi, và `Torch allocated peak` + `Torch reserved peak` (không gọi là VRAM toàn GPU).
   - Ghép composite vùng source sau biến đổi đúng vào bounding box; `assert` mã băm nguồn bất biến, khớp pixel-exact trên hình chữ nhật nguồn, và giải mã hợp lệ PNG `1920x1080`. Xuất ảnh overview kèm crop `100%` tại đường nối (seam).
   - Chấm điểm trực quan (bao gồm cả các ca kém) theo các trục: đồng nhất nhân vật (identity), số lượng chủ thể (count), giải phẫu (anatomy), nét vẽ (linework), màu sắc (colors), vật thể lạ (đồ lạ) và đường nối (seam). Không dùng PSNR/SSIM do thiếu ground truth và không hứa tái tạo đúng chi tiết thực tế nằm ngoài khung hình gốc.
   - **Điều kiện chọn recipe và bước tiếp**: Không chọn mặc định mới chỉ dựa trên 2 ảnh cũ; nếu kết quả khả quan cần kiểm chứng trên tập **anime holdout mới** đa dạng phong cách và tỷ lệ. Chỉ xét thêm ControlNet ở giai đoạn sau nếu IP-Adapter cải thiện ngoại hình nhưng bố cục vẫn lỗi (vì ảnh nguồn bị cắt không chứa sẵn pose ngoài khung). Không sửa UI/API/queue/cache ở scope lab; nếu tích hợp tương lai phải tách `recipe cache version` riêng.
