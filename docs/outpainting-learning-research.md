# Báo Cáo Nghiên Cứu: Mở Rộng Nhân Vật Bị Cắt Ở Mép Ảnh Và Hướng Học Cải Thiện Cục Bộ

- **Ngày thực hiện:** `2026-11-12` | **Tính chất:** Báo cáo nghiên cứu (`docs-only`, không sửa code/test/model/venv/hook, chưa tải hay chạy model mới)
- **Tài liệu nội bộ:** [README.md](../README.md), [docs/anime-detail-wallpaper-research.md](anime-detail-wallpaper-research.md), [app/outpaint_runner.py](../app/outpaint_runner.py), [app/wallpaper.py](../app/wallpaper.py), [tests/test_wallpaper.py](../tests/test_wallpaper.py), [tests/test_outpaint_runner.py](../tests/test_outpaint_runner.py)

## 1. Phương Pháp Nghiên Cứu Và Phạm Vi Bài Toán

### Phương pháp đối chiếu
Anti tổng hợp tài liệu gốc trên web (arXiv, Hugging Face Diffusers, model card); Codex đối chiếu trực tiếp mã nguồn cùng môi trường `diffusers 0.35.2` cục bộ và loại bỏ các kết luận sai trước đó về cơ chế giữ pixel bằng mask, số kênh `4` so với `9` của UNet, lớp `conv_in` khi gắn LoRA, và sự tồn tại của bộ kiểm thử. Các nguồn web chưa được audit độc lập toàn bộ và mô hình mới chưa chạy thử nghiệm. Đây là tài liệu nghiên cứu, không tự chốt kiến trúc hay tuyên bố đã tích hợp.

### Phạm vi người dùng đã chốt
- **Mục tiêu mở rộng:** Chỉ vẽ tiếp nhân vật hiện hữu bị cắt ở **mép ảnh** (trên/dưới/trái/phải) và cảnh vật, đồ vật, trang trí liên quan; không sửa vùng bị che bên trong ảnh gốc; không thêm người mới hay đồ vật lạc đề.
- **Bảo toàn & Riêng tư:** Giữ nguyên ảnh gốc; chạy local hoàn toàn, không dùng cloud hay API trả phí, không đưa ảnh ra ngoài.
- **Ràng buộc bắt buộc:** **TUYỆT ĐỐI KHÔNG thêm bộ lọc nội dung** (điều kiện bắt buộc, không có tùy chọn bật lại và không khuyến nghị dưới bất kỳ tên gọi nào khác); các kiểm tra tệp và tài nguyên hệ thống vẫn giữ nguyên. Lượt nghiên cứu này không dùng ảnh mẫu cá nhân, không sinh ảnh mới và không chạy benchmark.

## 2. Dữ Liệu Hệ Thống Local Đã Đối Chiếu

| Hạng mục | Trạng thái thực tế trong repo | Hệ quả kỹ thuật |
| :--- | :--- | :--- |
| **Runner & Phần cứng** | [app/outpaint_runner.py](../app/outpaint_runner.py) dùng `StableDiffusionInpaintPipeline` (SD 1.5 inpainting local), `diffusers 0.35.2`, `torch 2.8.0+cu128`, `RTX 4050 Laptop 6GB`. | Suy luận `768x432` rồi nâng `Lanczos` + `restore` về `1920x1080`, `seed=42`, `30` steps, `CFG=7`; `model/unet` có `in_channels = 9`. |
| **An toàn & Học** | Đặt `safety_checker=None`, `feature_extractor=None`, `requires_safety_checker=False`. Không có `optimizer`, `train`, `LoRA` hay `feedback`. | Tuân thủ yêu cầu không bộ lọc nội dung; mô hình tĩnh, không tự học sau mỗi lượt chạy. |
| **Bố cục `prepare`/`restore`** | [app/wallpaper.py](../app/wallpaper.py) dùng `contain` căn giữa (`center`). Bước `restore` ghép đè giữ pixel nguồn **sau** `transform` đã chốt. | Ảnh dọc phủ kín chiều cao nên **không có vùng trên/dưới** để vẽ đầu/chân bị cắt. Riêng mask không bảo đảm giữ pixel gốc qua `VAE`. |
| **Prompt hiện tại** | Negative `'person, people, face, body, hands...'` và positive nền trống (`empty background`). | Là hướng dẫn mềm (không phải bộ lọc nội dung) nhưng đi ngược mục tiêu vẽ tiếp bộ phận cơ thể bị cắt. |
| **Queue Cache** | Chỉ kiểm tra `prompt` + tệp tồn tại (`file exist`), thiếu `config signature`. | Đổi `model`/`seed`/`mask`/`placement`/`adapter` cần đánh version cache và giữ output cũ đến khi có output mới hợp lệ (chưa sửa queue lúc này). |
| **Tests & 34 lượt GPU cũ** | Repo **có** [tests/test_wallpaper.py](../tests/test_wallpaper.py) và [tests/test_outpaint_runner.py](../tests/test_outpaint_runner.py) (lượt quét chỉ đọc `app/` trước đây bị sót). 34 lượt GPU cũ = `10` `CFG 7/4` + `12` `positive/negative` + `12` `negative`. | 34 lượt cũ đạt decode, vùng nguồn và kích thước `1920x1080`, nhưng vẫn sinh thêm nhân vật hoặc vết nối (`seam`); **không được gọi là 34 test chất lượng đạt**. |

**Ví dụ hình học tại [app/wallpaper.py](../app/wallpaper.py):** Với ảnh dọc `768x1024` đưa vào `1920x1080`, phép `contain` căn giữa cho vùng ảnh chính `810x1080` (không có margin dọc). Nếu chừa `20%` chiều cao cho vùng mở rộng dọc thì vùng ảnh chính thu về `648x864` (chỉ là ví dụ tính toán minh họa, không phải mặc định hay mẫu đã chạy). Muốn vẽ tiếp hướng nào phải chừa margin hướng đó và dùng **cùng một transform** ở cả `prepare` và `restore`.

## 3. Sáu Kết Luận Và Đề Xuất Kỹ Thuật Trọng Tâm

### 3.1. Mô hình hiện không tự học sau mỗi ảnh
- **Suy luận tĩnh:** Pipeline hiện tại chỉ chạy suy luận thuận, không cập nhật trọng số (`weights`) sau mỗi ảnh.
- **Tìm kiếm khác với cập nhật trọng số:** Việc tìm kiếm `seed`, `config`, hoặc `conditioning` kèm thuật toán xếp hạng bên ngoài chỉ là suy luận/xếp hạng lúc chạy (`inference-time search`), không cập nhật weights. `IP-Adapter` và `ControlNet` khi chạy inference cũng không tự fine-tune mô hình.
- **Giới hạn suy diễn ngoài khung:** Mô hình không biết chính xác tuyệt đối giải phẫu (`anatomy`) thật nằm ngoài khung cắt; nó chỉ suy diễn hợp lý theo ngữ cảnh biên, nên không hứa hẹn chính xác 100% hay không bao giờ xuất hiện chi tiết lạ.

### 3.2. Ưu tiên P0: Sửa Placement + Mask theo hướng mở rộng và Sinh 2 lượt có điều khiển
- **Chừa margin hướng cắt và bảo vệ vùng ảnh gốc:** Cần sửa `placement` và `mask` tại [app/wallpaper.py](../app/wallpaper.py) theo hướng bị cắt, đồng thời bảo vệ nguyên vẹn hình chữ nhật ảnh gốc (`original rectangle`) bằng bước ghép đè sau transform đồng bộ trên nền `StableDiffusionInpaintPipeline` chuẩn của `diffusers` `v0.35.2` (https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py).
- **Đề xuất sinh 2 lượt có điều khiển (Two-pass Generation) thay vì cấm toàn bộ cơ thể:**
  1. **Lượt 1 (Vẽ tiếp nhân vật hiện hữu):** Mở mask riêng tại vùng biên cắt của nhân vật, gắn với điều kiện biên, dáng (`pose`) và tham chiếu từ nhân vật gốc.
  2. **Lượt 2 (Vẽ bối cảnh ở mask còn lại):** Sau khi khóa vùng gốc và phần nhân vật vừa nối, vẽ tiếp cảnh vật/đồ vật ở phần mask còn lại và hạn chế sinh thêm người mới.
- **Giữ trải nghiệm đơn giản:** Người dùng vẫn chỉ bấm **1 nút duy nhất** và không cần nhập prompt.
- **Giả thuyết mở rộng từng bước nhỏ (Progressive Small Steps):** Đây chỉ là **giả thuyết cần kiểm thử**, vì qua nhiều bước nhỏ lỗi hình học và lệch màu có thể tích lũy dần chứ **chưa được chứng minh mặc định khắc phục được vết nối (`seam`)**.

### 3.3. Ưu tiên P1: Điều kiện hóa bằng IP-Adapter và ControlNet trên Diffusers 0.35.2
- **Vai trò điều kiện hóa:** `IP-Adapter` (Ye et al., 2023, https://arxiv.org/abs/2308.06721 ; tài liệu Diffusers: https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter) giữ đặc trưng ngoại hình và phong cách nhân vật gốc. `ControlNet` (Zhang et al., 2023, https://arxiv.org/abs/2302.05543 ; tài liệu Diffusers: https://huggingface.co/docs/diffusers/using-diffusers/controlnet) ràng buộc bố cục qua `pose`, `depth`, hoặc `edges`.
- **Bản đồ điều khiển không chứa sẵn vùng ngoài khung:** Bản đồ trích xuất trực tiếp từ ảnh gốc không có dữ liệu `pose`/biên ở vùng đã bị cắt mất ngoài ảnh, nên không phải nguồn sự thật có sẵn cho vùng cắt. Cần bước lập kế hoạch/ước đoán ngoại suy hoặc tạo dẫn hướng tự động ở vùng mới — **không khẳng định bắt buộc người dùng phải tự vẽ tay**.
- **Xác nhận hỗ trợ UNet 9 kênh trong `diffusers v0.35.2`:** Mã nguồn `StableDiffusionControlNetInpaintPipeline` tại `v0.35.2` (https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint.py) kế thừa `IPAdapterMixin` và **hỗ trợ nhánh Main UNet 9 kênh**: nhánh `ControlNet` nhận `4` kênh latent, còn nhánh `Main UNet` nhận `4 + 1 + 4 = 9` kênh sau `concat`. Do đó, báo cáo **loại bỏ hoàn toàn nhận định sai rằng pipeline này chỉ chạy được 4 kênh**. Lưu ý: không nạp chéo checkpoint `SDXL` vào `SD 1.5`, và việc code hỗ trợ không đồng nghĩa mọi tổ hợp weights đều đã được benchmark.
- **Phân biệt Vision Processor và Safety Checker:** Bộ `CLIP` image encoder / preprocessor của `IP-Adapter` (https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter) là thành phần `conditioning`, khác hoàn toàn `StableDiffusionSafetyChecker`. Nếu PoC cần nạp vision processor này thì tuyệt đối không được tự thêm safety classifier hay bộ lọc nội dung.

### 3.4. Bảng Shortlist Mô hình, Giấy phép và Ràng buộc 6GB VRAM

| Phương án / Mô hình | Họ Pipeline & Mức thay đổi | Giấy phép (License) | Đánh giá trên Windows RTX 4050 6GB & Ghi chú |
| :--- | :--- | :--- | :--- |
| **Giữ `SD 1.5 Inpainting` + `IP-Adapter` / `ControlNet`** | `StableDiffusionControlNetInpaintPipeline` (https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint.py); **hướng ít phá code nhất**, giữ UNet 9 kênh. | Theo base weights SD 1.5 hiện tại + license riêng của adapter. | Base đã chạy local; gắn thêm adapter chưa có số đo chính thức trên đúng máy đích nhưng ưu tiên thử đầu tiên. |
| **`Lykon/dreamshaper-8-inpainting`** (https://huggingface.co/Lykon/dreamshaper-8-inpainting) | Cùng họ `SD 1.5` (`StableDiffusionInpaintPipeline`, 9 kênh); mức thay đổi thấp. | `CreativeML OpenRAIL-M`. | Ứng viên cùng cỡ SD 1.5 nhưng **không bảo đảm mặc định tốt hơn** nếu chưa sửa placement/mask. |
| **`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`** (https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | Họ `SDXL` (`StableDiffusionXLInpaintPipeline`); mức thay đổi trung bình–cao. | `CreativeML OpenRAIL++-M`. | Chưa có số liệu chính thức cho Windows RTX 4050 6GB; nặng hơn SD 1.5 đáng kể. |
| **`TencentARC/BrushNet`** (https://github.com/TencentARC/BrushNet) | Nhánh kép (`dual-branch`) + `base model` riêng; mức thay đổi cao. | Code/nhánh upstream khai báo `Apache-2.0`; **base weights có license khác, chưa audit từng weight/revision phụ**. | Chưa có số liệu chính thức cho Windows RTX 4050 6GB; tốn thêm bộ nhớ cho nhánh kép. |
| **`black-forest-labs/FLUX.1-Fill-dev`** (https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) | `FluxFillPipeline` quy mô `12B` tham số; mức thay đổi rất cao. | `FLUX.1 [dev] Non-Commercial License`. | Quy mô `12B` rất lớn nên **không ưu tiên ở giai đoạn đầu**. |

- **Trung thực về VRAM 6GB:** Các nguồn rà soát **không có số liệu chính thức riêng cho Windows RTX 4050 Laptop 6GB**, nên không suy ra "bất khả thi (`impossible`)" hay "chắc chắn vừa". Các kỹ thuật tối ưu bộ nhớ của Diffusers (https://huggingface.co/docs/diffusers/optimization/memory) như `offload`, `fp16`, `tiling` là tùy chọn PoC, không tuyên bố `tiling` luôn bắt buộc hay ước lượng VRAM cố định.
- **Giấy phép và ghim phiên bản:** License của SDK không đồng nhất với license của weights; không dùng bộ lọc nội dung không có nghĩa đã audit hết mọi ràng buộc license. Phải ghim `revision` / `hash` và kiểm tra license cụ thể trước khi tải; hiện **chưa chọn sẵn `winner`**.

### 3.5. Ưu tiên P2: Phản hồi cục bộ (Local Feedback), Bộ chọn cấu hình và Huấn luyện LoRA
- **Phản hồi cục bộ và chọn cấu hình:** Lưu `recipe`, `source hash`, `mask`, `seed`, `output` và đánh giá người dùng (`tốt`/`xấu` + lý do); **không train trực tiếp trên mọi ảnh do AI vừa sinh**. Bộ chọn cấu hình (`config selector`) học ưu tiên tham số từ lịch sử phản hồi là cơ chế khác với việc cập nhật `base weights`.
- **Tác vụ LoRA `Crop-to-Full`:** Sau khi có dataset chuẩn, huấn luyện `LoRA` (https://huggingface.co/docs/diffusers/training/lora) theo hướng **`crop-to-full`**: lấy ảnh đầy đủ, che mask ở mép mô phỏng nhân vật/cảnh bị cắt để học khôi phục vùng biên. Dữ liệu cần đa dạng và chia `train` / `holdout` theo **ảnh và nhân vật** để tránh rò rỉ (`leak`).
- **DreamBooth và Diffusion-DPO:** `DreamBooth` (Ruiz et al., 2022, https://arxiv.org/abs/2208.12242) / `Subject LoRA` học `identity/style` chứ không tự học giải phẫu hay outpaint; nguy cơ sinh lặp nhân vật là **suy luận nghiệp vụ**, không tuyên bố bài báo gốc đã đo cho bài toán này. `Diffusion-DPO` (Wallace et al., 2023, https://arxiv.org/abs/2311.12908) học từ cặp ưu tiên (`preference pairs`) có nguồn gốc rõ ràng nhưng nặng, **không làm bước đầu trên GPU 6GB**.
- **Lưu ý UNet 9 kênh khi train:** Vì UNet hiện đã `9` kênh, **không cần đổi `conv_in` để gắn LoRA**; tuy nhiên vòng lặp train phải chuẩn bị đúng `9-channel mask latents` và `loss` (đổi `4 → 9` chỉ khi chuyển từ base `4` kênh). Kho Diffusers **có** ví dụ chính thức `examples/research_projects/dreambooth_inpaint` (https://github.com/huggingface/diffusers/tree/main/examples/research_projects/dreambooth_inpaint); dù vậy không giả định script Text-to-Image LoRA chạy nguyên bản cho inpaint 9 kênh và **không hứa Windows 6GB train vừa chỉ vì inference offload chạy vừa**.

### 3.6. Lộ trình PoC ngắn có khả năng bác bỏ (Falsifiable PoC) và Quản trị
- **Các bước đối chứng:** So sánh tuần tự từng yếu tố: `Baseline cũ` → `Placement mới + Mask chừa margin` → `Tách Mask 2 lượt` → `Conditioning (IP-Adapter / ControlNet) + Checkpoint ứng viên`.
- **Tập thử nghiệm và Ground Truth (`GT`):** Dùng **5 nhóm ảnh cũ** cộng **tập `holdout` mới**, bao phủ cắt biên **4 hướng** và nhiều tỷ lệ, chạy **`> 1 seed`**, có bài kiểm tra chung `crop 100%`. Cặp `full → crop` chỉ cho `GT` vùng ngoài ở **tập giả lập**; **ảnh thực tế bị cắt sẵn không có `GT` vùng ngoài**.
- **Tiêu chí chấm chất lượng (KHÔNG phải bộ lọc nội dung):** Giữ nguyên 100% pixel nguồn sau transform; dáng/giải phẫu tự nhiên; đúng `identity` và `subject count`; không có đồ vật lạ; biên ghép (`seam`) và màu sắc mượt; đảm bảo giới hạn `VRAM` và không `timeout`; `source/output cache` đánh version chuẩn xác.
- **Quản trị an toàn:** Mọi cơ chế học tự động phải có phiên bản (`versioning`), đối chứng, cổng kiểm duyệt (`gate`) và `rollback`; **không `auto-download latest` hay `auto-train` mù quáng**. Hiện chỉ dừng ở báo cáo nghiên cứu, chưa cài hay chạy model mới.
