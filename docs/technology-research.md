# Nghiên cứu công nghệ HaUps: upscale ảnh offline trên Windows

Ngày tra cứu: **11/11/2026**. Phạm vi: đối chiếu tài liệu, mã nguồn và release chính thức; không tải binary/model, không cài môi trường, không chạy GPU và không đưa ảnh cá nhân lên dịch vụ ngoài. Đây là **khuyến nghị nghiên cứu**, không tự thay thế BRD hay quyết định kiến trúc đã thống nhất.

## Kết luận khả thi

**Có cơ sở để làm MVP** với Python 3.11, FastAPI/Uvicorn, HTML/CSS/JS thuần, Pillow và tiến trình Real-ESRGAN NCNN Vulkan riêng. Không có thành phần nào trong luồng này bắt buộc tài khoản, API trả phí, cloud hoặc database. Tuy nhiên, **chưa đủ bằng chứng để khẳng định tốc độ, chất lượng hay độ ổn định trên chiếc RTX 4050 thực tế**. Cần kiểm chứng binary, driver và ảnh mẫu trước khi chốt các chỉ tiêu đó.

- **Bằng chứng:** upstream cung cấp bản chạy Windows portable, dùng GPU Intel/AMD/NVIDIA và không yêu cầu CUDA/PyTorch. [README Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN#portable-executable-files-ncnn).
- **Bằng chứng:** Pillow có phiên bản hỗ trợ Python 3.11. [Ma trận Python/Pillow](https://pillow.readthedocs.io/en/stable/installation/python-support.html).
- **Bằng chứng:** FastAPI nhận file qua `UploadFile`, dùng file spool; multipart cần `python-multipart`. [Tài liệu upload FastAPI](https://fastapi.tiangolo.com/tutorial/request-files/).
- **Khuyến nghị:** phục vụ UI và API cùng origin; bind rõ `127.0.0.1`, một worker, một tác vụ GPU tại một thời điểm. Uvicorn ghi rõ `0.0.0.0` mở truy cập mạng LAN. [Thiết lập Uvicorn](https://github.com/encode/uvicorn/blob/master/docs/settings.md#socket-binding). Các tài nguyên UI phải nằm local; không dùng CDN/font/analytics ngoài nếu yêu cầu chạy offline hoàn toàn.
- **Suy luận:** filesystem cho uploads/outputs/temp đủ cho một người dùng, nhưng phải chốt quy tắc giữ/xóa file, lỗi hết đĩa, khôi phục sau restart; không thể suy từ “không database” rằng trạng thái công việc sẽ tự bền vững.

## Binary Windows: có thật, nhưng bản được giới thiệu đã cũ

**Bằng chứng:** trang asset chính thức của Real-ESRGAN `v0.2.5.0` liệt kê `realesrgan-ncnn-vulkan-20220424-windows.zip`, 43,4 MB, phát hành asset ngày 24/04/2022. Đến ngày tra cứu, gói này đã hơn bốn năm tuổi. [Danh sách asset](https://github.com/xinntao/Real-ESRGAN/releases/expanded_assets/v0.2.5.0), [liên kết gói Windows](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip).

Repo NCNN riêng đang hiển thị release `v0.2.0` là Latest; asset Windows của release đó là gói khác, 2,06 MB, cũng ngày 24/04/2022. Không đánh đồng hai ZIP hoặc mặc định gói nhỏ đã có đầy đủ weights. [Releases NCNN](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases), [asset v0.2.0](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/expanded_assets/v0.2.0).

**Giới hạn xác minh:** mới xác nhận metadata/link chính thức, chưa tải ZIP nên chưa kiểm tra tính toàn vẹn, inventory bên trong, antivirus hay khả năng chạy trên máy đích. Gói cũ không tự chứng minh có lỗi, cũng không chứng minh tương thích driver hiện tại.

**Khuyến nghị trước triển khai engine:** người triển khai lấy đúng gói upstream; ghi URL, ngày, SHA-256 tính tại máy, inventory `.exe`/`.param`/`.bin` và license; thử `-h`, một ảnh nhỏ và kiểm tra ảnh kết quả thực sự đọc được. Hash tự tính chỉ dùng nhận dạng bản đã chọn, không thay chữ ký/hash xác thực từ nhà phát hành. Không trộn binary/model từ nguồn không rõ.

## Model và hệ số: phân biệt x2 đầu ra với model x2

**Bằng chứng từ model zoo:** `RealESRGAN_x4plus` dành cho ảnh tổng quát, native x4; `RealESRGAN_x2plus` là weights PyTorch native x2; bản anime 6B dành cho minh họa anime, native x4. `realesr-animevideov3` là mạng nhỏ cho anime video. [Model zoo](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md).

| Nhu cầu | Khuyến nghị chọn model NCNN | Đường tạo đầu ra |
| --- | --- | --- |
| Ảnh chụp x4 | `realesrgan-x4plus` | Suy luận x4 |
| Ảnh chụp x2 với gói phổ biến | `realesrgan-x4plus` | Suy luận x4 rồi thu xuống đúng `2W × 2H` bằng Pillow Lanczos |
| Minh họa/anime tĩnh x4 | `realesrgan-x4plus-anime` | Suy luận x4 |
| Anime cần thử phương án nhẹ | `realesr-animevideov3` | Chọn biến thể x2/x4 tương ứng có trong inventory |

Các hàng trên là **đề xuất cần benchmark**, không phải preset chất lượng đã được chứng minh.

**Bằng chứng mã nguồn:** CLI nhận `-s 2/3/4`, nhưng chỉ `realesr-animevideov3` chọn tên file có hậu tố `-x{scale}`; model khác dùng cùng file `.param/.bin`. Vì vậy `-s 2` không biến weights ảnh x4 thành model ảnh x2. [Chọn đường dẫn model, `main.cpp` v0.2.0](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/blob/v0.2.0/src/main.cpp#L660). Mã upstream có thể khác executable đóng gói; phải kiểm chứng trên bản chọn dùng.

**Khuyến nghị:** allowlist cặp model/scale; chạy model ảnh x4 với `-s 4`, rồi post-resize khi người dùng chọn x2. Ghi rõ đây là “đầu ra x2 từ model x4”; chi phí suy luận và ảnh trung gian vẫn là x4. Không quảng bá đã có native photo x2 trong NCNN chỉ vì model zoo có file `.pth`. Cách post-resize có tiền lệ ở implementation Python: `outscale` resize sau inference bằng Lanczos. [RealESRGANer.enhance](https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py#L232).

Model zoo có `realesr-general-x4v3` nhẹ hơn, nhưng không suy ra weights đó đã được đóng gói hoặc tương thích với NCNN release đang chọn. Chỉ thêm sau khi có cặp `.param/.bin` đúng nguồn và kiểm thử riêng.

## “Balanced”, “Quality”, tile và VRAM

**Bằng chứng:** NCNN có `-x` bật TTA, `-t` điều khiển tile và `-j` điều khiển luồng; không có switch tên Balanced/Quality. Tile nhỏ giảm nhu cầu bộ nhớ, nhiều luồng có thể tăng nhu cầu bộ nhớ. [README NCNN](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan#computer-usages).

**Khuyến nghị cấu hình để thử, chưa chốt UX:**

- Balanced: cùng model theo loại ảnh, TTA tắt, tile auto ban đầu, xử lý một ảnh/lần.
- Quality: cùng model/scale, bật TTA nếu benchmark cho thấy lợi ích đáng thời gian chờ; không hứa ảnh nào cũng đẹp hơn.
- Giữ model cố định khi so TTA. Không đổi từ ảnh chụp sang model anime rồi gọi đó là chế độ tốc độ của ảnh chụp.
- Tile là điều khiển tài nguyên; tile lớn không phải bảo đảm chất lượng cao hơn. Xem riêng đường nối tile trên vùng phẳng/cạnh dài.

**Bằng chứng mã nguồn:** TTA thực hiện tám nhánh biến đổi/inference; alpha được phóng riêng bằng bicubic. [Xử lý NCNN](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/blob/master/src/realesrgan.cpp#L273). Đây là lý do dự kiến tốn thêm thời gian, không phải phép đo “chậm đúng 8 lần”.

**Đề xuất fallback có giới hạn:** chạy auto trước; nếu xác định lỗi thiếu bộ nhớ, thử lại tile `128 → 64 → 32` và `-j 1:1:1`; mỗi bước tối đa một lần, giữ nguyên model/scale/TTA. Nếu vẫn lỗi, báo thất bại rõ ràng. Không retry mọi lỗi decode/thiếu model/driver như OOM; không âm thầm hạ độ phân giải hay đổi thuật toán. Nếu muốn tắt TTA để cứu tác vụ phải hiển thị thay đổi đó. CLI này chọn GPU Vulkan; không hứa có CPU fallback tự động. [Kiểm tra GPU/tile trong mã NCNN](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/blob/master/src/main.cpp#L705).

**Suy luận tài nguyên:** tile không loại bỏ RAM cho decode/encode và output đầy đủ. Phóng gấp bốn chiều rộng/cao làm số pixel tăng 16 lần. Ví dụ RGBA 24 MP thành 384 MP cần khoảng 1,536 GB chỉ cho một buffer pixel 8-bit, chưa tính bản sao/model. Phải đặt giới hạn byte upload, pixel đầu vào, pixel trung gian/đầu ra, timeout và dung lượng trống; ngưỡng cụ thể đợi đo máy. Không giả định VRAM thực còn trống từ tên RTX 4050.

## Ranh giới Pillow: alpha, WebP, EXIF và màu

**Bằng chứng:** `ImageOps.exif_transpose` áp dụng hướng EXIF vào pixel và loại bỏ thông tin Orientation đã dùng. [Pillow ImageOps](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.exif_transpose).

**Khuyến nghị pipeline:** decode và kiểm tra nội dung thực → kiểm tra số frame/pixel → áp dụng EXIF orientation → chuẩn hóa RGB/RGBA 8-bit theo chính sách màu → ghi PNG trung gian → inference → kiểm tra decode/kích thước/alpha → post-resize nếu cần → encode kết quả. Không ghi đè ảnh gốc. Giới hạn decoder phải áp dụng trước thao tác tốn bộ nhớ; không tắt bảo vệ ảnh giải nén quá lớn chỉ để nhận mọi file.

- Alpha: PNG/RGBA cần giữ trong luồng; kiểm tra viền trên nền đen, trắng và caro. NCNN có xử lý alpha riêng, nhưng không đủ để hứa mọi file trong suốt giữ đúng viền. JPEG không có alpha: nên từ chối xuất JPEG khi có alpha hoặc yêu cầu chọn nền rõ ràng; không âm thầm đổ nền đen.
- WebP: Pillow hỗ trợ đọc/ghi khi có libwebp; hỗ trợ file nhiều frame không có nghĩa MVP hỗ trợ animation. Đề xuất từ chối WebP động trong MVP. Muốn WebP lossless phải đặt `lossless=True`; mặc định encode WebP của Pillow là lossy. [Pillow WebP](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#webp).
- Metadata: chốt trước việc giữ hay bỏ EXIF/GPS/ICC; không hứa metadata tự đi qua engine. Đề xuất bỏ metadata riêng tư, bảo toàn hướng hiển thị; ảnh có ICC cần chuyển đúng sang sRGB hoặc báo không hỗ trợ, không chỉ gắn nhãn sRGB lên pixel chưa chuyển.
- Phạm vi tối thiểu nên ghi rõ: ảnh tĩnh JPEG/PNG/WebP 8-bit. Không ngầm nhận HDR/16-bit/CMYK như ảnh chuẩn rồi làm mất dữ liệu. Các giới hạn này là đề xuất, cần so BRD trước khi thành contract.

## Chất lượng: sắc hơn không đồng nghĩa khôi phục đúng sự thật

**Bằng chứng nghiên cứu:** bài báo Real-ESRGAN nêu đường thẳng có thể bị xoắn, GAN có thể tạo artifact và suy giảm ngoài phân phối có thể còn bị khuếch đại. [Bài báo, mục 4.4](https://arxiv.org/html/2107.10833#S4.SS4).

**Suy luận cho sản phẩm:** cần cảnh báo ngắn rằng AI có thể tạo chi tiết mới; chữ nhỏ, texture tóc/da, nét kiến trúc và ảnh quá nén cần xem kỹ. Không bảo đảm phục hồi mặt đúng danh tính, chữ đúng nội dung hay chi tiết đã mất. Logo/text sắc sẵn đôi khi chỉ cần resize truyền thống; giữ bản gốc và so với Lanczos để người dùng đánh giá. Không thêm GFPGAN vào MVP như một tính năng có sẵn của NCNN; README NCNN vẫn liệt kê face restoration là việc chưa hỗ trợ.

## License: binary, thư viện và weights cần ghi riêng

**Bằng chứng:** mã wrapper NCNN có MIT và notice realsr-ncnn-vulkan; mã Real-ESRGAN gốc dùng BSD-3-Clause. Hai license cho phép dùng/phân phối theo điều kiện giữ notice/disclaimer; BSD có điều kiện không dùng tên tác giả để quảng bá nếu chưa được phép. [LICENSE NCNN](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/blob/master/LICENSE), [LICENSE Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).

**Giới hạn:** chưa kiểm kê dependency/license trong ZIP; chưa thấy thông báo license **riêng cho từng file weights** trong các trang đã đọc. Không suy từ MIT của wrapper rằng mọi weights bên thứ ba đều MIT. Không khẳng định weights bị cấm dùng; đây là khoảng trống bằng chứng về phạm vi áp dụng license.

**Khuyến nghị đóng gói:** lập bảng executable/dependency/weights với nguồn, phiên bản/hash, notice và căn cứ license tương ứng; giữ giấy phép đi cùng gói. Với weights official, lưu liên kết model zoo/release và căn cứ license của nguồn; nếu phân phối lại mà phạm vi chưa rõ thì làm rõ với upstream trước. Không thay bằng model cộng đồng không có license chỉ vì miễn phí tải.

## Khi nào đáng thử PyTorch/CUDA

**Bằng chứng:** upstream Python phụ thuộc BasicSR, torch, torchvision, OpenCV và các gói face enhancement; requirements phần lớn không khóa phiên bản. [requirements](https://github.com/xinntao/Real-ESRGAN/blob/master/requirements.txt). PyTorch cung cấp cài local Windows/CUDA và kiểm tra `torch.cuda.is_available()`; hỗ trợ Python của torch riêng lẻ không chứng minh cả chuỗi Real-ESRGAN tương thích. [PyTorch local](https://pytorch.org/get-started/locally/).

Ví dụ rủi ro cụ thể: BasicSR `v1.4.2` import `torchvision.transforms.functional_tensor`, trong khi TorchVision 0.17 đã xóa module đó. Đây là **xung đột có thể suy ra từ source/release**, chưa phải lỗi được tái hiện trên máy này. [BasicSR v1.4.2](https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/basicsr/data/degradations.py#L8), [breaking changes TorchVision](https://github.com/pytorch/vision/releases/tag/v0.17.0). BasicSR master đã dùng đường import khác; điều này không chứng minh bản package sẽ cài cũng đã có sửa.

**Khuyến nghị:** giữ NCNN là ứng viên đầu vì cách đóng gói gọn. Chỉ benchmark PyTorch trong môi trường tách biệt khi NCNN có seam/artifact không đạt, cần model ảnh native x2, cần tính năng Python cụ thể, hoặc tốc độ thực đo không đáp ứng nhu cầu. Khóa bộ Python/torch/torchvision/BasicSR/CUDA runtime/driver đã kiểm chứng; kiểm tra weights và tắt tự tải mạng khi inference. Không cài thêm CUDA/PyTorch chỉ vì RTX 4050 mang nhãn NVIDIA; cũng không kết luận CUDA luôn nhanh/đẹp hơn NCNN trước A/B cùng model và ảnh.

## Kế hoạch benchmark thị giác tại máy, chưa thực hiện

Đây là **đề xuất nghiệm thu**, không phải kết quả thử nghiệm. Dùng ảnh người dùng có quyền sử dụng, giữ toàn bộ ảnh/original/crop/log tại máy; không upload lên demo hay cloud. Tối thiểu 3 ảnh mỗi nhóm, có độ phân giải nhỏ/vừa/lớn nằm trong giới hạn dự kiến.

| Nhóm ảnh | Điều phải quan sát |
| --- | --- |
| Chân dung thật | Mắt, tóc, da, răng; độ tự nhiên và thay đổi nét mặt |
| Phong cảnh/kiến trúc | Lá cây, texture, cạnh thẳng; méo nét và đường nối tile |
| Ảnh JPEG nén mạnh/thiếu sáng | Block JPEG, noise, halo và bệt chi tiết |
| Anime/minh họa | Line art, vùng màu phẳng, chữ; so hai model anime |
| Screenshot/chữ/logo | Nội dung chữ và cạnh mảnh; tuyệt đối không chỉ chấm “sắc” |
| PNG/WebP trong suốt | Alpha bán trong suốt, viền màu trên ba nền |

1. Giữ original bất biến; ghi hash, format, kích thước lưu trữ, kích thước sau orientation, mode/alpha/frame và EXIF orientation. Thêm ảnh có EXIF quay/lật, kích thước lẻ, tên Unicode, ảnh lỗi và WebP động để kiểm tra từ chối đúng.
2. So tại cùng kích thước đích: baseline Pillow Lanczos, model phù hợp TTA tắt, cùng model TTA bật. Với ảnh chụp x2 ghi rõ x4→x2; nếu thử PyTorch native x2 thì ghi thành phương án khác. Giữ cố định tile khi so TTA, rồi thử tile riêng.
3. Lưu toàn ảnh để xem bố cục, crop tọa độ cố định tại vùng khó và vùng giáp tile; xem output ở **100% pixel**, đồng thời đối chiếu original/crop gốc ở tỷ lệ tương ứng. Không dùng preview “fit to window” làm bằng chứng duy nhất. Có ảnh tham chiếu gốc chất lượng cao thì so thêm bản downsample có kiểm soát; không thay toàn bộ tập ảnh thực bằng dữ liệu tổng hợp.
4. Kiểm tra tự động đúng `sW × sH` theo ảnh đã orientation, không crop/lệch tỷ lệ, output đọc được, alpha vẫn tồn tại khi cần, không ảnh đen/rỗng, metadata đúng chính sách. Kiểm tra rò viền alpha bằng mắt; tồn tại kênh alpha chưa đủ chứng minh đúng.
5. Ghi thời gian wall-clock riêng decode/preprocess, engine và encode/postprocess; đo một lần cold và ít nhất ba lần warm, báo median và khoảng dao động. Ghi model/hash, binary/hash, scale, tile, TTA, thread, Windows, driver, tên GPU thực, RAM/VRAM và nguồn điện/chế độ hiệu năng.
6. Lấy mẫu bộ nhớ GPU thực trong lúc chạy bằng công cụ trên máy; ghi chu kỳ lấy mẫu, peak dedicated GPU memory, mức nền và RAM của tiến trình. Nếu công cụ chỉ có tổng GPU/không gán đúng PID thì ghi rõ, không gọi số ước lượng là VRAM riêng của engine. Đo các nhánh fallback; ghi lần retry, ảnh lớn nhất thành công, lỗi và output chưa hoàn chỉnh có bị loại bỏ không.
7. Chấm theo ảnh: giữ đúng nội dung, tự nhiên, chi tiết hữu ích, artifact, seam và thời gian. Chốt Balanced/Quality sau khi xem A/B ẩn tên nếu làm được. Không đưa ngưỡng giây/VRAM hoặc câu “đẹp hơn X%” vào BRD khi chưa có kết quả. PSNR/SSIM chỉ bổ trợ nếu có ground truth; không thay đánh giá thị giác cho ảnh đời thực.

**Điểm cần chốt sau benchmark:** gói engine cụ thể, model/scale được hỗ trợ, có giữ chế độ TTA hay không, giới hạn kích thước, chính sách màu/metadata/alpha, thời gian chờ chấp nhận được và vòng đời file.

**Chưa kiểm tra:** phần cứng/driver, nội dung ZIP, license từng weights trong gói và chất lượng ảnh thực. Rủi ro còn lại: release cũ, artifact suy luận và giới hạn RAM/VRAM; mọi chỉ tiêu chất lượng/tốc độ vẫn chờ benchmark local.
