# Giữ chi tiết anime và xuất hình nền 1920×1080

## Trạng thái anime đã triển khai (2026-11-12)

> [!NOTE]
> Yêu cầu mới bỏ bộ phân loại nội dung ở tạo nền local, giữ nguyên ảnh nguồn và bảo vệ lớp ảnh chính. Model không qua huấn luyện lại hay weights mới và chất lượng còn cần gate nghiệm thu; toàn bộ phần thân tài liệu lịch sử giữ lại để tham khảo.

- **Cấu hình production:** Model NCNN `realesrgan-x4plus-anime`, scale native 4×, tile 128→64→32 khi thiếu VRAM, TTA tắt giữ nguyên. Không tích hợp AnimeSharp.
- **Tùy chọn Blend 50 (Giữ chi tiết nguồn):** Đã triển khai dưới dạng opt-in (mặc định OFF). Hòa trộn 50% native AI 4× và 50% nguồn Lanczos, giữ nguyên alpha native.
- **Đối chiếu local có trước:**
  - Đã xem 4 sheet `anime-details/app-smoke/sample{1,2}/comparison-{2,4}.png`: ở các vệt mảnh và texture, blend giữ thêm tín hiệu nguồn nhưng nét bị mềm hơn.
  - Sheet `gate/anime.png`: làm mềm cạnh tóc.
  - Bộ `anime-models/silver/{face,hair}-2x.png`: chi tiết mắt và tóc bị AI đơn giản hóa.
- **Giới hạn anime upscale:** Đối chiếu ở mục này là bằng chứng cũ, chưa khẳng định đã khắc phục việc mất má hồng. Benchmark tạo nền mới được ghi riêng bên dưới.

Ngày nghiên cứu: **2026-11-12**. Các đề xuất cũ nằm ở phần lịch sử bên dưới. Triển khai và benchmark mới nằm ở mục hiện hành; không tải thêm weights hoặc gửi ảnh ra ngoài.

## Đánh giá thực nghiệm tạo nền outpainting và bỏ classifier (Hiện hành 2026-11-12)

- **Triển khai gỡ bỏ classifier nội dung:** Theo yêu cầu người dùng bỏ classifier nội dung tạo nền local, hệ thống đã triển khai cấu hình `safety_checker=None`, `feature_extractor=None`, `requires_safety_checker=False` (requiresFalse), loại bỏ 3 file unused khỏi capabilities; vẫn duy trì validate chặt chẽ file, tài nguyên, ảnh nguồn (source), chế độ offline và hàng đợi FIFO. Chỉ dùng bộ ảnh kiểm thử đã duyệt.
- **Môi trường & cấu hình thực nghiệm:** SD 1.5 inpainting local revision `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb`, Diffusers 0.35.2, Torch 2.8.0+cu128, chạy trên RTX 4050. Quá trình infer thực hiện ở 768×432 rồi restore lớp chính lên 1920×1080 canvas; cố định seed 42, 30 steps, dùng model và mask cùng nhau.
- **Gate kiểm định & công cụ đo:** Sử dụng gate 5 nhóm crop RGB 256 cũ (portrait-face, landscape, text, anime, JPEG30), không dùng holdout mới. Script `.local-artifacts/outpainting/quality_gate.py` hỗ trợ đo sourcehash, foreground pixel, PNG size, crop seam 100%, xuất báo cáo JSON, đo thời gian và torch peak memory, xuất thư mục mới chống ghi đè.
- **Kết quả 3 lab chạy thực tế (tổng cộng 34 record):**
  - `quality-cfg-20261112` (10 record): Thử nghiệm so sánh giữa CFG 7 và CFG 4.
  - `quality-background-prompt-20261112` (12 record): Bổ sung mẫu user58 clothed native; positive background painting/photograph + empty scenery và negative expanded.
  - `quality-negative-prompt-20261112` (12 record): Chỉ dùng negative expanded so với baseline, giữ nguyên positive cũ.
  - Hai lab sau dùng module `quality_gate` với các biến toàn cục `PROMPTS`/`NEGATIVE_PROMPT`/`GROUPS` cho cấu hình thử nghiệm, không sửa production; thông số chi tiết lưu tại `trial-settings.json` từng run.
  - Tất cả 34 lượt sinh ảnh đều đạt self-check về decode, size, sourcehash và foreground exact (bảo toàn nguyên vẹn lớp chính); đây KHÔNG phải là suite hook hay chứng nhận chất lượng đạt chuẩn.
  - Thông số baseline: torch allocated peak 1992.913 MiB, reserved 2188 MiB (đây là bộ nhớ Torch cấp phát, không phải peak VRAM toàn GPU); thời gian inference 6.232–8.637 s/lượt sau khi load model (không có median hay tốc độ cam kết).
- **Kết luận quan sát chất lượng (overview 5 nhóm, crop seam 100% và user58):**
  - CFG 4 không giải quyết được lỗi extra heads hay sinh thêm nhân vật; bối cảnh biển tương đối liền mạch nhưng mẫu chân dung và text bị sinh thêm người.
  - Prompt background mới loại bỏ được nhân vật thừa trên user58 nhưng lại thêm cảnh/vật thể lệch, trên anime bị thêm chi tiết mắt kỳ dị và text biến thành pattern; cấu hình chỉ mở rộng negative prompt vẫn có nhân vật thừa.
  - Quyết định: KHÔNG đưa cả 3 cấu hình trial vào default; cấu hình production giữ nguyên prompt, negative, CFG 7, seed 42 trước đó, chỉ có classifier đã bỏ.
  - Đây là giới hạn tạo nền hiện có, không gọi là đã giải quyết chất lượng, không tự phục hồi hay tự huấn luyện. Bước tiếp kế tiếp cần thử nghiệm các ứng viên model/pipeline giữ bối cảnh tốt hơn qua nguồn, license, VRAM và gate kiểm định; không tự chốt kiến trúc hoặc tải model trong lượt này.
- **Kiểm thử thực tế user58 sau khi bỏ classifier:**
  - Kiểm tra tại `.local-artifacts/outpainting/classifier-removed-20261112-1610/` (kiểm tra giao diện browser qua 4 viewports, thao tác refresh, download và xác nhận source image được bảo toàn) và `classifier-same-input-20261112/` (thử nghiệm trên đúng ảnh native từng bị chặn cũ).
  - Xác nhận user58 đã tạo nền thành công sau khi bỏ classifier; chất lượng ảnh nền vẫn có thể xuất hiện thêm nhân vật hoặc lộ đường nối seam. Không ghi nhận full test pass hay hook đã nhận.

---

> [!NOTE]
> **Tài liệu nghiên cứu lịch sử (Tham khảo):** Toàn bộ phần thân tài liệu bên dưới (từ "Vấn đề cần giải quyết") là các phân tích, đề xuất và đối chiếu lịch sử ban đầu (2026-11-12), được giữ lại làm dữ liệu tham khảo; trạng thái triển khai và đánh giá thực nghiệm hiện hành được ghi nhận ở các mục phía trên.

## Vấn đề cần giải quyết

Người dùng đã dùng output 2×/4× và báo các vệt mồ hôi/nước, ửng hồng trên má và ánh sáng mềm trên cơ thể bị mịn, khó thấy. Mục tiêu là giữ những chi tiết vốn nhìn thấy trong nguồn, không tự vẽ thêm giọt nước, tăng hồng hoặc tăng bóng để tạo cảm giác phục hồi.

Handoff và [thử nghiệm anime](anime-model-experiments.md) xác nhận app dùng NCNN `realesrgan-x4plus-anime`, native 4×; output 2× là native 4× rồi Lanczos xuống. Bộ thử cũ đã thấy làm phẳng texture và thay đổi mi/mắt/ánh tóc. AnimeSharp trong lab giữ thêm một số texture nhưng còn viền lặp và thay đổi mắt. **Chưa có phép thử riêng xác nhận model nào giữ tốt ba loại chi tiết mới người dùng nêu.** Không suy từ PSNR cao hơn hoặc đường viền sắc hơn thành đã giải quyết vấn đề này.

## Hướng nâng chất lượng tối thiểu

**Suy luận cần kiểm:** chi tiết tương phản thấp có thể bị quá trình phục hồi của model làm mất. Riêng 2× còn có bước thu từ 4×; preview thu nhỏ cũng có thể làm chi tiết khó thấy. Vì người dùng gặp ở cả 2× và 4×, chưa đủ căn cứ quy hết cho resize. Cần so file tải xuống với ảnh nguồn, native 4× và output 2× ở cùng vùng/tỷ lệ trước khi kết luận nguyên nhân.

Không có căn cứ để coi tile lớn, TTA hoặc output 8× là cách phục hồi vệt nước/má hồng. Thử nghiệm cũ chưa thấy TTA cải thiện nhất quán; tile chủ yếu điều khiển tài nguyên. Không tự thêm sharpen, denoise mạnh hoặc phục hồi mặt.

**Đối chiếu công khai qua Antigravity:** CLI NCNN không cung cấp điều khiển denoise cho `x4plus-anime`; `-dn/--denoise_strength` của bản Python v0.3.0 áp dụng cho General `realesr-general-x4v3` bằng nội suy weights, không phải nút giảm khử nhiễu cho anime hiện hành. Nguồn: [NCNN usages](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan#full-usages), [Python inference v0.3.0](https://github.com/xinntao/Real-ESRGAN/blob/v0.3.0/inference_realesrgan.py), [release v0.3.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.3.0). Các đường nguồn này cũng có trong nghiên cứu hiện có; lượt này không mở web trực tiếp. Không dùng tên “giảm denoise” cho một phép blend ảnh hoặc giả định có flag đó trong binary.

Antigravity không tìm được bằng chứng chính thức so riêng vệt nước/má hồng cho native 2× anime-video hay blending. Vì vậy bỏ nhận định ở câu trả lời đầu rằng đó là giải pháp đã được tài liệu xác nhận. Đây chỉ là ứng viên thử; kết quả lab cũ đã cho thấy video có thể mềm viền và không thắng đều.

Thứ tự thử được đề xuất, chưa chạy trong lượt này:

1. Chọn ảnh có sẵn chi tiết nhìn thấy rõ ở nguồn; đánh dấu crop nước/mồ hôi, má hồng, highlight, kèm vùng mắt/tóc và vùng phẳng làm đối chứng. Thử thêm ảnh mới ngoài bộ pink/silver/blue cũ.
2. So cùng output 2× và 4×: Lanczos từ nguồn; model app hiện tại; AnimeSharp đã có trong lab. Lưu native 4× riêng để phân biệt mất chi tiết do inference với do thu xuống. Xem crop 100% và cùng độ lớn nội dung, không chỉ preview fit-to-window.
3. Chỉ nếu cần, thử hòa trộn output AI với nguồn được Lanczos lên đúng kích thước. Đây là giả thuyết hậu xử lý có thể giảm thay đổi so nguồn, không phải phục hồi chi tiết đã mất; có thể đưa nhiễu/JPEG/độ mềm trở lại. Không chọn tỷ lệ trộn mặc định hoặc thêm slider khi chưa A/B. Má hồng là chuyển sắc màu, nên chỉ trộn độ sáng không đủ kiểm chứng mục tiêu.
4. Chỉ nghiên cứu/tải model khác nếu các phương án đã có vẫn không đạt. Kiểm nguồn, phiên bản, backend và license weights riêng; không đổi CUDA vào app từ kết quả lab.

Tiêu chí bác bỏ một ứng viên: giọt/vệt bị xóa hoặc đổi hình, má hồng bị nhạt/đổi sắc, highlight bị phẳng; hoặc giữ được các vùng đó nhưng tạo viền kép, sai mặt, halo, seam hay nhiễu khó chấp nhận. Ảnh nén và ảnh sạch cần đánh giá riêng. Nếu không có phương án tốt hơn, giữ lựa chọn hiện hành và công bố giới hạn, không đổi mặc định để gọi là đã nâng cấp.

Trước tích hợp pipeline mới vẫn cần gate năm nhóm chân dung, phong cảnh, chữ/đồ họa, anime, JPEG/nhiễu; alpha, màu và ảnh lớn. Ghi model/hash, scale native/output, tile/TTA, crop, thời gian và RAM/VRAM thực tế. Các phép thử cũ chưa thay thế gate mới này.

## Ảnh dọc thành hình nền: phải chọn cách xử lý phần thiếu

Yêu cầu mới là phủ khung hình nền 1920×1080 và giữ hình dáng nhân vật. [Đặc tả resize hiện tại](resize.md) yêu cầu padding đen/trắng/trong suốt, giữ toàn ảnh, không crop và không thêm AI. Nó tạo đúng kích thước file nhưng **không tạo cảnh tiếp nối hai bên**. Đây là điểm khác biệt cần ghi rõ trước khi triển khai; tài liệu nghiên cứu này không tự sửa FR/AC đó.

Với ảnh khác tỷ lệ 16:9, phép resize đồng đều không thể cùng lúc giữ toàn bộ ảnh, phủ kín khung bằng nội dung sẵn có và không biến dạng. Muốn tránh kéo giãn phải chấp nhận cắt ảnh hoặc thêm nền/nội dung.

[Pillow ImageOps](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html) phân biệt `contain` giữ tỷ lệ nhưng không bảo đảm canvas đúng kích thước đích, `pad` thêm nền và `fit` resize/crop để phủ khung. Nền mờ là đề xuất ghép hai lớp từ các phép này, không phải tính năng AI hoặc một mode có sẵn của `pad`. [Nghiên cứu seam carving, SIGGRAPH 2007](https://doi.org/10.1145/1276377.1276390) mô tả resize theo nội dung; đánh giá không phù hợp làm mặc định cho ảnh anime dọc chênh tỷ lệ lớn là suy luận về rủi ro, chưa phải A/B của HaUps. Hai nguồn được Antigravity cung cấp, chưa đọc trực tiếp toàn văn trong lượt này.

| Cách xuất | Giữ dáng chủ thể | Giữ toàn ảnh | Hai bên hiển thị gì | Đánh đổi |
| --- | --- | --- | --- | --- |
| Padding màu/trong suốt | Có | Có | Màu nền hoặc nền desktop | Đúng spec hiện có nhưng còn dải nền; alpha không tự lấp thành cảnh |
| Nền mờ từ chính ảnh | Có ở lớp ảnh chính | Có ở lớp ảnh chính | Bản sao phóng lớn, crop và làm mờ | Phủ canvas, không phải cảnh mới liền mạch; có thể còn mảng bóng/màu giống nhân vật |
| Crop phủ khung | Có | Không | Nội dung ảnh phóng đồng đều | Ảnh dọc mất nhiều phần trên/dưới; cần chọn vùng giữ và preview |
| Seam carving | Không bảo đảm | Không bảo đảm nội dung/đường nét nguyên vẹn | Co/giãn theo các đường năng lượng | Có thể cong nét, lệch giải phẫu; không ưu tiên cho anime dọc sang ngang chênh lớn |
| AI outpainting local | Có thể giữ lớp gốc bằng compositing | Có thể | Cảnh mới AI tạo thêm | Cần model/pipeline khác, thời gian/tài nguyên và kiểm license; có thể sai phối cảnh, nối nền hoặc thêm chi tiết lạ |
| Kéo giãn trực tiếp | Không | Có về phạm vi ảnh | Ảnh cũ bị ép theo tỷ lệ mới | Không phù hợp mục tiêu giữ dáng nhân vật |

Ví dụ hình học, chưa phải output thực nghiệm: ảnh 1080×1920 đưa vào 1920×1080 bằng contain có nội dung khoảng **608×1080**, còn khoảng **656 px mỗi bên** sau làm tròn. Nếu crop phủ khung, ảnh cần rộng 1920 và cao khoảng 3413; chỉ giữ dải cao 1080, tương đương khoảng **31,6% chiều cao nguồn**. Nếu kéo giãn trực tiếp, tỷ lệ rộng/cao của nhân vật tăng khoảng **3,16 lần**. Vì vậy crop và stretch đều đánh đổi lớn với ảnh toàn thân dọc.

## Đề xuất phạm vi triển khai tiếp theo

**Phương án đơn giản trước:** giữ Original làm mặc định; giữ Padding theo spec; đề xuất thêm một chế độ riêng “1920×1080 — nền mờ”. Dùng ảnh chính contain nguyên tỷ lệ trên nền cover làm mờ từ cùng ảnh. Chỉ nền bị crop, không crop/kéo giãn lớp ảnh chính. Cách này không chạy thêm AI, không cần dependency mới và tránh hai dải đen; không mô tả là mở rộng cảnh.

Khi triển khai cần xác định nền RGB/RGBA: bản sao nền có alpha không tự bảo đảm phủ kín canvas, nên chế độ nền mờ cần quy tắc nền đặc rõ ràng. Transparent Padding vẫn là tùy chọn riêng. Không hứa alpha/ICC/màu được giữ tuyệt đối nếu chưa kiểm qua pipeline.

Nếu mục tiêu là cảnh liền mạch sang hai bên, hướng phù hợp hơn là **outpainting chỉ vùng nền thiếu**. Cần tạo canvas, mask vùng bổ sung, có thể cho phép vùng nối hẹp có kiểm soát, rồi compositing ảnh chính trở lại để bảo vệ phần nhân vật. Mask diffusion tự nó chưa đủ bảo đảm pixel vùng giữ nguyên không đổi sau encode/decode/resize. Đây là module sinh ảnh khác upscale, vượt yêu cầu “không chạy thêm AI” trong `resize.md`; chưa được chọn model hoặc tích hợp.

Nguồn công khai do Antigravity dẫn: [Diffusers inpainting](https://huggingface.co/docs/diffusers/using-diffusers/inpaint), [offline mode](https://huggingface.co/docs/diffusers/installation#offline-mode). Đây là hướng triển khai local có tài liệu, không chứng minh model inpainting bất kỳ phù hợp nét anime hoặc chắc chắn chạy vừa máy đích. Setup/tải model ban đầu khác inference offline. Khi áp dụng phải pin phiên bản và có đủ mọi thành phần local, không tự tải qua mạng trong lúc xử lý.

Về license, Antigravity dẫn [LICENSE Diffusers](https://github.com/huggingface/diffusers/blob/main/LICENSE) là Apache-2.0 và [model card Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) có nhãn CreativeML OpenRAIL-M riêng cho weights. Đây chỉ là đầu mối nghiên cứu, **chưa chọn checkpoint**, chưa đối chiếu revision/license đầy đủ trực tiếp trong lượt này và chưa đánh giá quyền phân phối. Không suy license thư viện thành license model. Antigravity xác nhận các nguồn trên không cung cấp benchmark Windows RTX 4050 6GB: chưa hứa vừa VRAM, tốc độ hoặc chất lượng anime; cần PoC riêng trước tích hợp.

Xuất wallpaper nên đọc trực tiếp cache native đã kiểm hợp lệ rồi resize lớp ảnh chính một lần về kích thước cần thiết, tránh chuỗi native→2×/8×→1080 gây resize lặp. Tách file xuất wallpaper khỏi output upscale gốc; không thay ảnh nguồn hoặc cache/model cũ. Khi triển khai phải kiểm callers/API, lựa chọn export, cache key, preview/download, refresh và cleanup để không trả nhầm biến thể. Chưa quyết định contract mới trong lượt nghiên cứu.

Nghiệm thu nền mờ: đúng 1920×1080; lớp chính giữ tỷ lệ/nội dung, không viền rỗng ngoài ý muốn; ảnh dọc/ngang/vuông và alpha hoạt động đúng; Original không đổi; preview/download trỏ đúng biến thể; không thêm GPU job; không ghi đè nguồn/output cũ. Kiểm bằng test hành vi và browser khi được triển khai, không coi bảng so sánh này là bằng chứng đã có tính năng.

## Nguồn và giới hạn kiểm chứng mới

Đã gửi **bốn câu hỏi công khai tuần tự** cho Antigravity: câu hỏi chung; làm rõ hình học/outpainting; kiểm chứng riêng điều khiển denoise và giới hạn bằng chứng chất lượng; bổ sung nguồn offline/license/compositing ngắn gọn. Không gửi ảnh, source, log, đường dẫn máy hay thông tin riêng. Hai phản hồi đầu bị cắt; hai câu hỏi sau được trả đủ để làm rõ các phần còn thiếu. Không tự mở trình duyệt hoặc tải web/model.

Bản trả lời đầu chỉ dẫn về trang gốc Real-ESRGAN và gợi ý native2/blend như đã có tài liệu; câu trả lời đối chiếu sau không xác minh được lợi ích riêng cho vệt nước/má hồng. Không đưa gợi ý đó thành kết luận. Một phản hồi còn đưa tag release `20220424`, khác đường release đã kiểm trong `technology-research.md`; không sử dụng tag này làm bằng chứng phiên bản binary. Lượt nghiên cứu này không kiểm trực tiếp nội dung/link web hoặc toàn văn paper; các nguồn mới được dẫn qua Antigravity như dữ liệu tham khảo, không phải audit độc lập. Bằng chứng lab có trước được dẫn riêng ở đầu tài liệu.

Đã đọc lại nội dung, kiểm UTF-8, đường dẫn tham chiếu local và các ví dụ hình học bằng PowerShell: đạt. Lệnh kiểm tài liệu bằng `py -3` ban đầu không thấy interpreter trong sandbox; thay bằng PowerShell sẵn có, không sửa/cài môi trường. Không sửa app/tests, không chạy lại suite cấu hình trước hook và chưa có trạng thái hook mới cho lượt này. Không gọi kết quả test cũ là kiểm chứng của nghiên cứu mới.
