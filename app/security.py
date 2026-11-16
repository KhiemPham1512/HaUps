from urllib.parse import urlsplit

from starlette.responses import JSONResponse


class LocalOnly:
    def __init__(self, app, allow_test_host=False):
        self.app = app
        self.hosts = {"127.0.0.1", "localhost"} | (
            {"testserver"} if allow_test_host else set()
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        host = headers.get(b"host", b"").decode("ascii", "replace")
        hostname = host.split(":")[0].lower()
        error = None
        if hostname not in self.hosts:
            error = (400, "Chỉ truy cập HaUps qua localhost.")
        elif scope["method"] == "POST":
            origin = headers.get(b"origin")
            if origin and urlsplit(origin.decode("ascii", "replace")).netloc != host:
                error = (403, "Chỉ thêm ảnh từ giao diện HaUps trên máy này.")
            else:
                length = headers.get(b"content-length", b"")
                if not length.isdigit():
                    error = (411, "Yêu cầu tải ảnh cần có dung lượng xác định.")
                elif int(length) > 1024 * 1024 * 1024:
                    error = (413, "Tổng dung lượng một lần tải tối đa 1 GB.")
        if error:
            return await JSONResponse({"detail": error[1]}, status_code=error[0])(
                scope, receive, send
            )
        await self.app(scope, receive, send)
