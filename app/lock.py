from contextlib import contextmanager
import msvcrt

from app.engine import ROOT


@contextmanager
def single_instance():
    path = ROOT / ".local-artifacts/haups-gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if path.stat().st_size == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError(
                "HaUps đang chạy. Hãy dùng cửa sổ hiện tại thay vì mở thêm server."
            ) from error
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
